# Dev Notes (scratch log for the README)

Working branch: `faris/dev`. These notes record what was implemented, design decisions,
and gotchas so the final README can be written from them.

## Part 1 — Point cloud MVP
- `point_cloud.wgsl`: `clip = proj * view * pos` (model matrix is identity; the PLY is already in world space).
- Camera conventions (from `camera.ts`): view space is +Z forward (`clip.w = view.z`),
  projection flips Y, near = 0.01, far = 100.

## Part 2 — Gaussian renderer (quad rasterization)
Pipeline per frame: reset counters (`queue.writeBuffer`) → preprocess compute → radix sort →
copy `keys_size` into the indirect draw's instance count → instanced triangle-strip quads (`drawIndirect`).

### Shader organisation
- The preprocess module is assembled on the host from pieces so the f16 variant can reuse the entry points:
  `preprocess_common.wgsl` (structs, bindings, helpers) + `preprocess.wgsl` (f32 math: SH, 3D/2D covariance,
  `project_gaussian`) + `preprocess_entries.wgsl` (entry points).
- Explicit bind group layouts for preprocess (instead of `layout: 'auto'`) so one set of bind groups works for
  every preprocess entry point and unused bindings aren't stripped.

### Preprocess details
- Near-plane cull at view z < 0.2 (always on, avoids dividing by ~0); frustum cull on NDC center outside ±1.2
  (toggleable in the GUI for the performance analysis).
- Quaternion stored w-first (rot_0 = w). Σ = R S Sᵀ Rᵀ with scale = exp(s) * gaussian_multiplier.
- 2D cov via EWA: J·W·Σ·Wᵀ·Jᵀ (Jacobian clamped to 1.3× FOV like the reference CUDA rasterizer), +0.3 px² low-pass.
- radius = ceil(3·sqrt(λ_max)); quad half-size in NDC = radius·2/viewport.
- Opacity = sigmoid(raw); color from SH (degree from the file, 3 for the provided scenes); view dir = pos − cam_pos.
- Sort key = `~bitcast<u32>(view_z)` — positive floats keep order as u32, inverting makes the ascending radix sort
  give back-to-front order. `sort_dispatch.dispatch_x` is atomically bumped every 256*15 keys.
- Splats are written compacted (index from the `atomicAdd`), sort payload = splat index.
- Preprocess dispatch is 2D when it would exceed 65535 workgroups (e.g. 1M+ gaussians with workgroup size 16/32).

### Splat layout (32 B)
pos (2×f32), conic (3×f32), size (f16x2), color rg (f16x2), color b + opacity (f16x2).
Position and conic kept in f32 on purpose: conic of a large splat is ~1e-6 px⁻² (f16 denormal) and f16 NDC
positions jitter by ~0.5 px on a 1080p+ screen.

### Rendering
- Vertex: `splats[sorted_indices[instance]]`, expand corner by ±size, pass flat color/opacity/conic/center (px).
- Fragment: d = center_px − frag.xy (both y-down pixel space), power = −½(a dx² + c dy²) − b dx dy,
  alpha = min(0.99, opacity·e^power), discard if power > 0 or alpha < 1/255.
- Blend: src-alpha / one-minus-src-alpha (back-to-front "over"), no depth buffer.

### GUI additions
`frustum_culling` checkbox and `workgroup_size` dropdown (32–256, rebuilds the preprocess pipeline) for perf analysis.

### Testing setup
- Headless Chrome (puppeteer-core, `--enable-unsafe-webgpu`) uploads the PLY + cameras and screenshots the canvas.
- Scenes: bonsai.ply (272,956 gaussians), bicycle_30000.cleaned.ply (1,063,091 gaussians).
- First results: bicycle default camera looks like the reference; headless fps ~74 gaussian vs ~120 point cloud
  (vsync-capped, need proper GPU timing for the analysis).

## GPU profiling (for the performance analysis)
- `src/utils/gpu-stats.ts`: timestamp queries around passes + u32 counters copied from GPU buffers, read back
  asynchronously (frames are skipped while a readback is pending), exponentially smoothed.
- Shown in the GUI `gpu_stats` monitor: visible gaussians, preprocess / sort / render ms (sort = time between the
  end of preprocess and the start of the render pass, i.e. the radix sort + indirect copy).
- `main.ts` requests `timestamp-query` and `shader-f16` when the adapter supports them.
- Point cloud renderer also reports its render pass time.

## Extra credit — tile based rendering (`gaussian_mode: tile`)
Follows the reference CUDA rasterizer:
1. `preprocess_tiles`: project as before, compute the gaussian's screen rect in 16×16 tiles, reserve
   `n_tiles` keys with one `atomicAdd` (no prefix scan needed since everything is sorted afterwards) and write
   key = `[tile id | depth]`, payload = compacted splat index.
   - 32-bit keys only (the provided radix sort is 32 bit): tile id uses `ceil(log2(num_tiles))` high bits,
     the depth uses the remaining bits — taken as the top bits of `bitcast<u32>(view_z) << 1` (drop sign bit),
     which is a log-like quantization (≈10 mantissa bits at 1080p → ~0.1% relative depth precision).
   - Ascending sort ⇒ per tile front to back.
2. `finalize_tiles` (1 thread): clamps key count to the sort buffer capacity, writes the sort's indirect
   dispatch and the indirect dispatch for range identification.
3. Radix sort (separate sorter sized for `min(8 × num_points, 16M)` keys, created lazily on first use).
4. `identify_ranges`: tile start/end where the tile id of neighbouring sorted keys changes.
5. `rasterize`: one 16×16 workgroup per tile, one thread per pixel; splats loaded cooperatively in batches of
   256 into workgroup memory; front-to-back compositing in f32 with early termination per pixel (T < 1e-4)
   and for the whole tile (shared atomic counter of finished pixels, checked with `workgroupUniformLoad`).
   Writes an rgba8unorm storage texture, blitted to the canvas with a fullscreen triangle.
- Uniformity gotcha: the batch loop contains barriers, so the tile range is read by one thread into workgroup
  memory and broadcast with `workgroupUniformLoad` (storage loads are considered non-uniform).
- Limit gotcha: default `maxStorageBuffersPerShaderStage` is 8 → the tile counters (splat count, indirect
  dispatch, requested keys) share one buffer.
- Culling bonus: gaussians whose tile rect is empty (center within the 1.2× guard band but splat off-screen)
  are dropped entirely.

First numbers (bonsai, preset 3, 1280×800, RTX 4050 Laptop GPU, headless Chrome 155):
| mode | visible | preprocess | sort | render/raster | total |
|---|---|---|---|---|---|
| quad | 135,862 | 0.16 ms | 0.56 ms | 2.85 ms | 3.57 ms |
| tile | 114,600 (1.12M keys) | 0.34 ms | 3.64 ms | 1.25 ms | 5.29 ms |
→ rasterization itself is >2× faster, but sorting ~8× more keys dominates. Output is visually identical.

## Extra credit — f16 preprocess (`precision: f16`)
- `preprocess_f16.wgsl` is a drop-in math backend (same function names), module prefixed with `enable f16;`;
  requires the `shader-f16` device feature (dropdown disabled otherwise).
- f16: decoding (`bitcast<vec2<f16>>` straight from the packed u32s — no unpack to f32), SH evaluation
  (48 coefs, the bulk of the ALU work), quaternion → rotation, 3D covariance, sigmoid.
- Trick: 3D covariance computed as R (S/s_max)² Rᵀ so all entries are in [0,1]; s/s_max = exp(log_s − log_max)
  (one subtraction in log space). Multiplied by s_max² after converting to f32. Without this exp(scale)² of small
  gaussians (scale ≈ e⁻¹⁰) underflows f16.
- Stays f32: view/projection (world positions need the mantissa) and the EWA projection to pixel space —
  2D covariance entries reach ~1e6 px² (> f16 max 65504) and conics of large splats are f16 denormals.
- Quality (bicycle, default camera): f16 vs f32 PSNR 54.2 dB (quad) / 54.3 dB (tile); <0.01% of pixels differ by
  more than 8/255. Quad vs tile renderer: PSNR 44.1 dB (8-bit framebuffer blending vs f32 compositing + early
  termination).
- Perf: preprocess ~1.0–1.1 ms either way on bicycle (1.06M gaussians) — the step is memory bound
  (24 B gaussian + 96 B SH read per gaussian), so halving ALU precision barely matters on this GPU.
  Need a proper benchmark sweep to quantify.

Machine: AMD Ryzen 7 8845HS, NVIDIA GeForce RTX 4050 Laptop GPU (driver 616.56), Chrome 155 (adapter = nvidia/lovelace).

## Optimization — tight, opacity-aware splat bounds (`tight_bounds`, default on)
- Reference: square around the 3σ circle of the major axis (radius = 3·sqrt(λ_max)).
- Tight: axis-aligned box of the ellipse region where alpha ≥ 1/255 (everything else is discarded anyway):
  k = min(3, sqrt(2·ln(255·opacity))) standard deviations, per axis sqrt(Σ'xx), sqrt(Σ'yy)
  (exact bounding box of the ellipse). Gaussians with 255·opacity ≤ 1 are culled entirely.
- Used for both the quad size and the tile rect.
- Bicycle, default camera, 1280×800:
| | tight off | tight on |
|---|---|---|
| visible gaussians | 1,063,091 | 974,698 (−8%) |
| quad render pass | 9.78 ms | 2.11 ms (4.6×) |
| quad gpu total | 14.13 ms | 6.04 ms |
| tile keys | 5.58 M | 2.30 M |
| tile sort / raster | 12.70 / 6.31 ms | 6.79 / 3.48 ms |
| tile gpu total | 20.05 ms | 11.20 ms |
| PSNR vs off | – | 75.6 dB (quad), 70.3 dB (tile) |
