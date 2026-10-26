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
