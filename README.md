# WebGPU Gaussian Splat Viewer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture — Project 5**

Zachary Kelly

[Live WebGPU demo](https://zkelly1.github.io/Project5-WebGPU-Gaussian-Splat-Viewer/) · [Base demo](https://zkelly1.github.io/Project5-WebGPU-Gaussian-Splat-Viewer/demo.webm) · [Extra-credit demo](https://zkelly1.github.io/Project5-WebGPU-Gaussian-Splat-Viewer/extra-credit.webm)

This viewer loads a trained Gaussian PLY scene and its camera presets. The point
cloud renderer draws each Gaussian center as a yellow point. The Gaussian renderer
projects the shape of each Gaussian onto the screen, sorts the visible splats by
depth, and blends their colors from back to front. There are also options for
half-precision calculations and tile-based sorting. Tile sorting trades extra
binning and sorting work for a compute-based compositor.

## Build and run

Run from the repository root:

```powershell
npm install
npm run dev
```

Open the local address printed by Vite in a browser with WebGPU support. Load the
PLY file through `ply_file`, then load the matching camera JSON through `cam_file`.
Both files are needed before rendering starts. Scene files are provided in the
[course scene folder](https://drive.google.com/drive/folders/1Fz0QhyDU12JTsl2e7umGi5iy_V9drrIW?usp=sharing).

Use `npm run build` to check TypeScript and build the static site.

| Input | Action |
| --- | --- |
| Left mouse button and drag | Rotate the camera |
| Right mouse button and drag | Pan the camera |
| Scroll wheel | Move forward or backward |
| 0–9 | Select an available camera preset |
| renderer | Switch between point clouds and Gaussian splats |
| gaussian_multiplier | Scale the Gaussian axes; zero hides the splats |
| culling | Enable or disable the Gaussian frustum test |
| workgroup_size | Set preprocessing to 64, 128, or 256 threads |
| Half precision | Use f16 for rotation and spherical-harmonic lighting when supported |
| Tile sorting | Sort splats within screen tiles and composite in compute |
| Save benchmark | Download timings and settings for the current view |

## Scene captures

![Bicycle rendered with Gaussian splats](img/gaussian.png)

![The same view rendered as a point cloud](img/pointcloud.png)

The Gaussian image recovers the bicycle frame, bench, and grass with learned color.
The point cloud marks centers in yellow; it loses the continuous appearance of
surfaces even where the centers are dense.

The course model files are downloaded locally under `assets/scenes/` and excluded
from Git. They are also excluded from the production site build.

## Rendering

The PLY loader stores positions, opacity, rotation, scale, and spherical-harmonic
coefficients as packed half floats. The shaders unpack them before doing the math
in full precision.

Each preprocessing invocation handles one Gaussian. It transforms the center into
camera space and rejects centers outside a frustum with a 20% margin at the screen
edges. Points behind the camera are rejected even when culling is disabled.

Scale is stored in log space in the PLY file. Exponentiating it and applying the
scale slider gives the three Gaussian axes. The normalized quaternion rotates
those axes, and their covariance is transformed into camera space. A perspective
Jacobian projects that covariance into pixels. Adding 0.3 to its diagonal keeps
small splats from vanishing. The largest eigenvalue gives a three-sigma quad
radius; the inverse covariance gives the conic used by the fragment shader.

The shader evaluates spherical harmonics up to the scene's stored degree using
the direction from the camera to the Gaussian. Opacity is recovered with a sigmoid.
Visible indices and inverted positive depth keys go into the supplied radix
sorter. The visible count is copied into the indirect draw buffer, so only those
instances are drawn. Preprocessing workgroup size can change independently of the
sorter's fixed block size.

Each instance draws a four-vertex triangle strip. The fragment shader evaluates
the Gaussian at the pixel offset, discards very small alpha values, and outputs
premultiplied color. Back-to-front blending combines those contributions.

## Performance analysis

The captures use the course's `bicycle_30000.cleaned.ply` scene with 1,063,091 Gaussians and camera preset 0 at 1280 × 720. Measurements ran in Chromium 155 on Windows 11 Pro, with an AMD Ryzen 7 3700X and NVIDIA GeForce RTX 3070.

`Save benchmark` renders into a separate texture at the current canvas resolution.
It warms up for 30 frames, then measures 120 frames. Each sample starts before
command encoding and ends after `queue.onSubmittedWorkDone()`. This includes CPU
submission, GPU execution, and completion notification. It excludes presentation
and is not an isolated GPU timestamp measurement. The JSON includes all samples,
mean, median, p95, point count, camera pose, browser, resolution, and settings.
Camera input and the settings panel are disabled during the measurement.

### Measured results

Each value below is the median of three run medians. Each run has 30 warmup
frames and 120 measured frames. The range gives the lowest and highest run median.
The full samples are in [scene-benchmarks.json](performance/scene-benchmarks.json).

| Renderer | Points | Preprocess threads | Culling | Median ms | Run range, ms |
| --- | ---: | ---: | --- | ---: | ---: |
| Point cloud | 1,063,091 | — | — | 3.05 | 3.00–3.10 |
| Gaussian | 1,063,091 | 64 | On | 10.20 | 10.00–10.30 |
| Gaussian | 1,063,091 | 128 | On | 10.40 | 10.30–10.45 |
| Gaussian | 1,063,091 | 256 | On | 10.10 | 10.10–10.50 |
| Gaussian | 1,063,091 | 64 | Off | 10.10 | 10.10–10.10 |
| Gaussian | 1,063,091 | 128 | Off | 10.10 | 10.05–10.50 |
| Gaussian | 1,063,091 | 256 | Off | 10.15 | 10.10–10.40 |
| Gaussian | 500,000 | 256 | On | 5.90 | 5.75–6.00 |
| Gaussian | 100,000 | 256 | On | 3.50 | 3.45–3.50 |

The Gaussian renderer took about 3.3 times as long as the point cloud renderer,
but produced continuous, colored surfaces. Workgroup size had a small effect
relative to run-to-run variation. These measurements do not establish a clear
winner between 64, 128, and 256 threads. The default remains 256.

Culling was effectively neutral at this camera: 10.10 ms with it enabled and
10.15 ms disabled at 256 threads. This view contains most of the cleaned scene,
so little work is removed. That result does not imply culling is useless when
the camera faces away or much of a larger scene is outside the view.

The smaller point counts are evenly spaced samples of the original PLY records,
using record index floor(i × original_count / target_count). The camera and scale
stay fixed. Lower density also changes coverage and overlap, so this measures the
cost of those subsets, not a perfectly isolated change in invocation count.
Runtime rose from 3.50 ms at 100,000 points to 10.10 ms at 1,063,091 points. Fixed
submission and completion overhead helps explain why it did not grow in direct
proportion to point count.

The runs were sequential, with no presentation or video recording in the timed
loop. Browser scheduling and clock resolution are still part of the results.

## Validation

TypeScript and the production build pass. Chrome WebGPU checks used 4,097 synthetic
Gaussians to cross a sorting-block boundary. All three preprocessing group sizes,
with culling on and off, produced identical pixels. The benchmark returned 120
finite samples without WebGPU validation errors. Earlier checks covered reversed
depth order, zero scale, and recovery after an empty frame.

The full bicycle scene was visually checked in both renderers. The timed scene
runs reported no WebGPU validation errors. The [36-second demo](public/demo.webm)
shows point clouds, Gaussian splats, camera presets, and scale changes.

The deployed version was checked on October 25, 2026 with a 100,000-point bicycle
sample at 1280 × 720. File loading, both renderers, camera presets, mouse movement,
scale changes, culling, and the three workgroup sizes worked. Half precision and
tile sorting worked separately and together. The benchmark export contained 120
samples and the selected settings. No browser or WebGPU errors were reported.
Both hosted videos match the checked local recordings.

GitHub Pages uses the **Deploy static content to Pages** workflow. It is configured
for pushes to `main` and can also be run manually from Actions. The latest upload
needed a manual run before the new controls appeared on the live site. The build
uses `/Project5-WebGPU-Gaussian-Splat-Viewer/` as its base path. Scene files are
selected locally and are not included in the deployed site.

The [extra-credit demo](public/extra-credit.webm) shows full and half precision
with global and tile-based sorting on the full bicycle scene. Each combination
is labeled in the video.

## Extra credit: half-precision preprocessing

Enable **Half precision** in Config to evaluate quaternion rotation and spherical
harmonic lighting with WGSL `f16` arithmetic. The projected conic and RGBA color
are also stored as f16. This reduces each output splat from 64 to 48 bytes,
a 25% reduction in the intermediate splat buffer. The input was already packed.
Camera transforms, covariance projection, conic inversion, and depth keys stay
in f32 to keep their range and sorting accuracy. The raster and compute renderers
both read the compressed output.

The viewer requests `shader-f16` only when the adapter supports it. Otherwise the
checkbox is disabled. Switching precision rebuilds the output buffer and its
bindings; switching back restores the full-precision layout. Full precision is
the default. This is mixed-precision preprocessing, not an all-f16 projection.

On the 100,000-point bicycle sample at 640 × 360, compressed half precision
changed RGB values by about 0.011 levels on average and at most 3 out of 255.
The checks include all three workgroup sizes and switching back to f32.
[Current GPU results](performance/gpu-validation-sample.json) record the individual
comparisons. The older half-precision results used the uncompressed output.

## Extra credit: tile-based sorting

Enable **Tile sorting** while using the Gaussian renderer. A count pass finds how
many splats overlap each 16 × 16 tile. A prefix pass assigns packed ranges, then
a second binning pass writes depth/index pairs into those ranges. Parallel merge
passes sort each tile independently, using the splat index to break depth ties.
There is no longer a 1,024-splat limit per tile.

The compute compositor walks each tile's list front to back and stops when
transmittance drops below 0.0001. It writes the final image to an RGBA texture;
a fullscreen triangle only presents that texture. Dense tiles use the same compute
compositor. There is no raster blending fallback. Projection and alpha equations
follow the assignment's [Gaussian Splatting Notes](https://github.com/kwea123/gaussian_splatting_notes).

The two packed-list buffers each reserve up to 32 entries per input splat, capped
by the device's buffer limits. If the total overlap count exceeds this pool, the
remaining tiles filter the complete globally sorted list in compute. This avoids
dropping splats, but that exceptional path is expensive. Normal frames disable
the global radix dispatch. Resizing rebuilds the tile ranges and output texture.
Both extra-credit options can be enabled together and appear in benchmark exports.
Global sorting remains the default.

Compute accumulation differs from blending into an 8-bit framebuffer after every
splat. Early termination and equal-depth ordering can also change pixels. The
old tile timings and demo used the earlier raster fallback and do not measure the
packed-list implementation.

The revised path passed 21 synthetic image checks and 30 bicycle-scene comparisons
through Dawn/D3D12, with no validation errors. This includes the full 1,063,091-point
scene at 1280 × 720, both precision modes, and a device without f16 enabled.
The full-scene tile image differed from the raster baseline by about 0.313 RGB
levels on average, with a maximum of 33 out of 255. The rendered image was also
visually checked. Results are saved for the [synthetic cases](performance/gpu-validation-synthetic.json),
[full scene](performance/gpu-validation-full.json), and [device without f16](performance/gpu-validation-no-f16.json).

The tile path was slower in these validation runs. Their elapsed times include
readback, CPU pixel comparison, and lazy pipeline setup, so they are not steady-state
benchmarks. No performance improvement is claimed. The current changes were checked
directly through Dawn after an earlier browser GPU reset. Restarting the desktop
app restored its adapter, and all eight browser tile checks then passed without
validation errors. The live-site checks above describe the earlier deployed version.

### Running the GPU checks

The browser-only synthetic checks are at `/tests/tile-validation.html` while the
development server is running. They need no external scene files.

The more detailed checks run the same shaders through Dawn's Node WebGPU runtime:

```powershell
npm install --prefix node_modules/gpu-check webgpu@0.6.2
node tests/gpu-validation.mjs synthetic
node tests/gpu-validation.mjs sample
node tests/gpu-validation.mjs full
node tests/gpu-validation.mjs no-f16
```

The scene checks expect `bicycle-smoke.ply`, `bicycle.ply`, and
`bicycle-cameras.json` under `assets/scenes/`. The sample contains 100,000 evenly
spaced records from the full scene. The runner writes results under `performance/`.
It checks pixels against a CPU reference, tied depths, low-opacity contributions,
packed-pool overflow, resizing, precision switching, and recovery after zero scale.
These are GPU execution checks; they do not exercise browser file dialogs or UI.

## Credits

- Course starter: PLY loading, camera controls, UI, and GPU radix sort.
- [Vite](https://vitejs.dev/)
- [Tweakpane](https://tweakpane.github.io/docs/)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
- [Float16Array](https://github.com/petamoriken/float16)
- Shrek Shao and the [Differential Gaussian Renderer](https://github.com/graphdeco-inria/diff-gaussian-rasterization).
