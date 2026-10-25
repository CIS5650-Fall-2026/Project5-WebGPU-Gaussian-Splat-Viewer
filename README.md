# Project5-WebGPU-Gaussian-Splat-Viewer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 5**

* Ethan Yang
* Tested on: Google Chrome 155, Windows 11, NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM)


### Live Demo

[WebGPU Gaussian Splat Viewer](https://eyang72004.github.io/Project5-WebGPU-Gaussian-Splat-Viewer/)




### Demo Video/GIF

[Bonsai Demo](images/07_bonsai_HQ.mp4)

[Bicycle Demo](images/08_bicycle_HQ.mp4)



### Project Overview

This project implements a WebGPU-based 3D Gaussian Splatting viewer that supports both point-cloud rendering and Gaussian splat rendering. The point-cloud renderer provides a baseline for comparing rendering quality and GPU performance, while the Gaussian renderer reconstructs scenes using projected Gaussian splats.

The viewer supports loading PLY scene files, importing camera presets from JSON files, interactive camera navigation, and switching between rendering modes. The Gaussian rendering pipeline uses GPU compute shaders to preprocess Gaussian attributes and perform depth sorting before rendering the resulting splats.

A major focus of this project is performance analysis. I implemented GPU timestamp benchmarking to compare point-cloud and Gaussian rendering, evaluate different compute shader workgroup sizes, measure the effect of frustum culling, and investigate how performance scales with the number of Gaussians.


### Implemented Features

- **Point-Cloud Renderer:** Renders the loaded PLY scene as points, providing a baseline for comparison with Gaussian splatting.
- **Gaussian Splat Renderer:** Renders projected 3D Gaussians using their positions, scales, rotations, opacities, and color information.
- **GPU Preprocessing:** Uses WebGPU compute shaders to prepare Gaussian attributes for screen-space rendering.
- **GPU Depth Sorting:** Sorts Gaussian splats according to depth for ordered rendering.
- **Frustum Culling:** Performs visibility checks during preprocessing, including configurable viewport-overlap rejection.
- **Configurable Compute Workgroup Size:** Supports workgroup sizes of 64, 128, and 256 threads for preprocessing experiments.
- **Gaussian Count Selection:** Supports loading 25%, 50%, 75%, or 100% of a PLY scene to evaluate performance scaling.
- **Camera Controls:** Supports interactive camera navigation and camera presets loaded from JSON files.
- **GPU Performance Benchmarking:** Uses WebGPU timestamp queries to measure GPU execution time and export individual measurements as CSV files.
- **Half-Precision Gaussian Preprocessing (+10 Extra Credit):** Implements an optional `shader-f16` preprocessing path with native half-precision spherical-harmonics color, opacity, and scale evaluation and a compressed 32-byte intermediate splat representation, compared with the 48-byte baseline.
- **Precision Benchmarking:** Automatically compares f32 and f16 rendering using GPU timestamp queries and exports per-sample CSV results.



### Implementation Details




#### Point-Cloud Rendering

The point-cloud renderer provides a straightforward baseline for displaying the input scene. Each point represents a position from the loaded PLY data, with its associated color information.

Compared with Gaussian splatting, point-cloud rendering requires substantially less processing per point. However, points do not represent the projected spatial extent of a Gaussian, which can result in a less continuous scene reconstruction.

The point-cloud renderer is also used as a performance baseline in the benchmarking experiments.



#### Gaussian Splat Rendering

The Gaussian renderer processes the input scene through three main stages:

1. **Gaussian Preprocessing:** Gaussian attributes are transformed and projected into screen space. Visibility and validity checks are applied, and the resulting attributes are prepared for rendering.
2. **Depth Sorting:** Gaussians are sorted according to depth to support ordered splat rendering.
3. **Splat Rendering:** The projected Gaussians are rendered using their screen-space shapes, opacities, and color information to reconstruct the scene.

Gaussian splatting represents a scene using spatially extended primitives rather than isolated points. This enables smoother and more continuous reconstruction, but requires additional GPU work for preprocessing, sorting, and rendering.



#### Frustum Culling

The Gaussian preprocessing pipeline includes checks that reject Gaussians which should not contribute to the rendered view.

For performance experiments, I implemented a toggle for the viewport-overlap culling check. This allows the effect of that particular culling operation to be measured.

Importantly, disabling this toggle does not disable all preprocessing visibility or validity checks. Behind-camera, depth-range, and invalid-projection checks remain active.

#### Configurable Gaussian Count

To evaluate how GPU execution time changes with scene size, I extended the PLY loading process to support four Gaussian-count configurations:

- 25% of the original Gaussians
- 50% of the original Gaussians
- 75% of the original Gaussians
- 100% of the original Gaussians

The loader computes the requested number of Gaussians from the original PLY vertex count and allocates GPU buffers accordingly.

The fractional configurations use the first N Gaussians in the PLY file rather than random sampling. Therefore, smaller configurations may have different spatial distributions from the complete scene.


#### Half-Precision Gaussian Preprocessing (Extra Credit)

I implemented a half-precision Gaussian preprocessing path using WebGPU's `shader-f16` feature. The original f32 pipeline is preserved as a baseline, and the viewer provides a precision selector to switch between the two implementations.

The f16 preprocessing shader (`preprocess_f16.wgsl`) uses native half-precision arithmetic for three per-Gaussian calculations:

- **Spherical-harmonics color evaluation.** SH coefficients are already stored as f16 by the loader, so each packed `u32` is reinterpreted with `bitcast<vec2<f16>>` instead of being widened to f32. The view direction is normalized in f32 (squared world-space distances can exceed the f16 range) and then converted to f16, after which the full degree-0 to degree-3 SH evaluation, the `+0.5` offset, and the clamp to non-negative values are performed in f16 using f16-rounded basis constants. The SH basis, coefficient ordering, and clamping match the f32 shader.
- **Opacity sigmoid.** `1 / (1 + exp(-x))` is evaluated in f16. Because `exp(-x)` overflows f16 when `x < -ln(65504) ≈ -11.09`, the logit is clamped from below at -10.5, where the sigmoid is about 2.8e-5, far below one 8-bit alpha step.
- **Scale exponentiation.** `exp(log_scale)` is evaluated in f16, with the log-scale clamped from above at 10.5 (a scale of about 3.6e4 world units). Without this bound, an overflowed scale would make the covariance NaN (`inf * 0`). Very small scales may underflow to f16 subnormals or zero; no lower bound is applied because such Gaussians are already dominated by the projected-covariance regularization.

Covariance construction, the view and perspective projection, conic inversion, frustum and validity decisions, and depth-sort keys remain in f32 to avoid introducing geometric error. Final opacity and color are widened exactly to f32 only so they can be packed with `pack2x16float`.

f16 SH evaluation carries roughly three significant decimal digits, so colors can differ from the f32 baseline by a fraction of an 8-bit step, and occasionally by one step after accumulated rounding. Opacity and color were already quantized to f16 by the compressed splat format before this change.

The optimized pipeline also compresses the intermediate splat representation from 48 bytes to 32 bytes per Gaussian. Projected centers and inverse covariance coefficients remain in f32, while opacity and RGB color are packed into four 16-bit floating-point values using `pack2x16float`. A separate rendering shader decodes these values using `unpack2x16float`.

The compressed representation reduces the intermediate splat stride by 33.3%. Both the original and compressed buffers are currently allocated to support runtime switching, so this does not imply a 33.3% reduction in total GPU memory usage.

The f16 pipeline is available only when the GPU supports `shader-f16`; otherwise, the original f32 implementation remains available.

The viewer starts in the f32 baseline mode. The f16 path is opt-in through the precision selector, which is only offered when the device exposes `shader-f16`.



#### GPU Performance Benchmarking

I implemented GPU timestamp-based benchmarking to evaluate rendering performance.

Each benchmark configuration uses 30 warm-up frames followed by 100 measured GPU samples. The benchmark reports mean GPU execution time and exports individual samples to CSV files.

For the Gaussian renderer, each sample is the interval between a timestamp written at the beginning of the preprocessing compute pass and a timestamp written at the end of the splat render pass. Every Gaussian timing therefore includes **preprocessing, depth sorting, and rendering** together, not any one stage in isolation.

Four benchmark modes were implemented:

- **Workgroup-size comparison:** Measures Gaussian rendering performance with 64, 128, and 256 preprocessing threads per workgroup.
- **Frustum-culling comparison:** Measures Gaussian rendering performance with viewport-overlap culling enabled and disabled.
- **Renderer comparison:** Measures point-cloud and Gaussian rendering under the same loaded scene configuration.
- **Precision comparison:** Measures the original f32 Gaussian pipeline against the compressed f16 implementation, using identical workgroup-size and frustum-culling settings.

The renderer comparison was also used to evaluate performance at different Gaussian counts.

All reported GPU timings were collected using Google Chrome 155 on Windows 11 with an NVIDIA GeForce RTX 5060 Laptop GPU.



### Performance Analysis

The following experiments investigate the performance characteristics of the implemented WebGPU Gaussian splatting pipeline.

For each configuration, I collected 100 GPU timestamp samples after 30 warm-up frames. The tables report the arithmetic mean GPU execution time.

GPU execution time is distinct from the frame rate displayed by the viewer. The results measure GPU work captured by the benchmark rather than complete end-to-end application latency.

#### Benchmark Consistency

Experiments 1–4 used the original f32 pipeline, while Experiment 5 compares f32 against the final f16 implementation.

Some configurations appear in multiple experiments with different GPU times. For example, Bonsai at 256 threads with culling enabled measured 1.922 ms in the workgroup experiment, 2.252 ms in the culling experiment, and 1.834 ms in the renderer comparison. These measurements came from separate sessions, with implementation changes between some runs and potentially different camera views.

For this reason, I compare configurations within each experiment rather than directly comparing GPU times across separate sessions. The Bicycle measurement in Experiment 1 is also reused as the 100% configuration in Experiment 4.



#### Visual Comparison

The following screenshots compare point-cloud and Gaussian splat rendering for the Bonsai and Bicycle scenes.

**Bonsai — Point-Cloud Rendering**

![Bonsai point-cloud rendering](images/01_bonsai_pointcloud.png)

**Bonsai — Gaussian Splat Rendering**

![Bonsai Gaussian splat rendering](images/02_bonsai_gaussian.png)

**Bicycle — Point-Cloud Rendering**

![Bicycle point-cloud rendering](images/03_bicycle_pointcloud.png)

**Bicycle — Gaussian Splat Rendering**

![Bicycle Gaussian splat rendering](images/04_bicycle_gaussian.png)




#### 1. Point-Cloud Rendering vs. Gaussian Splat Rendering

The first experiment compares the performance of the point-cloud renderer and the Gaussian splat renderer.

| Scene | Gaussian Count | Point-Cloud GPU Time | Gaussian GPU Time | Gaussian / Point-Cloud |
|---|---:|---:|---:|---:|
| Bonsai | 272,956 | 0.193 ms | 1.834 ms | 9.52x |
| Bicycle | 1,063,091 | 0.732 ms | 7.878 ms | 10.76x |

For both scenes, point-cloud rendering was substantially faster than Gaussian splatting.

On Bonsai, Gaussian rendering required approximately 9.52 times as much GPU time as point-cloud rendering. On Bicycle, Gaussian rendering required approximately 10.76 times as much GPU time.

This difference reflects the additional operations performed by the Gaussian renderer. Whereas point-cloud rendering draws relatively simple point primitives, Gaussian splatting requires preprocessing Gaussian attributes, sorting splats by depth, and rendering their projected screen-space coverage.

These additional operations allow Gaussian splatting to represent the projected size and shape of each primitive, producing a more continuous reconstruction than isolated points.

The results demonstrate a tradeoff between the richer representation provided by Gaussian splatting and its higher computational cost.





#### 2. Effect of Compute Shader Workgroup Size

The second experiment investigates whether increasing the Gaussian preprocessing workgroup size improves performance.

Three workgroup sizes were evaluated: 64, 128, and 256 threads.

| Workgroup Size | Bonsai GPU Time | Bicycle GPU Time |
|---|---:|---:|
| 64 | 1.956 ms | 7.662 ms |
| 128 | 1.934 ms | 7.823 ms |
| 256 | 1.922 ms | 7.841 ms |

![Workgroup size performance](images/09_workgroup_performance.png)

For Bonsai, increasing the workgroup size from 64 to 256 threads slightly reduced GPU execution time, from 1.956 ms to 1.922 ms (about 1.7%).

For Bicycle, the 64-thread configuration was the fastest: 7.662 ms, compared with 7.841 ms at 256 threads (about 2.3% slower at 256).

Neither scene shows a consistent trend, and the measured differences are relatively small. The preprocessing workgroup size affects only the preprocessing dispatch, while depth sorting and rendering remain unchanged. Since the timing includes all three stages, improvements in preprocessing may have a limited effect on total GPU execution time.

The preprocessing kernel also processes Gaussians independently, without shared memory or workgroup barriers. Changing the workgroup size primarily affects GPU scheduling rather than the amount of computation.

Each configuration was measured once using 100 GPU samples. For comparison, the repeated f32 Bicycle measurements in Experiment 5 varied by approximately 2.3%, similar to the differences observed here. These results therefore do not establish a single optimal workgroup size for this GPU.




#### 3. Effect of Frustum Culling

The third experiment evaluates the performance impact of the implemented viewport-overlap culling operation.

Both configurations used a preprocessing workgroup size of 256 threads.

| Scene | Culling ON | Culling OFF | Performance Difference |
|---|---:|---:|---|
| Bonsai | 2.252 ms | 2.241 ms | OFF approximately 0.5% faster |
| Bicycle | 7.828 ms | 7.983 ms | ON approximately 1.9% faster |

![Frustum culling performance](images/10_culling_performance.png)

For Bonsai, enabling viewport-overlap culling did not improve performance. The culling-disabled configuration was approximately 0.5% faster.

For Bicycle, enabling culling reduced GPU execution time by approximately 1.9%.

The culling toggle controls only the viewport-overlap test, which checks whether a Gaussian's projected three-sigma footprint intersects the screen. Other visibility and validity checks, including behind-camera, depth-range, and invalid-covariance rejection, remain enabled in both configurations.

With culling disabled, Gaussians that are in front of the camera but entirely outside the viewport may still undergo spherical-harmonics color evaluation, depth sorting, and instanced quad rendering. Enabling viewport-overlap culling rejects these Gaussians earlier, reducing preprocessing, sorting, and vertex-processing work.

However, off-screen quads produce essentially no fragments because they are clipped or discarded during rasterization. Therefore, viewport-overlap culling does not significantly reduce fragment shading for fully off-screen Gaussians. Its performance benefit must also outweigh the additional overlap test performed during preprocessing.

The results suggest that the effectiveness of viewport-overlap culling depends on the scene and camera view. For Bicycle, enabling culling reduced GPU execution time by approximately 1.9%, while Bonsai showed a negligible difference of 0.011 ms. Since I did not record visible-Gaussian counts, I cannot determine whether the Bicycle view contained more off-screen Gaussians. These measurements indicate a small, scene-dependent benefit rather than a consistent performance improvement.



#### 4. Effect of Gaussian Count

The fourth experiment investigates how GPU execution time changes as the number of Gaussians increases.

I used the Bicycle scene and loaded 25%, 50%, 75%, and 100% of its original Gaussian data.

Each configuration used a 256-thread preprocessing workgroup size with viewport-overlap culling enabled.

| Gaussian Fraction | Gaussian Count | Point-Cloud GPU Time | Gaussian GPU Time |
|---|---:|---:|---:|
| 25% | 265,772 | 0.196 ms | 1.982 ms |
| 50% | 531,545 | 0.396 ms | 4.169 ms |
| 75% | 797,318 | 0.563 ms | 5.390 ms |
| 100% | 1,063,091 | 0.732 ms | 7.878 ms |

![Gaussian count scaling](images/11_gaussian_count_scaling.png)

Increasing the Gaussian count from 265,772 to 1,063,091, approximately a fourfold increase, increased Gaussian GPU execution time from 1.982 ms to 7.878 ms.

This corresponds to approximately a 3.98x increase in execution time.

Point-cloud rendering time also increased with scene size, from 0.196 ms at 25% to 0.732 ms at 100%.

The measurements indicate approximately linear scaling with Gaussian count over the tested range.

As the number of Gaussians increases, more primitives must be preprocessed, sorted, and potentially rendered. These additional operations increase the total GPU workload.

However, the measurements do not establish a formal asymptotic complexity bound for the rendering pipeline.

Additionally, the reduced-size datasets were created by selecting the first N Gaussians in the PLY file rather than randomly sampling the full scene. Consequently, changes in spatial distribution and visible coverage may also influence the measured GPU times.



#### 5. Half-Precision Gaussian Preprocessing (Extra Credit)

This experiment compares the original f32 Gaussian pipeline against the final f16 implementation, which uses half-precision arithmetic for spherical-harmonics color evaluation, opacity, and scale calculations, along with compressed intermediate splat storage.

I tested the Bicycle scene with all 1,063,091 Gaussians, 256 preprocessing threads per workgroup, and viewport-overlap culling enabled. Each run measured f32 followed by f16, with 30 warm-up frames and 100 GPU timestamp samples per mode. I repeated the comparison three times for 300 samples per precision mode.

| Run | f32 Mean GPU Time | f16 Mean GPU Time | GPU Time Reduction |
|---|---:|---:|---:|
| 1 | 7.068 ms | 7.054 ms | 0.20% |
| 2 | 7.230 ms | 7.031 ms | 2.75% |
| 3 | 7.171 ms | 7.069 ms | 1.43% |
| **Overall** | **7.156 ms** | **7.051 ms** | **1.47%** |


![f32 vs. f16 Gaussian rendering performance](images/12_f16_vs_f32_performance.png)

Across the three runs, f16 reduced average GPU execution time from 7.156 ms to 7.051 ms, an improvement of approximately 1.47%. It was faster in all three runs, although the improvement varied between runs.

The compressed intermediate splat representation uses 32 bytes per Gaussian instead of 48 bytes, reducing its stride by 33.3%. Both buffer formats are still allocated for runtime switching, so this is not a 33.3% reduction in total GPU memory usage.

Since the benchmark measures preprocessing, depth sorting, and rendering together, the results reflect the complete Gaussian pipeline rather than the preprocessing shader alone. The modest improvement also suggests that reducing intermediate storage and using f16 arithmetic does not necessarily produce an equally large reduction in total rendering time.

These measurements were collected using the final f16 shader, including native f16 spherical-harmonics evaluation and the opacity and scale range safeguards.





#### Performance Analysis Summary

The five experiments demonstrate several characteristics of the WebGPU Gaussian splatting pipeline:

- Point-cloud rendering is substantially faster than Gaussian splatting because it avoids much of the additional preprocessing, sorting, and splat rendering work.
- Increasing compute shader workgroup size does not consistently improve performance across scenes.
- The benefit of viewport-overlap culling depends on the scene and the amount of work eliminated.
- GPU execution time increases approximately proportionally with Gaussian count over the tested range.
- The compressed f16 Gaussian pipeline achieved a modest reduction in measured GPU execution time across three benchmark runs, while reducing intermediate splat storage from 48 to 32 bytes per Gaussian.

Overall, these experiments show that Gaussian rendering performance depends on both the number of primitives and the characteristics of the GPU workload. They also demonstrate why GPU optimizations should be evaluated empirically rather than assumed to improve performance in every scene.






### Credits

- [Vite](https://vitejs.dev/)
- [tweakpane](https://tweakpane.github.io/docs//v3/monitor-bindings/)
- [stats.js](https://github.com/mrdoob/stats.js)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
- Special Thanks to: Shrek Shao (Google WebGPU team) & [Differential Guassian Renderer](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
