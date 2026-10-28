## 2. Effect of workgroup size on Gaussian rendering performance

I compared preprocessing workgroup sizes of 64, 128, and 256 by running the original project at `http://localhost:5173/` in local Google Chrome. All runs used `bicycle_30000.cleaned.ply`, which actually contains **1,063,091 Gaussians**, camera preset 0, Gaussian scaling 1.00, and a **1283 × 741** canvas. Frustum culling and the rendering and sorting algorithms remained unchanged.

For each run, I allowed approximately five seconds of warm-up and then sampled the Viewer’s displayed instantaneous FPS 21 times, approximately 0.5 seconds apart. The table reports the median and observed range of these displayed readings, rather than an average calculated over every rendered frame.

| Preprocess workgroup size | Median displayed FPS | Observed displayed FPS range |
| --- | ---: | ---: |
| 64 | 153.85 | 128.21–185.19 |
| 128 | 149.25 | 128.21–163.93 |
| 256 | 147.06 | 125.00–172.41 |

The 64-thread configuration had the highest median in this comparison, approximately **4.6% above 256**. However, the observed ranges overlap substantially. These short runs therefore suggest only a small difference in end-to-end rendering performance; they do not establish a statistically reliable winner or an optimal workgroup size for other scenes or devices.

Workgroup size changes how the preprocessing invocations are grouped and scheduled. Smaller groups can allow more flexible scheduling, while larger groups can change resource allocation and the number of groups that can execute concurrently. The balance depends on shader resource requirements and the GPU. NVIDIA’s guidance notes that a larger block size does not necessarily imply higher occupancy. This is a possible explanation for the measured behavior, rather than an occupancy bottleneck confirmed by this test. See the [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).

Only the preprocessing workgroup size was varied. To keep sorting comparable, `sortKeyPerThread` was adjusted so that `workgroupSize × sortKeyPerThread` remained **3840**, matching the original radix-sort dispatch capacity. Sorting and rasterization still contribute to total frame time, so a change in preprocessing alone may have a limited effect on the displayed FPS.

These measurements use the application’s existing FPS display and do not isolate GPU preprocessing time. Repeated runs and GPU timing would be needed to distinguish a small scheduling benefit from normal timing variation.

The original renderer file was restored byte-for-byte after testing. The final 256-thread run used that restored file. TypeScript checking passed, and the browser reported no captured rendering errors in the completed runs.

### Measurement records

- [64-thread raw readings](original-viewer-64.json) · [Screenshot](original-viewer-64.png)
- [128-thread raw readings](original-viewer-128.json) · [Screenshot](original-viewer-128.png)
- [256-thread raw readings](original-viewer-256.json) · [Screenshot](original-viewer-256.png)

An earlier 256-thread baseline also had a median displayed FPS of 147.06, with a range of 131.58–166.67. The table above uses the final 256-thread run, collected with the same five-second warm-up procedure as the 64- and 128-thread runs.
