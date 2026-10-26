# Project5-WebGPU-Gaussian-Splat-Viewer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 5**

* Luke (Hyuk Che) Kwon
  * [LinkedIn](https://www.linkedin.com/in/hyukchekwon/), [Personal Website](https://lukekwon98.github.io/), [Github](https://github.com/lukekwon98)
* Tested on: Windows 11, AMD Ryzen 5 5600X 6-Core Processor @ ~3.7GHz 16GB, Nvidia GeForce RTX 3060 (Compute Capability 8.6)

A WebGPU viewer for pretrained 3D Gaussian scenes, with point-cloud and Gaussian-splat rendering modes. The Gaussian renderer projects 3D Gaussians into screen-space ellipses, evaluates view-dependent color using spherical harmonics, sorts splats by depth, and composites them with alpha blending.

### Screenshot

<img width="2556" height="1268" alt="Screenshot 2026-10-25 214850" src="https://github.com/user-attachments/assets/6dc4af81-2b55-4c0f-a855-03bb9f1e6535" />

### Video / Live Demo

<a href="https://www.youtube.com/watch?v=zuF_spSZdO0">
  <img src="https://img.youtube.com/vi/zuF_spSZdO0/maxresdefault.jpg"
       alt="Watch the demo" width="650">
</a>

[Live Demo Link](https://luke-kwon.com/Project5-WebGPU-Gaussian-Splat-Viewer/)

## Features

- Point-cloud rendering for comparison with Gaussian splatting.
- Compute-based preprocessing, including view-frustum culling and projection of 3D covariance into screen space.
- Spherical harmonics color evaluation.
- Depth-sorted Gaussian quads with exponential opacity falloff and alpha blending.
- Indirect drawing based on the number of retained Gaussians.
- Adjustable Gaussian scale and camera presets.

## Rendering Pipeline

1. Load Gaussian attributes and spherical harmonics coefficients from a PLY file.
2. Transform Gaussian centers and reject those outside the configured frustum bounds.
3. Compute projected covariance, inverse covariance, screen-space bounds, and color.
4. Populate depth keys and indices for the provided radix sorter.
5. Sort retained Gaussians for back-to-front rendering.
6. Update the indirect draw count and render Gaussian quads with alpha blending.

## Performance Analysis

### Scenes

| Bicycle | Bicycle Cleaned | Bonsai |
| :---: | :---: | :---: |
| <img width="400" alt="Bicycle" src="https://github.com/user-attachments/assets/6dc4af81-2b55-4c0f-a855-03bb9f1e6535" /> | <img width="400" alt="Bicycle Cleaned" src="https://github.com/user-attachments/assets/99327adb-49de-4629-ae48-2c108d29cbf5" /> | <img width="400" alt="Bonsai" src="https://github.com/user-attachments/assets/94a1df7e-57f9-4692-9e83-24e7ae4bd724" /> |

### Test Baseline

| Setting | Value |
| --- | --- |
| Browser | Chrome |
| Canvas resolution | 1280 x 655 |
| Gaussian multiplier | 1.0 |
| Preprocessing workgroup size | 256 |
| Timing method | WebGPU GPU timestamp queries |
| Benchmark duration | 5 second warmup, followed by 10 seconds of sampling |

All timings are mean GPU times in milliseconds, so lower is better. Point-cloud timing covers its render pass, and total Gaussian timing spans the beginning of preprocessing through the end of splat rendering, including sorting and intervening GPU commands.

### Point Cloud vs. Gaussian Rendering

<img width="600" height="371" alt="pointcloudGaussian_numpoints" src="https://github.com/user-attachments/assets/c5f97399-2547-49b8-bb4c-837deee373a6" />

| Scene | Gaussians | Point cloud GPU time (ms) | Gaussian GPU time (ms) |
| --- | ---: | ---: | ---: |
| Bonsai | 272,956 | 1.5848 | 3.7092 |
| Bicycle_Cleaned | 1,063,091 | 2.0947 | 11.5255 |
| Bicycle | 6,131,954 | 2.6165 | 19.2526 |

The point-cloud renderer displays individual Gaussian centers as yellow points. It reveals the distribution of the samples, but does not represent size, opacity, or view-dependent color, while Gaussian rendering uses overlapping, colored elliptical splats to reconstruct more continuous surfaces.

Gaussian rendering takes approximately 2.34×, 5.50x, and 7.36× as long as point-cloud rendering for Bonsai, Bicycle_Cleaned, and Bicycle. Its additional work includes covariance projection, color evaluation, depth sorting, and rasterization of overlapping blended quads. The visual reconstruction therefore comes with a higher GPU cost.

### Effect of Gaussian Count

Across the three scenes, total Gaussian GPU time rises from 3.7092 ms at 272,956 Gaussians to 19.2526 ms at 6,131,954 Gaussians. More input Gaussians require more preprocessing, while more retained Gaussians increase sorting and drawing work. Larger numbers of overlapping splats can also increase fragment shading and blending work.

We can also find that the relationship is not proportional, as Bicycle contains approximately 22.5× as many Gaussians as Bonsai, but takes approximately 5× as long to render. Visibility, projected splat size, and overlap also influence performance.

### Effect of XY Frustum Culling

<img width="600" height="371" alt="XY frustum culling" src="https://github.com/user-attachments/assets/7a2cf329-9a3b-4274-8994-232221c5762d" />

This experiment toggles XY frustum rejection, using the existing 1.2× clip-space margin when enabled. Depth rejection checks remain enabled in both configurations, so it measures XY culling rather than disabling all frustum checks.

| Scene | XY culling on (ms) | XY culling off (ms) |
| --- | ---: | ---: |
| Bonsai | 3.7092 | 3.6008 |
| Bicycle_Cleaned | 11.5255 | 11.6789 |
| Bicycle | 19.2526 | 27.5085 |

Culling has the largest benefit for Bicycle, reducing GPU time by approximately 30% while Bicycle_Cleaned and Bonsai show a much smaller reduction.

Early rejection avoids unnecessary covariance and color calculations, sorting entries, and vertex processing for off-screen Gaussians. The benefit depends on how much work can be rejected from the selected view, and these test results are consistent since Bicycle has gaussians populating the entire space around the camera, while Bonsai and Bicycle_Cleaned are small scenes where the majority of Gaussians exist within the viewport in the first place.

### Effect of Preprocessing Workgroup Size

<img width="600" height="371" alt="workgroupsize" src="https://github.com/user-attachments/assets/40775dba-c4f1-4c24-ac54-608443f8fb07" />

| Preprocessing threads per workgroup | Total Gaussian GPU time (ms) |
| ---: | ---: |
| 32 | 11.5496 |
| 64 | 11.6974 |
| 128 | 11.5686 |
| 256 | 11.5656 |

The results show little correlation to workgroup size, as the largest difference is only around 0.1 ms. Although 32 threads produces the lowest recorded time, this is mostly due to variance. 
Workgroup size can affect GPU scheduling and resource occupancy, but in our case each preprocessing invocation handles one Gaussian, and all tested configurations perform the same overall computation. Similar hardware utilization across these sizes is one possible explanation. In addition, total timing includes sorting and rendering, which can dilute changes applied to preprocessing.

### Credits

- [Vite](https://vitejs.dev/)
- [tweakpane](https://tweakpane.github.io/docs//v3/monitor-bindings/)
- [stats.js](https://github.com/mrdoob/stats.js)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
- Special Thanks to: Shrek Shao (Google WebGPU team) & [Differential Guassian Renderer](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
