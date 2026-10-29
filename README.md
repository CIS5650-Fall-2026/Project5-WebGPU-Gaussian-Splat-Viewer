# Project5-WebGPU-Gaussian-Splat-Viewer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 5**

* Baron Ping-Yeh Hsieh
  * [LinkedIn](https://www.linkedin.com/in/baron-ping-yeh-hsieh-976376290/)
* Tested on: **Google Chrome 156.0** on
  Windows 11, AMD Ryzen 9 7940HS w/ Radeon 780M Graphics, 16GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU
![](/images/bicycle.png)

### Live Demo
![](/images/bonsai_point_cloud.png)

[Link to Live Demo Website](https://baronhsieh2005.github.io/Project5-WebGPU-Gaussian-Splat-Viewer/)

### Demo Video/GIF
[Video Demo With Bicycle](https://drive.google.com/file/d/1Ww_MiR5BUdd8oVFoMi8R4yVLF-RXD5IO/view?usp=sharing) \
[Video Demo With Bonsai Tree](https://drive.google.com/file/d/1V3ADP7C6papadf5cpzMV-qjaZYOdQX86/view?usp=sharing)

## Performance Analysis

For the performance analysis, I tested the Gaussian Splat Viewer using two different scenes (`bicycle_30000.cleaned.ply` and `bonsai.ply`), with the bicycle scene containing 1,063,091 Gaussians and the bonsai scene containing 272,956 Gaussians. The FPS values were observed through the application's built-in FPS monitor while switching between the point-cloud and Gaussian renderers, and moving the camera around to inspect the scene from different viewpoints.

### 1. Compare Your Results From Point-Cloud and Gaussian Renderer, What Are the Differences?

| Scene | Gaussian Count | Point Cloud (FPS) | Gaussian (FPS) |
|---|---:|---:|---:|
| Bicycle | 1,063,091 | ~163.9 | ~48.8–73.0 |
| Bonsai | 272,956 | ~163.9–166.7 | ~161.3–166.7 |

- Visually, the difference between the two renderers is pretty significant. The point-cloud renderer only draws individual yellow points to represent the scene, and while we can still distinguish the general shape of the bicycle, bench, or bonsai tree, the objects look relatively sparse and noisy (especially when zooming in, since the individual points become much more apparent). The Gaussian renderer, on the other hand, produces a much more complete and realistic reconstruction of the scene, with proper colors and smoother surfaces, since each point is represented by an elliptical Gaussian splat rather than just a single pixel.

- Performance-wise, the point-cloud renderer generally performs better, particularly for the much larger bicycle scene where it stayed around 164 FPS, while the Gaussian renderer fluctuated between roughly 49 and 73 FPS during the demonstration. This is pretty much what I expected since the Gaussian renderer requires several additional steps per frame, including projecting the 3D covariance into 2D, evaluating spherical harmonics for the colors, sorting the visible Gaussians by depth, and finally rasterizing and alpha-blending all the overlapping splats (as opposed to the point-cloud renderer which essentially just transforms and draws the points).

- Interestingly, the performance difference was much smaller for the bonsai scene, where both renderers were observed running around 160–167 FPS. My assumption is that since the bonsai scene contains significantly fewer Gaussians, the additional computation required by Gaussian rendering is much more manageable, so the performance difference becomes less apparent. Overall, the Gaussian renderer produces a significantly better visual result, but this does come at a higher computational cost, which becomes especially noticeable with larger and more complex scenes.

### 2. For Gaussian Renderer, How Does Changing the Workgroup Size Affect Performance? Why Do You Think This Is?

Changing the workgroup size affects how the GPU distributes and schedules the preprocessing workload. With smaller workgroups (such as 64 threads), more workgroups need to be dispatched to process the same number of Gaussians, which could potentially increase scheduling overhead, although it might also allow the GPU to distribute the workload more flexibly across its compute units. On the other hand, larger workgroups (such as 128 or 256 threads) allow more threads to execute within each workgroup, which could improve GPU utilization and reduce some of the scheduling overhead. However, increasing the workgroup size indefinitely doesn't necessarily mean better performance, since larger workgroups can also increase resource pressure and potentially reduce the number of workgroups that can execute concurrently (which would end up hurting occupancy instead). So, the performance to improve initially as the workgroup size increases from a relatively small value, before eventually leveling off or possibly decreasing. Although, since preprocessing is only one portion of the entire Gaussian rendering pipeline (sorting and rasterization still need to happen afterward), changes to its workgroup size might not have a very significant impact on the overall FPS if another stage ends up being the primary bottleneck.

### 3. Does View-Frustum Culling Give Performance Improvement? Why Do You Think This Is?

I would expect view-frustum culling to improve performance, especially for larger scenes like the bicycle scene, since there could be a pretty substantial number of Gaussians that exist in the scene but are completely outside the camera's current view. Without culling, those Gaussians would still go through the preprocessing, sorting, and potentially rasterization pipeline, even though they wouldn't contribute anything meaningful to the final image (essentially wasting GPU work on things that the user can't even see). With culling enabled, Gaussians whose centers are outside the expanded view-frustum bounds can be discarded early in preprocessing, which reduces the number of Gaussians that need their covariance, color, and other rendering information computed, and also reduces how many elements need to be sorted and drawn afterward. Although every Gaussian still needs to be checked initially, the amount of work performed after that point should be noticeably smaller if a large portion of the scene is outside the camera's view. The actual performance improvement would depend quite heavily on the camera position and how many Gaussians are visible at a given time. For instance, when zooming out and having most of the scene visible, culling probably wouldn't help as much, whereas when zooming into a specific part of the scene, a much larger percentage of the Gaussians could be excluded (and thus the benefits of culling should become more noticeable).

### 4. Does the Number of Gaussians Affect Performance? Why Do You Think This Is?

For this comparison, I used the two scenes from the demonstrations, which have significantly different Gaussian counts.

| Scene | Gaussian Count | Gaussian Renderer (FPS) |
|---|---:|---:|
| Bonsai | 272,956 | ~161.3–166.7 |
| Bicycle | 1,063,091 | ~48.8–73.0 |

- From the results, increasing the number of Gaussians seems to have a pretty significant impact on performance. The bonsai scene, with around 273 thousand Gaussians, was able to run at approximately 160+ FPS, whereas the bicycle scene, with slightly over 1 million Gaussians (almost 4 times as many), had noticeably lower FPS, often sitting around 50–70 FPS during the demonstration.

- This is generally the outcome that I expected, since every additional Gaussian introduces more work for the GPU. During preprocessing, each Gaussian needs to have its 3D position projected onto the screen, its covariance transformed, and its color evaluated using spherical harmonics. Afterward, the visible Gaussians also need to be sorted by depth and rendered as quads, so having more Gaussians naturally increases the amount of computation and memory accesses required per frame (and there's also potentially more overlapping fragments that need to be evaluated and blended together).

- One thing I thinl that is worth noting is that the performance difference isn't necessarily determined by Gaussian count alone, since the actual number of visible Gaussians and their projected sizes also matter quite a lot. For example, even if two scenes contain the same total number of Gaussians, one could still perform worse if its splats cover larger portions of the screen and overlap more frequently, since this would require more fragment processing. Also, since the bicycle and bonsai are different scenes rather than two different Gaussian counts of the exact same scene, their rendering workloads aren't completely identical.

- Overall, the results suggest that larger Gaussian counts generally lead to lower FPS, which makes sense considering the additional preprocessing, sorting, and rasterization required. Compared to the point-cloud renderer, this scaling cost also seems to be much more noticeable for Gaussian rendering, since each Gaussian involves significantly more computation than simply drawing an individual point.



### Credits

- [Vite](https://vitejs.dev/)
- [tweakpane](https://tweakpane.github.io/docs//v3/monitor-bindings/)
- [stats.js](https://github.com/mrdoob/stats.js)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
- Special Thanks to: Shrek Shao (Google WebGPU team) & [Differential Guassian Renderer](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
