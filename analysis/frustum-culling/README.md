## 3. Effect of view-frustum culling on performance

I compared the existing screen-space culling check with that check disabled in the original local Viewer. Both runs used the bicycle scene with **1,063,091 Gaussians**, camera preset 0, Gaussian scaling 1.00, preprocessing workgroup size 256, and a **1283 × 741** canvas. Near-camera rejection (`view_pos.z < 0.2`) remained enabled in both cases.

After approximately five seconds of warm-up, I sampled the Viewer’s instantaneous FPS display 41 times at approximately 0.5-second intervals per configuration. These are summaries of sampled displayed readings, rather than average throughput calculated from every frame.

| Screen-space culling | Median displayed FPS | Observed displayed FPS range |
| --- | ---: | ---: |
| Disabled | 149.25 | 126.58–172.41 |
| Enabled | 149.25 | 129.87–172.41 |

**This comparison did not show a clear performance improvement from culling.** The medians were identical at the display’s precision, and the observed ranges overlapped substantially. This does not show that culling is always ineffective; the result applies to these short runs at this particular viewpoint.

With culling enabled, the preprocess shader rejects centers outside an expanded screen region (`abs(ndc.x) > 1.2` or `abs(ndc.y) > 1.2`). The margin retains centers near the screen edges. Rejected Gaussians skip the subsequent 2D covariance and conic calculations, spherical-harmonics color evaluation, splat-buffer writes, and insertion into the sorting buffers. Reducing the surviving count can also reduce sorting and submitted rendering work. Earlier operations, including the 3D covariance calculation, still run before this rejection in the current implementation.

The gain depends on how much work is actually removed and on the remaining rendering costs. One possible explanation for the small measured difference is that this viewpoint does not exclude enough expensive work to noticeably change the displayed FPS. Surviving Gaussian counts and individual GPU pass timings were not measured, so this is a hypothesis rather than a confirmed bottleneck. Additional viewpoints and repeated, interleaved runs would be needed for a broader conclusion.

The final code restores screen-space culling. The build passed and no rendering errors were captured in the completed comparison.

### Measurement records

- [Culling disabled: raw readings](culling-off.json) · [Screenshot](culling-off.png)
- [Culling enabled: raw readings](culling-on.json) · [Screenshot](culling-on.png)
