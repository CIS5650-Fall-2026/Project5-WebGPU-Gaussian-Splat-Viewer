# Demo and deployment verification

## Recording

- [42-second MP4](../../public/demo.mp4), silent H.264, 1280 × 740, 30 FPS.
- The original local Viewer loaded the full bicycle PLY and its matching camera file.
- Real Chrome tab frames were recorded for approximately 42 seconds. The source
  frame timestamps were retained when encoding 1,260 output frames; this is not
  an animation assembled from the earlier benchmark screenshots.
- The capture includes Gaussian rendering, the point-cloud baseline, Gaussian
  scale changes to 0.50 and 1.50, and camera presets 0, 1, and 2.
- The video was decoded in full without reported errors. Extracted frames from
  the point-cloud, reduced-scale, and alternate-camera portions were inspected.
- The recorded FPS display is not used for performance measurements.

## Public deployment

- URL: [online Viewer](https://ronan83.github.io/Project5-WebGPU-Gaussian-Splat-Viewer/).
- The initial 404 came from an unconfigured Pages source and disabled Actions
  on the fork. Actions were enabled and Pages was set to use GitHub Actions.
- The Vite base is normalized to `/Project5-WebGPU-Gaussian-Splat-Viewer/` for
  the deployment build, while the local default remains `/`.
- The [first successful deployment run](https://github.com/Ronan83/Project5-WebGPU-Gaussian-Splat-Viewer/actions/runs/40618178031)
  deployed commit `bb8f5cf042e6970f902cec9db1cb625948db3976`.
- The site returned HTTP 200. The deployed app was then opened in Chrome,
  loaded with 1,063,091 bicycle Gaussians and 194 camera presets, and switched
  to Gaussian rendering. The live image was inspected and no browser errors
  were captured during this verification.
- [Online runtime screenshot](live-viewer.png).

Scene files are not bundled into the site. Users select their downloaded PLY
and matching camera JSON through the existing file controls.
