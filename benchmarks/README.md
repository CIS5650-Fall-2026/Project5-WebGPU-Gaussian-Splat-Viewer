# External Benchmark

`run-benchmark.mjs` drives the existing viewer through the Chrome DevTools
protocol, samples `requestAnimationFrame` intervals, and writes metrics as JSON.
It does not modify or import application code, and it does not add benchmark
controls to the viewer. Measurements are end-to-end frame timings rather than
GPU timestamp-query stage timings.

```powershell
npm run benchmark -- `
  --ply scenes/bezier/scene_frame0.ply `
  --cameras scenes/bezier/cameras.json `
  --warmup 5 `
  --seconds 10 `
  --runs 3
```

The default report includes mean, median, minimum FPS, median/p95/p99 frame
time, and raw frame intervals for every run. Results are written to
`benchmarks/results/` and are ignored by Git.

Use `--headed` only when a visible browser window is needed for debugging.

Use `--precision f16` to select the optional half-precision preprocessing
shader. The default is `f32`; pages fall back to `f32` automatically when the
adapter does not expose `shader-f16`.
