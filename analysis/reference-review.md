# Reference review: presentation and experiment design

The following PR descriptions and their README files were reviewed. This is a
selection of useful examples, not a claim about their grades or a ranking of
all course submissions. Their reported performance numbers were not reused.

| Reference | Useful observed approach | Application in this repository |
| --- | --- | --- |
| [PR #1: Stephen Gavin Sears](https://github.com/CIS5650-Fall-2026/Project5-WebGPU-Gaussian-Splat-Viewer/pull/1), [reviewed README](https://github.com/Gavin-Sears/WebGPU-Gaussian-Splat-Viewer/blob/0051f6e4ee7501002a0d9a474b742401a76acdaa/README.md) | Overview, feature explanations, scene images, and development-stage screenshots make the result easier to inspect. | Replace starter text with an overview, verified feature list, controls, and our own local screenshots. Historic development-stage screenshots are not claimed where they were not captured. |
| [PR #7: Yikai Li](https://github.com/CIS5650-Fall-2026/Project5-WebGPU-Gaussian-Splat-Viewer/pull/7), [reviewed README](https://github.com/sqqqqqqs/Project5-WebGPU-Gaussian-Splat-Viewer/blob/48d1e40c18e5b8ec63092f2be37b8ee4e1292948/README.md) | Fixed measurement conditions, pass-level GPU timings, visible counts, and multiple viewpoints help distinguish costs and explain culling behavior. | State the conditions and exact metric used, attach raw readings, and avoid treating a zero difference at one view as a general result. GPU pass timing and visible-count instrumentation are not implemented here. |
| [PR #12: Jing Huang](https://github.com/CIS5650-Fall-2026/Project5-WebGPU-Gaussian-Splat-Viewer/pull/12), [reviewed README](https://github.com/Stabil1ze/Project5-WebGPU-Gaussian-Splat-Viewer/blob/96152fc19fdd74df24638675ce6f5975d91aeb75/README.md) | Running instructions, controls, a measurement setup, and Gaussian-fraction experiments connect usability with performance and quality. | Compare deterministic subsets of the same scene and show the resulting images. Our subsets are prepared offline by copying every nth complete PLY record, then loaded with the unchanged Viewer. |

## What was independently produced

- The scalar-record PLY sampling utility in `scripts/sample-ply.mjs`.
- Local Viewer experiments and their JSON records.
- Every screenshot and the Gaussian-count sample-distribution plot in this repository's analysis folders.
- The explanations and conclusions written from those experiments.

Only PR metadata and README content were reviewed from these submissions;
no peer shader or renderer implementation was imported. The renderer continues
to use this repository's own implementation and the course-provided radix sorter.

## Remaining work

- Public Pages deployment and a 42-second demonstration video are now complete;
  see the main README and [verification notes](demo/README.md).
- For stronger performance conclusions, add repeated runs, other viewpoints,
  and optionally GPU stage timing through an independently designed implementation.
