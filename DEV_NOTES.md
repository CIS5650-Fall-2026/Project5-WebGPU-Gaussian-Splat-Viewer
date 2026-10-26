# Dev Notes (scratch log for the README)

Working branch: `faris/dev`. These notes record what was implemented, design decisions,
and gotchas so the final README can be written from them.

## Part 1 — Point cloud MVP
- `point_cloud.wgsl`: `clip = proj * view * pos` (model matrix is identity; the PLY is already in world space).
- Camera conventions (from `camera.ts`): view space is +Z forward (`clip.w = view.z`),
  projection flips Y, near = 0.01, far = 100.
