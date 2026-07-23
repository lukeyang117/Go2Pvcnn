# Joint MPC RTI Final Flat Gate First Diagnosis

## Purpose

Continue final-plan Task 16 from the Task 15 checkpoint and diagnose the first monitored 19-cell flat gate.

## Stage And Todo

- Stage: final pure-kinematic Joint MPC RTI, Task 16 flat acceptance
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Baseline Ref: `aaca7cc85186f2b711001c22f3616bb76910b1e5`
- Candidate Ref: uncommitted `work/joint-mpc-kinematic`

## Changes Under Test

- Future swing nodes use full analytic IK instead of returning toward shifted warm joints.
- Swept checks use 48 subdivisions; touchdown selector uses 65 swing samples.
- Swept support contact permits only sole-safe interpolation contact; discrete nodes remain strict.
- Preview tail holds terminal root z/roll/pitch and carries an explicit contact mask.
- Primary and preview touchdown regions are checked only over their own stance intervals.
- CUDA fixed SPD solve makes matrix and RHS contiguous before Triton flat indexing.

## TDD And Root Causes

1. Zero command failed at refresh 8 because future swing warm blending penetrated flat ground and the preview contact semantics did not match the modeled tail.
2. The preview root z/roll/pitch trend extrapolated 12 nodes and could drive root z from about `0.345m` to `0.428m`, violating leg joint bounds.
3. Line search applied the primary touchdown region to later preview stance nodes.
4. The first formal CUDA gate selected alpha zero nearly everywhere. CPU/CUDA scan parity isolated a non-contiguous SPD matrix passed to a Triton kernel that assumes flat contiguous storage. B=3 synthetic direction was CPU `0.1906` versus CUDA `585454.75`; contiguous input reduced production-float parity error to `1.79e-6`.

## Verification

- Nominal/swept/line-search/pipeline focused union: `51` tests passed.
- Trajectory scan including CUDA parity: `9 passed`.
- Current-map B=3, 24 refreshes, commands `vx=[0,0.2,0.8]`: valid and nominal counts `[24,24,24]`.
- First monitored formal flat before CUDA fix: `19/19` cells failed, `47.438s`, about `1.84GiB` GPU memory.
- Second monitored formal flat after CUDA fix: `19/19` cells still failed, `47.629s`, about `1.84GiB` GPU memory.

## Remaining Flat Failures

- Zero command: publish/valid are `1.0`, but alpha-zero ratio `0.73793`, max run `10`, joint step `0.52842rad`, and strict zero-drift/carry/lead metrics fail.
- `|vx|>=0.4`: nominal-only behavior remains dominant; root velocity error is approximately `0.55*|vx|`, matching `nominal.command_scale=0.45`.
- Lateral `|vy|=0.3/0.5`, `vx=-1`, and yaw `|wz|=1` still lose publication during parts of the trace.
- Across moving cells, nonzero QP directions are frequently rejected by stance XYZ and nonlinear node/swept safety.
- Cold/startup joint step remains above the `0.35rad` gate.

## Conclusion And Follow-Up

CUDA scan corruption is fixed and guarded. Task 16 remains RED on real trajectory behavior. Continue with owner-level RED tests for cold first-step continuity, feasible stance-preserving QP directions, and command tracking scale; do not start small, large, viewer, or performance acceptance.
