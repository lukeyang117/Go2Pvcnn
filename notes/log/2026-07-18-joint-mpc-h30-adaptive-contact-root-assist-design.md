# Joint MPC H30 Adaptive Contact And Root Assist Design

## Purpose

Record the approved design decisions that extend T302v.7 after the H16 small-obstacle attitude and airborne-touchdown diagnosis.

## Stage

T302v.8 design; Stage A remains open, Stage B remains paused and incomplete.

## Related Todo

- [T302v joint MPC RTI GPU](../todo/T302v-joint-mpc-rti-gpu.md)
- [Investigation dashboard](../todo.md)

## Inputs And Decisions

- Inherit `docs/superpowers/plans/2026-07-17-joint-mpc-rti-root-joint-coupled-gait-implementation-plan.md`; no historical Stage A result is accepted because the original gate was not closed.
- Fix production horizon to H30 with `dt=0.02`, `half_cycle_steps=15`, nominal stance/swing 15 frames each.
- Use per-leg touchdown confirmation, up to 10 extension frames, independent touchdown, protected liftoff, and recovery without forced stance.
- Require grounded touchdown plus foot/knee/calf/thigh/base small-obstacle safety and x1/x2 stance lookahead safety before establishing a stance anchor.
- Plan root lateral/roll/pitch/yaw in the same LQ with SDF/phase/reachability soft release and cumulative plus per-frame clamps.
- Use medium clamps: lateral `0.06m`/`0.20m/s`, roll/pitch `6deg`/`0.6rad/s`, yaw `10deg`/`0.8rad/s` relative to command nominal.
- Expand parallel line search to `(1.0, 0.5, 0.25, 0.1)`.
- Extend the shared JointMetrics contract with root roll/pitch deviation, adaptive-contact/recovery, root-assist clamps, knee/base small collision, and alpha histograms/rejection reasons.
- Use the full signed command Cartesian product: `vx={0,±0.2,±0.4,±0.6,±0.8,±1.0}`, `vy={0,±0.3,±0.5}`, and `yaw={0,±0.5,±1.0}`, for 275 combinations before shape/phase/placement expansion.

## Evidence Carried Into The Design

- Small-obstacle worst roll/pitch: `26.23deg` / `25.84deg`; flat controls: `7.02deg` / `6.63deg`.
- Airborne touchdown above 20mm: `2677/5760`; above 5mm: `3837/5760`.
- Small-obstacle alpha=0: `6092/23040 = 26.44%`.
- Collision frames: foot `75`, knee `38`, calf `144`, thigh `1`, base `15`; most foot/calf collision frames occur in continuing stance.

## Result

Chinese HTML design written at:

- [H30 adaptive contact and root assist design](../../docs/superpowers/specs/2026-07-18-joint-mpc-rti-h30-adaptive-contact-root-assist-design.html)

The design fixes the Stage B performance contract to idle-GPU realistic `1024 x H30 x 1000 <=5.0s`, preserves the original Stage A -> Stage B -> same-candidate Stage C completion rule, and records the paused CUDA Graph/Triton/associative-scan progress.

## Verification

- Placeholder scan: no `TBD` or `TODO`.
- Inheritance, H30 timing, per-leg scheduler, collision parts, root clamps, JointMetrics, Stage A/B/C, paused Stage B progress and completion definition are all present.
- The formal Stage A/Stage C command matrix records all 275 raw `(vx,vy,yaw_rate)` keys; the older `0.1/0.2/0.4` matrix remains evidence only.
- No planner implementation was changed by this design pass.

## Follow-up

User reviews the HTML. After approval, amend the inherited 2026-07-17 implementation plan rather than starting an unrelated plan.

## Git Refs

- Baseline Ref: `6739d16`
- Candidate Ref: `joint_mpc` working tree
- Key Files: `docs/superpowers/specs/2026-07-18-joint-mpc-rti-h30-adaptive-contact-root-assist-design.html`, `Go2Pvcnn/tests/joint_mpc_rti/joint_metrics.py`, `Go2Pvcnn/extension/joint_mpc_rti/planner.py`
