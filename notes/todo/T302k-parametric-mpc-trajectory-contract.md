# T302k Parametric MPC Trajectory Contract

## Current State

- T302k is the active implementation front for `Go2Pvcnn/extension/batch_mpc_planner`.
- The old dense residual planner route is retired. Current `plan_segment()` is parametric-only.
- Current active route is the approved low-small loss redesign plan: [T302k low-small loss redesign plan](T302k-low-small-loss-redesign-plan.md).
- Design source: [../../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html](../../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html).
- Task 1 implementation restored the nominal extraction contract:
  - `semantic_policy.py` owns `ParametricTrajectoryNominal` and `build_parametric_nominal()`;
  - `planner.py` builds nominal once before the Adam loop and passes `nominal.command` as the planning command;
  - `decode_parametric_trajectory()` now consumes `nominal + variables` and no longer performs high/large semantic search inside decode.
- Task 2 added optional plane-terrain metadata:
  - `MpcPlannerTerrain.is_plane_terrain` is optional and remains `None` if unavailable;
  - scanner terrain construction, subset helpers, planner normalization, and MPC manager preserve it;
  - MPC manager infers plane terrain from IsaacLab `terrain_types` and `terrain_generator.sub_terrains` names, where only `flat` and `plane` are treated as plane.
- Task 3 added GPU low-small component circle approximation:
  - `semantic_geometry.py` provides `LowSmallCircles` and `low_small_component_circles()`;
  - helper returns fixed-shape centers/radii/valid/truncated tensors on the input device.
- Task 4 replaced the sampled low-small crossing loss key:
  - `parametric_losses.py` provides `parametric_touchdown_keepout_loss()`;
  - `_parametric_sampled_frame_losses()` now emits `parametric_touchdown_keepout` instead of `parametric_low_small_crossing`;
  - `low_small_crossing` config remains only for shared height-threshold/classification context and standalone legacy progress-loss tests.
- Task 5 added sampled swing target terrain clearance:
  - `parametric_losses.py` provides `parametric_swing_foot_clearance_loss()`;
  - `_parametric_sampled_frame_losses()` emits `parametric_swing_foot_clearance`.
- Task 6 added final FK realized collision:
  - `MpcLegPoints` exposes `shank_pos_world`;
  - `parametric_losses.py` provides `parametric_fk_body_leg_collision_loss()`;
  - final `loss_breakdown/cost_breakdown` includes `parametric_fk_body_leg_collision`;
  - limitation: this key is computed after optimization and FK solve, not inside Adam yet.
- Task 7 added final FK consistency:
  - `parametric_losses.py` provides `parametric_trajectory_fk_consistency_loss()`;
  - final `loss_breakdown/cost_breakdown` includes `parametric_trajectory_fk_consistency`.
- Task 8 added plane-only root z target:
  - `parametric_losses.py` provides `parametric_plane_root_z_target_loss()`;
  - sampled `loss_breakdown/cost_breakdown` includes `parametric_plane_root_z_target`, gated by `terrain.is_plane_terrain`.
- Current trajectory contract:
  - optimize touchdown `xy`; derive touchdown `z` from `height_at(terrain, touchdown_xy)`;
  - build root and foot cubic curves over the configured horizon;
  - sample frames for losses;
  - solve clamped IK and export FK-realized future feet;
  - keep frame0 root/foot state aligned to current IsaacLab state.
- Approved low-small redesign constraints:
  - 第 0 条 is test/diagnostic only, not an optimizer loss;
  - new losses are touchdown circle keepout, swing target terrain clearance, FK body/leg collision, optimized-vs-FK trajectory consistency, and plane root-z target;
  - no decode-time hard projection, touchdown snapping, or hard foot separation;
  - adding any new loss requires user approval.
- Parent evidence pages:
  - [T302h](T302h-semantic-obstacle-jitter-reproduction.md): semantic obstacle/jitter reproduction history.
  - [T302i](T302i-viewer-realized-foot-mismatch.md): IK/FK reachability mismatch history.
  - [T302j](T302j-touchdown-endpoint-consistency.md): touchdown endpoint/export history.

## Open Children

| Child | Status | Priority | Purpose | Primary Files |
| --- | --- | --- | --- | --- |
| T302k.18 | active | P0 | Execute the approved low-small loss redesign plan as the current mainline. | [T302k-low-small-loss-redesign-plan.md](T302k-low-small-loss-redesign-plan.md) |
| T302k.12 | active | P0 | Parent reachability/collision problem addressed by T302k.18: FK mismatch, small-obstacle collision, root/foot relative drift. | `planner.py`, `parametric.py`, `kinematics.py`, `mpc_low_small_reachable_crossing_probe.py` |
| T302k.17 | verify | P0 | Prerequisite in T302k.18 restored locally: decode consumes `nominal + variables`; commit pending in Task 1 Step 7. | `semantic_policy.py`, `parametric.py`, `planner.py`, `test_batch_mpc_parametric.py`, `test_batch_mpc_backend.py` |

## Closed Children Archive

- T302k.1 done: Added parametric command-frame helpers and curve sampling tests.
- T302k.2 done: Added `MpcParametricVariables` and initialization tensors.
- T302k.3 done: Added 25-frame parametric root/foot decode with grounded touchdown `z`.
- T302k.4 done: Integrated clamped IK and FK-realized default output.
- T302k.5 verify: Added sampled parametric losses and Adam optimization over parametric variables.
- T302k.6 verify: Added low-small parametric crossing probe entrypoints.
- T302k.8 done: Removed obsolete dense residual modules and feature switch.
- T302k.9 verify: Added parametric semantic avoidance, endpoint, foot-height, touchdown semantic-ground, and spacing losses; high/large acceptance still needs real rerun under current structure.
- T302k.10 verify: Foot curves now use per-leg local swing phase, preserving diagonal trot alternation.
- T302k.11 verify: Parametric replan output preserves current IsaacLab foot positions at frame0.
- T302k.13 verify: Viewer MPC body/root-frame `vx/vy` is rotated to world frame before planning.
- T302k.14 verify: Root roll/pitch follows contact-weighted foot support plane after frame0.
- T302k.15 verify: Full-cycle terminal feet anchor to a canonical body-yaw footprint; major repeated-replan foot drift is reduced.
- T302k.16 verify: Touchdown semantic/spacing losses exist; decode-time hard repair was removed for loss-only visualization.

## Related Logs

- Design commit `97c5b60`: [../../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html](../../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html)
- [../log/2026-05-28-2125-t302k-plane-root-z-target.md](../log/2026-05-28-2125-t302k-plane-root-z-target.md)
- [../log/2026-05-28-2117-t302k-fk-trajectory-consistency.md](../log/2026-05-28-2117-t302k-fk-trajectory-consistency.md)
- [../log/2026-05-28-2110-t302k-fk-body-leg-collision.md](../log/2026-05-28-2110-t302k-fk-body-leg-collision.md)
- [../log/2026-05-28-2057-t302k-swing-target-clearance.md](../log/2026-05-28-2057-t302k-swing-target-clearance.md)
- [../log/2026-05-28-2048-t302k-touchdown-circle-keepout.md](../log/2026-05-28-2048-t302k-touchdown-circle-keepout.md)
- [../log/2026-05-28-2034-t302k-low-small-gpu-circles.md](../log/2026-05-28-2034-t302k-low-small-gpu-circles.md)
- [../log/2026-05-28-2025-t302k-plane-terrain-metadata.md](../log/2026-05-28-2025-t302k-plane-terrain-metadata.md)
- [../log/2026-05-28-2014-t302k-nominal-extraction-contract.md](../log/2026-05-28-2014-t302k-nominal-extraction-contract.md)
- [../log/2026-05-26-2133-t302k-body-relative-foot-anchor-fix.md](../log/2026-05-26-2133-t302k-body-relative-foot-anchor-fix.md)
- [../log/2026-05-26-2040-t302k-long-step-root-relative-foot-drift-repro.md](../log/2026-05-26-2040-t302k-long-step-root-relative-foot-drift-repro.md)
- [../log/2026-05-26-2021-t302k-support-plane-root-roll-pitch.md](../log/2026-05-26-2021-t302k-support-plane-root-roll-pitch.md)
- [../log/2026-05-26-1949-viewer-mpc-body-frame-command.md](../log/2026-05-26-1949-viewer-mpc-body-frame-command.md)
- [../log/2026-05-26-1757-t302k-dense-path-retirement.md](../log/2026-05-26-1757-t302k-dense-path-retirement.md)
- [../log/2026-05-26-1717-t302k-isaaclab-current-foot-touchdown-check.md](../log/2026-05-26-1717-t302k-isaaclab-current-foot-touchdown-check.md)
- [../log/2026-05-26-1713-t302k-parametric-current-foot-replan-anchor.md](../log/2026-05-26-1713-t302k-parametric-current-foot-replan-anchor.md)
- [../log/2026-05-26-1649-t302k-parametric-trot-phase-foot-curves.md](../log/2026-05-26-1649-t302k-parametric-trot-phase-foot-curves.md)
- [../log/2026-05-26-1554-t302k-parametric-semantic-endpoint-losses.md](../log/2026-05-26-1554-t302k-parametric-semantic-endpoint-losses.md)
- [../log/2026-05-26-1524-t302k-parametric-sampled-loss-and-isaaclab-smoke.md](../log/2026-05-26-1524-t302k-parametric-sampled-loss-and-isaaclab-smoke.md)
- [../log/2026-05-26-1450-t302k-parametric-default-fk-output.md](../log/2026-05-26-1450-t302k-parametric-default-fk-output.md)

## Git Refs

- Last Feature Commit: `1b799cd` (parametric helper module)
- Last Verified Commit: `2ac52e2` plus design-only commit `97c5b60`
- Current Work Ref: Task 8 plane root-z target pending commit
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py](../../Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py](../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/kinematics.py](../../Go2Pvcnn/extension/batch_mpc_planner/kinematics.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_parametric.py](../../Go2Pvcnn/tests/test_batch_mpc_parametric.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)

## Next Step

- Commit Task 8 plane root-z target, then continue with Task 9: plane low-small FK semantic collision probe.
- Use IsaacLab `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python` and `CUDA_VISIBLE_DEVICES` for plane low-small FK semantic collision verification.

## Node Details

### T302k.12 Replan Touchdown And FK Reachability

- why-created: IsaacLab checks show frame0 foot state is aligned, while planned touchdowns and clamped FK can still differ by large margins.
- evidence: `max_replan_initial_touchdown_to_current_foot_error` around `0.44-0.70m`; `max_touchdown_ik_fk_error` around `0.66-0.70m` in recent logs.
- current direction: solve in the parametric trajectory contract, probably by coupling touchdown anchors, swing endpoints, and reachable FK output rather than adding dense per-frame residuals.

### T302k.17 Nominal Extraction

- why-created: `decode_parametric_trajectory()` mixed nominal construction, semantic high/large command shaping, and optimization decode.
- current plan contract:
  - build `ParametricTrajectoryNominal` in `semantic_policy.py`;
  - pass `nominal.command` as planning command;
  - construct nominal before `for _ in range(steps):`;
  - decode consumes `nominal` and variables only.
- status: verify locally; commit pending.

### T302k.18 Low-Small Loss Redesign Plan

- why-created: user approved a new loss-only design for small obstacles and requested the todo page to act as the detailed implementation plan.
- plan: [T302k-low-small-loss-redesign-plan.md](T302k-low-small-loss-redesign-plan.md).
- hard guard:
  - no hard projection;
  - no touchdown snapping;
  - no hard foot separation;
  - tune only confirmed weights/parameters unless user approves a new loss.
