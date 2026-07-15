# 2026-07-07 MPC QP Idle Jitter Fix

## Purpose

Fix the fast zero-command jitter seen in `mpc_qp` visualization. The target symptom was rapid idle foot/joint/root motion when no key was pressed.

## Stage

MPC-QP backend / continuous gait and viewer runtime config.

## Related Todo

[T302v MPC QP backend](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Procedure

- Added focused regression coverage for:
  - viewer CLI `--n-frames`, `--plan-dt`, and `--qp-iterations` propagation into `mpc_qp_planner_cfg`
  - zero-command gait switching to all stance
  - zero-command repeated replans anchoring the current joint state and limiting jumps
  - stance anchors staying terrain-bound after continuous decode
- Ran focused pytest and real GPU idle planner probes with `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`.
- Ran full `Go2Pvcnn/tests/test_mpc_qp_backend.py` to record remaining non-idle failures.

## Key Changes

- `alternating_diagonal_gait_masks()` now accepts `command` and `idle_command_threshold`; idle rows become all-stance.
- `plan_segment_qp()` passes command-aware gait masks and adds `qp_idle_all_stance_active`.
- Idle rows are anchored to the incoming root pose and joint angles, with all contact states true.
- Viewer runtime CLI overrides now sync horizon, replan interval, dt, and QP iterations into the QP config when `--planner-backend mpc_qp` is selected.
- Continuous control initialization binds stance start `P0.z` to terrain so fixed-gait stance anchors do not inherit stale airborne foot z.
- Continuous decode preserves stance anchors while swing frames use IK/FK readback, so contact feet remain stable and swing output stays executable.

## Verification

Focused pytest:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q -k "viewer_cli_overrides_sync_runtime_to_qp_cfg or gait_masks_use_all_stance_for_zero_command_idle or zero_command_idle_anchors_joint_state or fixed_gait_keeps_stance_feet_anchored or fixed_gait_places_touchdown_at_swing_to_stance_boundary or fixed_gait_stance_anchors_bind_start_feet_to_terrain or continuous_reports_fk_readback_error_without_repair"
```

Result: `7 passed, 57 deselected`.

Real GPU idle probe:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python <inline idle probe>
```

Key metrics:

- first cycle all stance: `true`
- second cycle all stance: `true`
- foot frame jump max: `0.0m`
- joint frame jump max: `0.0rad`
- root frame jump max: `0.0m`
- replan-boundary joint delta: `0.0rad`
- replan-boundary foot delta: `0.0m`
- `qp_idle_all_stance_active=1.0`

Full QP suite:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
```

Result: `53 passed, 11 failed`.

The remaining failures are not idle regressions. They are the existing low-small / edge / reachability issues around semantic touchdown improvement, root progress, FK terrain penetration on height edges, and high-arc reachable crossing/readback.

## Conclusion

Idle fast jitter is fixed in the planner output and in repeated replans. Full flat-small and edge acceptance remains open and should continue under the existing coupled loss / QP-iteration tuning direction, without hard repair or candidate search.

## Git Refs

- Baseline Ref: dirty worktree before this pass
- Candidate Ref: dirty worktree after this pass
- Key Files:
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/gait.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/continuous.py`
  - `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
  - `Go2Pvcnn/tests/test_mpc_qp_backend.py`
