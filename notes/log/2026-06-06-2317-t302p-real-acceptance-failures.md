# T302p Real Acceptance Failures

## Purpose

Record the real IsaacLab acceptance continuation for T302p after the command-frame implementation. This pass checks whether the coordinate-frame fix closes the flat all-direction direction metrics and preserves low-small semantic compatibility.

## Stage

MPC semantic policy evaluation / batch MPC command-frame acceptance.

## Related Todo

- [../todo/T302p-mpc-command-frame-alignment-plan.md](../todo/T302p-mpc-command-frame-alignment-plan.md)
- Previous implementation log: [2026-06-06-1858-t302p-command-frame-implementation.md](2026-06-06-1858-t302p-command-frame-implementation.md)

## Commands

Eight-direction flat tracking run, sequential one-command IsaacLab launches:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python - <<'PY'
# sequentially ran mpc_policy_eval.py for:
# forward, backward, left, right, diag_fl, diag_fr, diag_bl, diag_br
# each with --mode tracking --headless --device cuda:0 --num-envs 1
# --num-rounds 1 --max-steps 120 --terrain-rows 0 --terrain-cols 0
PY
```

Aggregate output:

```text
logs/mpc_policy_eval/t302p_eight_direction_120step/aggregate.json
```

Low-small semantic compatibility regression:

```bash
CUDA_VISIBLE_DEVICES=3 timeout 300s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py \
  --device cuda:0 \
  --variants parametric_v1 \
  --requested-n-frames 300 \
  --warmup-steps 6 \
  > tmp/t302p-command-frame/low_small_regression_gpu3.jsonl 2>&1
```

## Input Conditions

- Policy checkpoint: `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic/2026-05-31_20-03-27/model_14000.pt`
- Eight-direction commands:
  - `[1.0, 0.0, 0.0]`
  - `[-1.0, 0.0, 0.0]`
  - `[0.0, 1.0, 0.0]`
  - `[0.0, -1.0, 0.0]`
  - `[0.7, 0.7, 0.0]`
  - `[0.7, -0.7, 0.0]`
  - `[-0.7, 0.7, 0.0]`
  - `[-0.7, -0.7, 0.0]`
- Public command contract remained body/root-yaw frame.
- No runtime code changes were made during this continuation.

## Eight-Direction Result

All eight `mpc_policy_eval.py` processes exited `0`. Command-source equality passed for every command:

```text
command_body_match_max_abs_error == 0.0
```

Direction acceptance failed:

| Command | Root Cos | Root Lat | Min Leg Cos | Max Leg Lat | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| forward | 0.991923 | 0.126840 | 0.966787 | 0.255585 | root + leg fail |
| backward | 0.997631 | 0.068792 | 0.912456 | 0.409176 | leg fail |
| left | 0.969988 | 0.243155 | 0.585759 | 0.810486 | root + leg fail |
| right | 0.959527 | 0.281618 | 0.860883 | 0.508804 | root + leg fail |
| diag_fl | 0.998694 | 0.051099 | 0.860411 | 0.509602 | leg fail |
| diag_fr | 0.999775 | 0.021204 | 0.958868 | 0.283851 | leg fail |
| diag_bl | 0.997176 | 0.075098 | 0.934038 | 0.357174 | leg fail |
| diag_br | 0.998921 | 0.046438 | 0.885851 | 0.463971 | leg fail |

Root-level failed commands:

```text
forward, left, right
```

Moving-leg direction failed for all eight commands under the current hard threshold.

## Low-Small Result

The low-small probe process exited `0`, but hard compatibility metrics failed.

Parsed JSON rows:

```text
json_rows = 7
covered crossing rows = 4
```

Hard metric failures:

```text
max fk_semantic_collision_count = 21
max fk_semantic_collision_rate = 0.0175
max planned_vs_fk_foot_error_crossing_leg_max_m = 0.13923048973083496
```

Failing rows:

- `lateral_v050`: `fk_semantic_collision_count=21`, `fk_semantic_collision_rate=0.0175`, `planned_vs_fk_foot_error_crossing_leg_max_m=0.0697169080376625`
- `mixed_yaw_v050`: `fk_semantic_collision_count=1`, `fk_semantic_collision_rate=0.0008333333333333334`, `planned_vs_fk_foot_error_crossing_leg_max_m=0.13923048973083496`

Rows that stayed clean for FK semantic collision:

- `forward_v050`: no FK semantic collision but `crossing_leg_count=0`
- `diagonal_v050`: no FK semantic collision, crossing error `0.04160968214273453`
- `yaw100`: no FK semantic collision, crossing error `0.009545918554067612`

## Result

T302p remains open. The command source contract is verified in real IsaacLab, but the behavioral acceptance is not met:

- flat all-direction root XY fails for forward/left/right
- moving-leg XY direction fails for all eight tested commands
- low-small semantic compatibility regressed for lateral and mixed-yaw rows

## Conclusion

The coordinate-frame boundary fix is necessary but not sufficient. The next debugging step should locate whether the remaining failures come from:

- metric definition using terminal reference-cache displacement rather than per-step/phase-aware motion
- MPC planned root/foot trajectory still using an unaligned heading path not covered by the previous audit
- policy/reference interaction under side and diagonal commands
- low-small lateral/mixed-yaw geometry after body-command rotation

No fix should be applied before tracing one failing direction, preferably `left` for root-direction failure and `lateral_v050` for low-small semantic collision.

## Follow-Up

- Keep T302p.2 active.
- Do not close T302p acceptance.
- Start systematic debugging from:
  - `logs/mpc_policy_eval/t302p_eight_direction_120step/left/.../metrics.jsonl`
  - `tmp/t302p-command-frame/low_small_regression_gpu3.jsonl`

## Git Refs

- Baseline Ref: local T302p command-frame implementation dirty worktree.
- Candidate Ref: same code; verification-only continuation.
- Key Files:
  - `Go2Pvcnn/scripts/mpc_policy_eval.py`
  - `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`
  - `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`
  - `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
  - `Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py`
  - `Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py`
