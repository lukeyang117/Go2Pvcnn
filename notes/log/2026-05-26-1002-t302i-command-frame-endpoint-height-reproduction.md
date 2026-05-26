# T302i Command-Frame Endpoint And Height Reproduction

## Purpose

Reproduce the user's visual observation that, while crossing a low-small obstacle, swing feet move forward along the command direction but planned touchdowns are later placed behind the swing/foot endpoint, causing a discontinuous-looking trajectory. Also reproduce the supplemental observation that the foot trajectory can rise above the root.

## Stage

`extension/batch_mpc_planner` reachable low-small crossing probe and real IsaacLab viewer/runtime diagnostics.

## Related Todo

- [../todo/T302i-viewer-realized-foot-mismatch.md](../todo/T302i-viewer-realized-foot-mismatch.md)

## Command

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py \
  --device cuda:0 \
  --commands 'mixed_yaw_v050:0.50 0.25 1.00' \
  --variants baseline,reachable_fk_cross_v9 \
  --cycles 1 \
  --requested-n-frames 300 \
  --warmup-steps 6 \
  > tmp/t302i-viewer-realized-foot-mismatch/command_frame_endpoint_height_mixed_yaw_baseline_v9.jsonl 2>&1
```

## Input Conditions

- Real env: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`.
- Terrain: semantic low-small anchor from the task course.
- Command: `vx=0.50, vy=0.25, yaw=1.00`.
- Compared variants: production baseline and debug `reachable_fk_cross_v9`.

## Key Metrics

New diagnostics added to the probe:

- `touchdown_behind_planned_foot_along_max_m`: command-frame max amount by which the touchdown lies behind the planned foot position.
- `touchdown_behind_swing_foot_along_max_m`: same value restricted to swing frames.
- `planned_swing_along_forward_step_max_m`: max forward command-frame swing-foot step.
- `planned_vs_fk_along_error_max_m`: max command-frame along mismatch between exported planned foot and FK-realized foot.
- `planned_swing_foot_above_root_z_max_m` / `fk_swing_foot_above_root_z_max_m`: max swing foot height relative to root height.

Baseline:

- `planned_swing_along_forward_step_max_m=0.347791`
- `touchdown_behind_planned_foot_along_max_m=0.569856`
- `touchdown_behind_swing_foot_along_max_m=0.569856`
- `planned_vs_fk_along_error_max_m=0.356298`
- `command_direction_cosine=-0.595472`
- `terminal_planned_vs_fk_foot_error_max=0.388287`
- `touchdown_ik_fk_error_max=0.661772`
- `planned_swing_foot_above_root_z_max_m=-0.005118`
- `fk_swing_foot_above_root_z_max_m=-0.008270`

V9:

- `planned_swing_along_forward_step_max_m=0.317100`
- `touchdown_behind_planned_foot_along_max_m=0.316026`
- `touchdown_behind_swing_foot_along_max_m=0.314493`
- `planned_vs_fk_along_error_max_m=0.318253`
- `command_direction_cosine=0.998519`
- `terminal_planned_vs_fk_foot_error_max=0.353239`
- `touchdown_ik_fk_error_max=0.745050`
- `root_height_min=0.117291`
- `planned_swing_foot_above_root_z_max_m=0.120522`
- `fk_swing_foot_above_root_z_max_m=0.069278`

## Result

Pass as reproduction. The baseline reproduces the user's main command-frame endpoint conflict: the planned swing foot moves forward by about `0.35m`, while touchdown can sit about `0.57m` behind the planned/swing foot in the same command frame. This matches the visible "forward swing, rear touchdown" discontinuity.

V9 reduces the command-frame rear touchdown amount to about `0.316m`, but the conflict remains. V9 also reproduces the supplemental foot-height issue: planned swing foot rises `0.1205m` above root, and FK-realized swing foot rises `0.0693m` above root. V9 still has poor touchdown IK/FK consistency and low root height, so it remains diagnostic only.

## Conclusion

The visual discontinuity is not just a marker problem. It has a measurable planner-output geometry signature in command frame:

1. swing endpoint follows the command direction;
2. touchdown is still allowed to land behind that endpoint;
3. exported planned foot and FK-realized foot remain mismatched;
4. in V9, foot height can exceed root height.

The next debugging step should inspect which terms generate or overwrite planned touchdown relative to swing foot endpoints, then decide whether the output contract must enforce command-frame endpoint consistency and reachable FK before export.

## Follow-Up

- Add a T302i child node for command-frame endpoint consistency and foot-above-root diagnostics.
- Do not fix by changing viewer markers.
- Keep V9 as visual/debug variant only.

## Git Refs

- Baseline Ref: `c54dc5c`
- Candidate Ref: `working tree`
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py)
