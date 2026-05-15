# 2026-05-13 18:15 MPC Command Switch Gait Probe

## Purpose

Answer whether the current-code gait failure was tested under command switching, not only steady-state speed segments.

## Stage

T300d production `extension/batch_mpc_planner` viewer-style IsaacLab runtime diagnostics.

## Related Todo

- [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)

## Procedure

Used [../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py](../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py) with explicit command-switch sequences:

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
MPC_TEST_DEVICE=cuda:2 \
MPC_YAW_GAIT_CYCLES=10 \
MPC_YAW_GAIT_SEQUENCES='forward_to_yaw_left:forward,yaw_left;forward_to_yaw_right:forward,yaw_right;backward_to_yaw_left:backward,yaw_left;backward_to_yaw_right:backward,yaw_right;lateral_left_to_yaw_left:lateral_left,yaw_left;lateral_left_to_yaw_right:lateral_left,yaw_right;lateral_right_to_yaw_left:lateral_right,yaw_left;lateral_right_to_yaw_right:lateral_right,yaw_right;yaw_left_to_forward:yaw_left,forward;yaw_right_to_forward:yaw_right,forward;yaw_left_to_lateral_left:yaw_left,lateral_left;yaw_right_to_lateral_right:yaw_right,lateral_right;forward_backward_switch:forward,backward,forward;lateral_lr_switch:lateral_left,lateral_right,lateral_left;yaw_lr_switch:yaw_left,yaw_right,yaw_left' \
MPC_YAW_GAIT_OUTPUT=/tmp/mpc_gait_switch_matrix_10.jsonl \
timeout 1200s python Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py
```

## Input Conditions

- Device: `cuda:2`
- Runtime: IsaacLab headless via `env_isaacsim`
- Planner backend: `mpc`
- Cycles per segment: `10`
- Segment summaries: `33`
- Exceptions: `0`

## Key Metrics

| Switch group | Segments | Actual abs gap mean | Actual air ratio mean | Max actual gap | Actual-plan foot err | Front CV | Switch sum mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| switch into yaw | `8` | `0.0494` | `0.4062` | `0.1965` | `0.0392` | `0.2777` | `0.50` |
| switch out of yaw | `4` | `0.0789` | `0.7500` | `0.1867` | `0.0667` | `0.0842` | `0.00` |
| forward/backward switch | `3` | `0.0024` | `0.0000` | `0.0000` | `0.0085` | `0.1966` | `0.00` |
| lateral left/right switch | `3` | `0.0332` | `0.3500` | `0.2002` | `0.0298` | `0.0838` | `0.00` |
| yaw left/right switch | `3` | `0.1822` | `0.8667` | `0.3606` | `0.1102` | `0.5399` | `7.33` |

Representative high-risk switches:

| Sequence / Segment | Actual abs gap | Actual air ratio | Actual-plan foot err | Front CV |
| --- | ---: | ---: | ---: | ---: |
| `yaw_left_to_forward / forward` | `0.1199` | `1.0000` | `0.0921` | `0.0878` |
| `yaw_left_to_lateral_left / lateral_left` | `0.1337` | `1.0000` | `0.0923` | `0.1888` |
| `yaw_lr_switch / yaw_right` | `0.1804` | `1.0000` | `0.1111` | `0.6740` |
| `yaw_lr_switch / yaw_left` | `0.3047` | `1.0000` | `0.1644` | `0.6120` |
| `lateral_lr_switch / lateral_left` after LR switch | `0.0878` | `0.7500` | `0.0729` | `0.1620` |

## Result

Pass as command-switch reproduction.

Switching into yaw is bad, switching out of yaw is worse on actual stance-ground metrics, and yaw-left/yaw-right switching is the worst tested transition. Pure forward/backward switching is clean. Lateral left/right switching has a smaller but real stance-ground degradation.

## Conclusion

The current failure is not just steady yaw. The command-switch boundary, especially yaw exit and yaw left/right reversal, is a distinct high-risk case and must be included in the regression gate for any planner fix.

## Follow-Up

- Any yaw fix should test steady yaw, linear+yaw combos, switch into yaw, switch out of yaw, and yaw left/right reversal.
- Preserve clean forward/backward switching.
- Watch lateral left/right switching as a secondary regression guard.

## Git Refs

- Baseline Ref: working tree on top of `e90e3a4`
- Candidate Ref: working tree with updated [../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py](../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py)
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py](../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py](../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py)
