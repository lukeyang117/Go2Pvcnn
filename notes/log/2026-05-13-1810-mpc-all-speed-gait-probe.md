# 2026-05-13 18:10 MPC All-Speed Gait Probe

## Purpose

Extend the standalone yaw gait failure probe to front/back, lateral, yaw, speed levels, and linear+yaw combinations so the remaining current-code failure is not overfit to pure yaw-only commands.

## Stage

T300d production `extension/batch_mpc_planner` viewer-style IsaacLab runtime diagnostics.

## Related Todo

- [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)

## Procedure

Updated [../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py](../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py) with custom command aliases for slow/fast linear, lateral, yaw, and linear+yaw combinations.

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
MPC_TEST_DEVICE=cuda:2 \
MPC_YAW_GAIT_CYCLES=12 \
MPC_YAW_GAIT_SEQUENCES='forward:forward;backward:backward;lateral_left:lateral_left;lateral_right:lateral_right;yaw_left:yaw_left;yaw_right:yaw_right;forward_speeds:forward_slow,forward,forward_fast;backward_speeds:backward_slow,backward,backward_fast;lateral_speeds:lateral_left_slow,lateral_left,lateral_left_fast,lateral_right_slow,lateral_right,lateral_right_fast;yaw_speeds:yaw_left_slow,yaw_left,yaw_left_fast,yaw_right_slow,yaw_right,yaw_right_fast;linear_yaw_combos:forward_yaw_left,forward_yaw_right,backward_yaw_left,backward_yaw_right,lateral_left_yaw_left,lateral_left_yaw_right,lateral_right_yaw_left,lateral_right_yaw_right' \
MPC_YAW_GAIT_OUTPUT=/tmp/mpc_gait_all_speeds_12.jsonl \
timeout 1200s python Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py
```

## Input Conditions

- Device: `cuda:2`
- Runtime: IsaacLab headless via `env_isaacsim`
- Planner backend: `mpc`
- Cycles: `12`
- Segment summaries: `32`
- Exceptions: `0`

## Key Metrics

| Group | Segments | Actual abs gap mean | Actual air ratio mean | Max actual gap | Actual-plan foot err | Front CV | Switch sum mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pure linear/lateral | `8` | `0.0046` | `0.0677` | `0.0657` | `0.0096` | `0.1095` | `0.00` |
| linear speed sweep | `16` | `0.0100` | `0.0781` | `0.0911` | `0.0148` | `0.1443` | `1.50` |
| pure yaw | `4` | `0.1954` | `0.7083` | `0.5607` | `0.1158` | `0.4057` | `1.00` |
| yaw speed sweep | `8` | `0.2639` | `0.7448` | `0.6449` | `0.1481` | `0.4341` | `1.88` |
| linear+yaw combos | `8` | `0.0554` | `0.4323` | `0.2084` | `0.0534` | `0.5086` | `4.62` |

Representative segment results:

| Segment | Actual abs gap | Actual air ratio | Actual-plan foot err | Front CV | FL front | FR front | RL front | RR front |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `forward` | `0.0036` | `0.0000` | `0.0122` | `0.1262` | `1.000` | `1.000` | `0.000` | `0.000` |
| `backward` | `0.0000` | `0.0000` | `0.0000` | `0.0530` | `1.000` | `1.000` | `0.000` | `0.000` |
| `lateral_left` | `0.0128` | `0.2917` | `0.0146` | `0.0210` | `1.000` | `1.000` | `0.000` | `0.000` |
| `lateral_right` | `0.0102` | `0.2500` | `0.0190` | `0.0000` | `1.000` | `1.000` | `0.000` | `0.000` |
| `yaw_left` | `0.0807` | `0.6667` | `0.0640` | `0.3621` | `1.000` | `1.000` | `0.576` | `0.000` |
| `yaw_right` | `0.0092` | `0.1667` | `0.0130` | `0.1366` | `1.000` | `1.000` | `0.000` | `0.000` |
| `yaw_left_fast` | `0.3077` | `1.0000` | `0.1718` | `0.5839` | `1.000` | `1.000` | `1.000` | `0.000` |
| `yaw_right_fast` | `0.6007` | `1.0000` | `0.3178` | `0.2547` | `1.000` | `1.000` | `0.000` | `0.011` |
| `forward_yaw_left` | `0.0753` | `0.6667` | `0.0677` | `0.5557` | `1.000` | `0.955` | `0.235` | `0.000` |
| `lateral_right_yaw_right` | `0.0842` | `0.6250` | `0.0872` | `0.6716` | `1.000` | `1.000` | `1.000` | `0.000` |

## Result

Pass as reproduction across directions.

The failure is not evenly distributed:

- Pure front/back is mostly clean on actual stance-ground metrics.
- Pure lateral has mild actual airborne samples, especially at faster lateral speed, but much lower than yaw.
- Pure yaw and yaw-speed sweeps are the dominant failure mode, with actual stance air ratios around `0.71-0.74` and max gaps up to `0.6449m`.
- Linear+yaw combinations inherit the yaw problem: air ratio mean `0.4323`, high front-extension inconsistency, and asymmetric front occupancy.

## Conclusion

The current planner/playback stack still handles pure linear motion much better than yaw. The next planner change should be yaw-specific but must be guarded by forward/back/lateral acceptance metrics, because fast lateral already shows a smaller stance-ground degradation.

## Follow-Up

- Keep all-speed matrix as the regression gate before planner changes.
- Fix direction should target yaw swing target generation/contact phase semantics and actual playback/IK stance realization.
- Preserve forward/back/lateral behavior while reducing yaw and mixed-yaw airborne ratios.

## Git Refs

- Baseline Ref: working tree on top of `e90e3a4`
- Candidate Ref: working tree with updated [../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py](../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py)
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py](../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py](../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py)
