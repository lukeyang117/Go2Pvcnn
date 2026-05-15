# 2026-05-13 17:57 MPC Yaw Gait Failure Probe

## Purpose

Reproduce the user-observed current MPC planner failure after production yaw anchor-memory changes: stance feet can still appear airborne in IsaacLab playback, and yaw rotation does not alternate feet front/back relative to root.

## Stage

T300d production `extension/batch_mpc_planner` viewer-style IsaacLab runtime diagnostics.

## Related Todo

- [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)

## Procedure

Added standalone probe [../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py](../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py).

The probe uses the viewer/IsaacLab startup fixture and computes planned stance gap, actual post-playback stance gap, per-leg swing front/rear occupancy in root-yaw frame, front/rear switches, side/diagonal imbalance, and front-extension consistency.

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
MPC_TEST_DEVICE=cuda:2 \
MPC_YAW_GAIT_CYCLES=16 \
MPC_YAW_GAIT_SEQUENCES='yaw_left_only:yaw_left;yaw_right_only:yaw_right;lateral_left_yaw_right_lateral_left:lateral_left,yaw_right,lateral_left' \
MPC_YAW_GAIT_OUTPUT=/tmp/mpc_yaw_gait_failure_actual_16b.jsonl \
timeout 900s python Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py
```

## Input Conditions

- Device: `cuda:2`
- Runtime: IsaacLab headless via `env_isaacsim`
- Planner backend: `mpc`
- Envs: fixture default `2`
- Cycles: `16`

## Key Metrics

- jsonl: `/tmp/mpc_yaw_gait_failure_actual_16b.jsonl`
- rows: `80` cycles, `80` actual-cycle rows, `5` segment summaries
- exceptions: `0`

| Sequence / Segment | Planned stance abs gap | Planned air ratio | Actual stance abs gap | Actual max gap | Actual air ratio | Actual-plan foot err | Front CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yaw_left_only / yaw_left` | `0.0000` | `0.0000` | `0.1161` | `0.2215` | `0.7812` | `0.0796` | `0.4215` |
| `yaw_right_only / yaw_right` | `0.0000` | `0.0000` | `0.0389` | `0.1584` | `0.4062` | `0.0368` | `0.2444` |
| `lateral_left_yaw_right_lateral_left / lateral_left` | `0.0000` | `0.0000` | `0.0216` | `0.1073` | `0.3438` | `0.0220` | `0.0489` |
| `lateral_left_yaw_right_lateral_left / yaw_right` | `0.0000` | `0.0000` | `0.1099` | `0.1929` | `0.7812` | `0.0759` | `0.4425` |
| `lateral_left_yaw_right_lateral_left / lateral_left` | `0.0000` | `0.0000` | `0.2708` | `0.3656` | `1.0000` | `0.1433` | `0.6238` |

Yaw-only alternation evidence:

| Segment | Leg | Front ratio | Rear ratio | rel_x mean | Switches |
| --- | --- | ---: | ---: | ---: | ---: |
| `yaw_left` | `FL` | `1.0000` | `0.0000` | `0.2022` | `0` |
| `yaw_left` | `FR` | `1.0000` | `0.0000` | `0.2585` | `0` |
| `yaw_left` | `RL` | `0.6818` | `0.3125` | `-0.0029` | `0` |
| `yaw_left` | `RR` | `0.0000` | `1.0000` | `-0.2029` | `0` |
| `yaw_right` | `FL` | `1.0000` | `0.0000` | `0.1732` | `0` |
| `yaw_right` | `FR` | `1.0000` | `0.0000` | `0.2572` | `0` |
| `yaw_right` | `RL` | `0.0284` | `0.9119` | `-0.1910` | `4` |
| `yaw_right` | `RR` | `0.0000` | `1.0000` | `-0.2478` | `0` |

## Result

Pass as reproduction.

The probe separates two signals:

- planned MPC output reports stance feet grounded against the height map
- after direct IsaacLab playback, actual stance feet can be above terrain, especially in yaw and after yaw-to-lateral continuation

The yaw front/back gait failure is reproduced: `FL` and `FR` remain in front of root for all swing samples in pure yaw, while `RR` remains behind root.

## Conclusion

The remaining visual problem is not only terrain z in the planner result. There is an actual playback/IK realization gap plus an asymmetric yaw swing layout: the current yaw anchor-memory keeps a body-foot footprint, but it does not create a yaw-specific alternating front/back foot cycle relative to the root.

## Follow-Up

- Treat yaw alternation as an open T300d child issue.
- Next implementation direction should inspect yaw-mode swing target generation/contact phase semantics, not only output z grounding.
- Keep the new probe as the reproduction script before further planner changes.

## Git Refs

- Baseline Ref: working tree on top of `e90e3a4`
- Candidate Ref: working tree with [../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py](../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py)
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py](../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py](../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py)
