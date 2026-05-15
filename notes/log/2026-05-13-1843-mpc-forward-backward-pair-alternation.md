# 2026-05-13 18:43 MPC Forward/Backward Pair Alternation

## Purpose

Measure the user-requested front-leg and rear-leg left/right alternation for forward/backward commands:

- front pair: whether `FL` or `FR` is more forward relative to root during swing-active frames
- rear pair: whether `RL` or `RR` is more forward relative to root during swing-active frames

This is different from the earlier root-crossing metric.

## Stage

T300d production `extension/batch_mpc_planner` viewer-style IsaacLab runtime diagnostics.

## Related Todo

- [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)

## Procedure

Added pair-wise alternation metrics to [../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py](../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py):

- `front_pair_left_ahead_ratio`: `FL.x > FR.x`
- `front_pair_right_ahead_ratio`: `FR.x > FL.x`
- `front_pair_lead_switches`: lead sign switches between `FL` and `FR`
- `rear_pair_left_ahead_ratio`: `RL.x > RR.x`
- `rear_pair_right_ahead_ratio`: `RR.x > RL.x`
- `rear_pair_lead_switches`: lead sign switches between `RL` and `RR`

Command:

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
MPC_TEST_DEVICE=cuda:2 \
MPC_YAW_GAIT_CYCLES=16 \
MPC_YAW_GAIT_SEQUENCES='forward:forward;backward:backward;forward_speeds:forward_slow,forward,forward_fast;backward_speeds:backward_slow,backward,backward_fast;forward_backward_switch:forward,backward,forward;backward_forward_switch:backward,forward,backward' \
MPC_YAW_GAIT_OUTPUT=/tmp/mpc_forward_backward_pair_alternation_16.jsonl \
timeout 1200s python Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py
```

## Input Conditions

- Device: `cuda:2`
- Runtime: IsaacLab headless via `env_isaacsim`
- Planner backend: `mpc`
- Cycles per segment: `16`
- Segment summaries: `14`
- Exceptions: `0`

## Key Metrics

| Group | Front left ahead | Front right ahead | Front switches | Rear left ahead | Rear right ahead | Rear switches | Actual air |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| forward segments | `0.7948` | `0.1262` | `1.02` | `0.2699` | `0.5777` | `2.20` | `0.0000` |
| backward segments | `0.4397` | `0.4117` | `1.90` | `0.4649` | `0.3756` | `2.48` | `0.0402` |
| forward/back switch segments | `0.7131` | `0.1951` | `1.28` | `0.3598` | `0.4912` | `2.55` | `0.0000` |

Representative segment results:

| Segment | Front L ahead | Front R ahead | Front switches | Rear L ahead | Rear R ahead | Rear switches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `forward` | `0.7386` | `0.1577` | `1.56` | `0.1776` | `0.6349` | `2.19` |
| `forward_slow` | `0.6832` | `0.1364` | `1.44` | `0.1349` | `0.6818` | `1.38` |
| `forward_fast` | `1.0000` | `0.0000` | `0.00` | `0.3523` | `0.5511` | `4.00` |
| `backward` | `0.2940` | `0.4901` | `3.44` | `0.5043` | `0.3068` | `3.44` |
| `backward_fast` | `0.0199` | `0.9233` | `0.50` | `0.9261` | `0.0213` | `0.50` |

## Result

Pass as pair-wise alternation measurement.

Forward shows a clear left/right leading bias:

- Front pair: `FL` leads much more often than `FR` (`0.7948` vs `0.1262`).
- Rear pair: `RR` leads more often than `RL` (`0.5777` vs `0.2699`).
- `forward_fast` is worst for the front pair: `FL` leads `1.0000`, `FR` leads `0.0000`, and front lead switches are `0`.

Backward is more balanced at normal speed, but fast backward has a strong opposite diagonal bias:

- `backward_fast`: `FR` leads the front pair (`0.9233`) and `RL` leads the rear pair (`0.9261`), with low switches.

## Conclusion

The user-requested forward/backward pair alternation is not fully healthy. It is a different problem from yaw:

- yaw failure: feet do not alternate around root and actual stance grounding fails badly.
- forward/backward pair failure: stance grounding is mostly clean, but left/right pair leading is biased, especially `forward_fast` and `backward_fast`.

This suggests nominal swing target generation has a lateral/diagonal phase bias even in linear motion, but it is not currently causing the same airborne severity as yaw.

## Follow-Up

- Add forward/backward acceptance to future planner fixes using pair-wise lead balance and switch counts.
- For forward, target front-pair and rear-pair lead ratios closer to balanced over repeated cycles.
- Keep actual stance-ground as a separate guard; forward currently passes grounding but fails pair balance at higher speeds.

## Git Refs

- Baseline Ref: working tree on top of `e90e3a4`
- Candidate Ref: working tree with updated [../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py](../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py)
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py](../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py](../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py)
