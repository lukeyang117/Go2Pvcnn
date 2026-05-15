# 2026-05-13 18:35 MPC Forward/Backward Alternation Probe

## Purpose

Test front/back commands specifically for the same root-relative front/rear occupancy metrics used in the yaw gait probe, without lateral commands.

## Stage

T300d production `extension/batch_mpc_planner` viewer-style IsaacLab runtime diagnostics.

## Related Todo

- [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)

## Procedure

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
MPC_TEST_DEVICE=cuda:2 \
MPC_YAW_GAIT_CYCLES=16 \
MPC_YAW_GAIT_SEQUENCES='forward:forward;backward:backward;forward_speeds:forward_slow,forward,forward_fast;backward_speeds:backward_slow,backward,backward_fast;forward_backward_switch:forward,backward,forward;backward_forward_switch:backward,forward,backward' \
MPC_YAW_GAIT_OUTPUT=/tmp/mpc_forward_backward_alternation_16.jsonl \
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

| Group | Segments | Actual abs gap mean | Actual air ratio mean | Front CV mean | Switch sum mean | Stuck score mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| steady forward segments | `7` | `0.0080` | `0.0000` | `0.2632` | `5.43` | `0.9950` |
| steady backward segments | `7` | `0.0028` | `0.0402` | `0.1217` | `0.00` | `1.0000` |
| forward/backward switch segments | `6` | `0.0019` | `0.0000` | `0.1711` | `0.00` | `1.0000` |

Representative segment results:

| Segment | Actual abs gap | Actual air ratio | Actual-plan foot err | Front CV | Switch sum | FL front | FR front | RL rear | RR rear |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `forward` | `0.0045` | `0.0000` | `0.0147` | `0.2028` | `0` | `1.000` | `1.000` | `1.000` | `1.000` |
| `forward_fast` | `0.0285` | `0.0000` | `0.0769` | `0.5399` | `30` | `1.000` | `0.898` | `1.000` | `1.000` |
| `backward` | `0.0000` | `0.0000` | `0.0000` | `0.0727` | `0` | `1.000` | `1.000` | `1.000` | `1.000` |
| `backward_fast` | `0.0195` | `0.2812` | `0.0116` | `0.1995` | `0` | `1.000` | `1.000` | `1.000` | `1.000` |
| `forward -> backward -> forward` | `0.0019` avg | `0.0000` avg | `0.0228` avg | `0.2399` avg | `0` | fixed | fixed | fixed | fixed |

## Result

Pass as front/back-only measurement.

Important interpretation: root-relative front/rear sign switching is not a good failure criterion for pure forward/backward commands, because front legs are anatomically expected to remain in front of the root and rear legs behind the root. The `stuck_score≈1.0` in forward/backward is therefore not the same type of failure as yaw.

Actual stance-ground metrics are much cleaner than yaw:

- forward and forward/backward switching have `actual air ratio=0`
- backward is mostly clean, with `backward_fast` showing a small degradation (`actual air ratio=0.2812`)
- forward_fast has larger actual-plan foot error and high front extent CV, but not airborne

## Conclusion

The yaw diagnosis still stands: yaw lacks a valid alternating front/back gait around the root. For forward/backward, the existing root-front/root-rear sign metric mostly measures anatomical leg placement, not gait alternation. A better forward/back metric should compare each foot around its own nominal hip/stance anchor or touchdown fore/aft excursion, rather than expecting front legs to cross behind root or rear legs to cross ahead of root.

## Follow-Up

- Do not use root-relative sign switching alone as the forward/backward failure criterion.
- If forward/backward alternation matters, add a per-leg nominal-anchor excursion metric: swing foot moves from rear-of-own-anchor to front-of-own-anchor for forward, and inverse for backward.
- Preserve forward/backward stance-ground behavior while fixing yaw.

## Git Refs

- Baseline Ref: working tree on top of `e90e3a4`
- Candidate Ref: working tree with [../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py](../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py)
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py](../../Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py](../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py)
