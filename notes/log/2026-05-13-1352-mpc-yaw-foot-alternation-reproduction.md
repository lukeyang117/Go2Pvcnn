# MPC Yaw Foot Alternation Reproduction

- Time: 2026-05-13 13:52 CST
- Purpose: reproduce the user-observed visual issue where forward/lateral look acceptable but yaw rotation still has poor foot alternation / flying feet.
- Stage: `extension/batch_mpc_planner` viewer-style IsaacLab runtime diagnostics.
- Related todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- Baseline Ref: `57b5c64`
- Candidate Ref: working tree with diagnostic probe aligned to viewer-side MPC foothold memory
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_touchdown_sequence_probe.py](../../Go2Pvcnn/tests/mpc_touchdown_sequence_probe.py)

## Procedure

The probe was adjusted so direct diagnostic planning carries the same viewer-side MPC foothold memory as `go2_foostep_planner.py --planner-backend mpc`.

Commands:

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
python -m py_compile Go2Pvcnn/tests/mpc_touchdown_sequence_probe.py

MPC_LONG_DRIFT_VARIANTS=baseline \
MPC_LONG_DRIFT_SEQUENCES='yaw_left_only:yaw_left;yaw_right_only:yaw_right' \
MPC_PROBE_CYCLES=30 \
MPC_PROBE_TRANSITION_WINDOW=6 \
MPC_TEST_DEVICE=cuda:2 \
timeout 420s python Go2Pvcnn/tests/mpc_touchdown_sequence_probe.py
cp /tmp/mpc_joint_metrics.jsonl /tmp/mpc_joint_metrics_yaw_30.jsonl

MPC_LONG_DRIFT_VARIANTS=baseline \
MPC_LONG_DRIFT_SEQUENCES='forward_only:forward;lateral_left_only:lateral_left' \
MPC_PROBE_CYCLES=30 \
MPC_PROBE_TRANSITION_WINDOW=6 \
MPC_TEST_DEVICE=cuda:2 \
timeout 420s python Go2Pvcnn/tests/mpc_touchdown_sequence_probe.py
cp /tmp/mpc_joint_metrics.jsonl /tmp/mpc_joint_metrics_forward_lateral_30.saved.jsonl
```

IsaacLab emitted known PhysX GPU kernel warnings, but both probe runs completed and wrote JSONL metrics.

## Key Metrics

| Segment | foot_err_mean | root_rel_foot_err_mean | foot_step_mean | foot_step_max | stance_anchor_error | touchdown_jump_distance | touchdown_jump_max | grounding gaps | dyaw_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yaw_left` | `0.1200` | `0.1200` | `0.0266` | `0.6112` | `0.1147` | `0.1441` | `0.3377` | `0.0` | `+0.0836` |
| `yaw_right` | `0.1087` | `0.1087` | `0.0263` | `0.5737` | `0.1045` | `0.1308` | `0.2873` | `0.0` | `-0.0831` |
| `forward` | `0.0374` | `0.0374` | `0.0123` | `0.0931` | not comparable in world-anchor metric under translation | not comparable | not comparable | `0.0` | `0.0` |
| `lateral_left` | `0.0449` | `0.0449` | `0.0121` | `0.2227` | not comparable in world-anchor metric under translation | not comparable | not comparable | `0.0` | `0.0` |

All four commands kept:

- `touchdown_ground_gap_mean=0.0`
- `touchdown_event_ground_gap_mean=0.0`
- `stance_ground_gap_mean=0.0`

The gait-pattern counters were also stable:

- `diagonal_swing_pair_ratio=1.0`
- `lateral_swing_pair_ratio=0.0`
- `front_hind_swing_pair_ratio=0.0`
- `triple_or_more_swing_ratio=0.0`
- `contact_flip_count=420` for 30 cycles

## Interpretation

This reproduces the user-visible yaw problem as a foot-trajectory / playback mismatch rather than a terrain-grounding or gross contact-schedule failure:

- terrain grounding is fixed in this probe; touchdown and stance gaps are zero
- contact pairing remains diagonal, so the schedule is not switching to lateral/front-hind pairs
- yaw still has much larger foot tracking error than forward/lateral
  - about `10-12 cm` for yaw vs `3.7-4.5 cm` for forward/lateral
- yaw has very large single-frame foot displacement spikes
  - `0.57-0.61 m` max for yaw vs `0.09 m` forward and `0.22 m` lateral
- yaw stance-anchor and touchdown-jump metrics are large in an origin-stationary command, so these are meaningful for the yaw case

The most likely bad signal is discontinuous yaw foothold replacement / IK handoff around touchdown, not airborne touchdown height.

## Artifacts

- `/tmp/mpc_joint_metrics_yaw_30.jsonl`
- `/tmp/mpc_joint_metrics_forward_lateral_30.saved.jsonl`

## Follow-Up

Next investigation should focus on yaw-specific smoothing:

- reduce per-frame foot displacement spikes
- smooth stance-anchor replacement over horizon time, not only yaw-entry replan time
- add or tune touchdown jump / foot acceleration penalties for yaw-dominant commands
- inspect IK limits around yaw touchdown frames
