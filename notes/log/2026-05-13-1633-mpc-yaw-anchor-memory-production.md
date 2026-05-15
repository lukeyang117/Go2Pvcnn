# T300d MPC yaw anchor memory production implementation

- Time: 2026-05-13 16:33
- Stage: `extension/batch_mpc_planner` production runtime/viewer path
- Related todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- Baseline Ref: `57b5c64`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/types.py](../../Go2Pvcnn/extension/batch_mpc_planner/types.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py](../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)

## Purpose

Move the accepted post-sweep design into production planner code: do not copy local `yawfix4` masks directly; instead add a yaw-mode anchor-memory lifecycle.

## Implementation

- Added `MpcFootholdMemory.yaw_foot_rel_body_seed`.
- Added runtime config:
  - `foothold_yaw_body_anchor_blend`
  - `foothold_yaw_body_anchor_max_step_m`
  - `foothold_yaw_stable_contact_steps`
- Manager memory now keeps:
  - running linear/body footprint seed for linear-dominant commands
  - separate fixed yaw footprint seed
  - stable-contact counters
- On yaw-mode entry, yaw seed is rebased from current running footprint once.
- While yaw-dominant, ordinary contact/touchdown updates do not continuously drift the yaw seed.
- Stance anchors update only after stable contact instead of every contact/touchdown frame.
- Nominal builder blends stance anchor toward yaw body seed under yaw gate/ramp and caps per-replan XY displacement in the nominal target.
- Viewer direct path mirrors the same lightweight memory fields so visual inspection uses the same contract as manager-driven runtime.

## Verification

Red-green focused tests:

```bash
python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -k 'yaw_foothold_memory or nominal_yaw_body_anchor' -q
```

Result: first run failed because production memory lacked `yaw_foot_rel_body_seed`; after implementation and test correction, final result was `2 passed, 12 deselected`.

Backend suite:

```bash
python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Result: `14 passed in 4.40s`.

Compile:

```bash
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/types.py Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/manager.py Go2Pvcnn/extension/batch_mpc_planner/nominal.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_batch_mpc_backend.py
```

Result: exit `0`.

IsaacLab smoke:

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
MPC_LONG_DRIFT_VARIANTS=baseline \
MPC_LONG_DRIFT_SEQUENCES='yaw_left_only:yaw_left;forward_only:forward;lateral_left_yaw_right_lateral_left:lateral_left,yaw_right,lateral_left' \
MPC_PROBE_CYCLES=4 \
MPC_PROBE_TRANSITION_WINDOW=2 \
MPC_TEST_DEVICE=cuda:2 \
timeout 480s python Go2Pvcnn/tests/mpc_touchdown_sequence_probe.py
```

Artifact: `/tmp/mpc_yaw_memory_prod_smoke_4.jsonl`

Key metrics:

| Sequence | Segment | foot_err | stepmax | touchdown_gap | airborne |
| --- | --- | ---: | ---: | ---: | ---: |
| yaw_left_only | yaw_left | 0.0130 | 0.1093 | 0.0000 | 0.0000 |
| forward_only | forward | 0.0003 | 0.0869 | 0.0000 | 0.0000 |
| lateral_left_yaw_right_lateral_left | lateral_left | 0.0007 | 0.0883 | 0.0000 | 0.0000 |
| lateral_left_yaw_right_lateral_left | yaw_right | 0.0042 | 0.1273 | 0.0000 | 0.0000 |
| lateral_left_yaw_right_lateral_left | lateral_left | 0.0090 | 0.0929 | 0.0000 | 0.0000 |

## Conclusion

Production code now implements the yaw anchor-memory lifecycle and passes focused/backend/compile/short IsaacLab smoke verification. Long 48-cycle acceptance remains unverified after production integration and should be the next evidence step before judging visual quality.
