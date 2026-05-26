# T302i IK Clamp Foot Mismatch Trace

## Purpose

Localize the reproduced low-small viewer planned-foot vs realized-foot mismatch to either viewer/Isaac readback, joint writeback, or planner IK/FK feasibility.

## Stage

- `extension/batch_mpc_planner`
- semantic MPC viewer/runtime diagnostics
- related todo: [../todo/T302i-viewer-realized-foot-mismatch.md](../todo/T302i-viewer-realized-foot-mismatch.md)

## Command / Procedure

```bash
python -m py_compile Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py
pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py -k "rolling_segment_terminal_trace_rows"
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 24 --warmup-steps 6 --commands 'forward_v050:0.50 0.00 0.00' --variants baseline --trace-foot-mismatch > tmp/t302i-viewer-realized-foot-mismatch/rolling25_low_small_trace_forward_clamp.jsonl 2>&1
```

## Input Conditions

- Python for real run: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`
- Device: `cuda:0`
- Case: `small`
- Command: `forward_v050: 0.50 0.00 0.00`
- Total rollout: `300`
- Rolling segment horizon: `25`
- Trace output: `tmp/t302i-viewer-realized-foot-mismatch/rolling25_low_small_trace_forward_clamp.jsonl`

## Key Metrics

- Local trace helper check: `1 passed, 57 deselected`.
- `py_compile`: pass.
- Worst terminal trace row:
  - `segment=0`, `frame=24`, `worst_leg=0/FL`.
  - `planned_foot_xyz=[15.415548, 1.815011, 0.0]`.
  - `planned_touchdown_xyz=[15.415548, 1.815011, 0.0]`.
  - `actual_foot_xyz=[15.393563, 1.648044, 0.231984]`.
  - `foot_error_norm=0.286667m`.
  - `internal_fk_error_norm=0.286668m`.
  - `actual_vs_internal_fk_error_norm=4.34e-7m`.
  - `joint_error_max_abs=0.0`.
  - `ik_clamp_worst_joint_name=FL_calf`.
  - `ik_raw_worst_joint_value=-0.0`.
  - `ik_clamped_worst_joint_value=-0.837800`.
  - `ik_worst_joint_limit_upper=-0.8378`.
  - `ik_clamp_delta_max_abs=0.837800`.
- Summary row:
  - `rolling_segment_terminal_foot_error_max=0.286667m`.
  - `playback_root_error_max=2.98e-08m`.
  - `touchdown_on_semantic_rate=0`.
  - `stance_on_semantic_rate=0`.
  - `foot_semantic_penetration_rate=0`.

## Result

Pass as root-cause localization.

The simulated robot's realized foot agrees with internal FK from the exported joint sequence to numerical precision, and the joint sequence agrees with actual joint readback. The mismatch is therefore not caused by Isaac readback, joint order, root playback, or marker extraction.

## Conclusion

The current failure is caused by exporting unreachable Cartesian `foot_pos` / touchdown targets after clamped IK. In the worst low-small crossing frame, raw IK would require `FL_calf=-0.0`, outside the Go2 calf upper limit `-0.8378`; clamped IK produces a feasible joint at the limit, and FK places the foot about `0.2867m` away from the planned touchdown marker.

## Follow-Up

- Discuss production contract before changing behavior:
  - export FK-realized `foot_pos` / touchdown after clamped IK;
  - strengthen IK/FK residual / joint-limit feasibility as a hard gate before export;
  - constrain low-small foot-over target generation to stay in reachable workspace.
- Keep viewer `--planner-backend mpc --n-frames 25` entrypoint mismatch as a separate T302i issue.

## Git Refs

- Baseline Ref: `working tree @ c54dc5c`
- Candidate Ref: `working tree`
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/kinematics.py](../../Go2Pvcnn/extension/batch_mpc_planner/kinematics.py)
  - `../../tmp/t302i-viewer-realized-foot-mismatch/rolling25_low_small_trace_forward_clamp.jsonl`
