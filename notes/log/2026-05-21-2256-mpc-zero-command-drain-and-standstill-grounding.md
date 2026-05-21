# 2026-05-21 22:56 MPC Zero Command Drain And Standstill Grounding

## Purpose

Fix the user-visible airborne touchdown cuboids seen when a moving MPC trajectory is interrupted by a zero command while some feet are still in swing.

## Stage

`extension/batch_mpc_planner` zero-command shortcut and `extension/viz` viewer replan loop.

## Related Todo

- [T302g MPC Semantic RL Training Config](../todo/T302g-mpc-semantic-rl-training-config.md)

## Command / Procedure

```bash
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/mpc_flat_touchdown_height_probe.py
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k "zero_command_standstill or outputs_grounded_touchdowns"
pytest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q -k "zero_command_drains_until_grounded_landing_frame"
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
CUDA_VISIBLE_DEVICES=1 python Go2Pvcnn/tests/mpc_flat_touchdown_height_probe.py --device cuda:0 --cycles 1 --playback-frame 49 --speeds 0.30 --zero-after-forward-frame 20
```

## Input Conditions

- Flat terrain.
- `env_isaacsim`.
- MPC backend, horizon `50`.
- Reproduction path: forward `vx=0.30`, playback to frame `20`, then zero-command replan.

## Key Metrics

- Before fix reproduction:
  - `viz_td_minus_mpc=[+0.06443, +0.12734, +0.11847, +0.00000]`
  - `td_equals_state_foot_max_abs=0.000000`
  - Cause: `zero_command_standstill` copied current airborne `state.foot_pos` into `planned_touchdown_w`.
- After fix reproduction:
  - `viz_td_minus_mpc=[+0.00000, +0.00000, +0.00000, +0.00000]`
  - `state_foot_minus_mpc=[+0.06056, +0.09577, +0.09677, -0.00000]`
  - `td_equals_state_foot_max_abs=0.096767`
  - Meaning: current feet can still be airborne at forced zero replan, but exported touchdown markers are now terrain-grounded.
- Tests:
  - targeted MPC backend: `2 passed, 94 deselected`
  - viewer drain focused: `1 passed, 33 deselected`
  - full backend: `96 passed, 1 warning`
  - `py_compile`: exit `0`

## Result

Pass.

## Conclusion

`_standstill_result_from_state(...)` now accepts terrain and grounds standstill `foot_pos`, `touchdown_seq`, and `planned_touchdown_w` with `height_at(terrain, foot_xy)`. This fixes direct zero-command standstill exports even when current IsaacLab feet are airborne.

The viewer replan loop now detects MPC command transitions from nonzero to zero and drains the current moving trajectory until a future frame where all four planned feet are at terrain height before allowing zero-command replan. During draining, it keeps `last_cmd` as the previous moving command so the zero command does not prematurely become the accepted command before landing.

## Follow-Up

- The forced zero-replan probe validates the standstill export guard. Interactive viewer behavior should now continue the existing trajectory until a grounded frame before replanning to standstill.
- Separate command-direction negative-`dx` signals from earlier probes remain unrelated and not addressed here.

## Git Refs

- Baseline Ref: working tree after reproducing airborne zero-command touchdown cuboids.
- Candidate Ref: working tree with zero-command drain and standstill grounding.
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - [../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py)
  - [../../Go2Pvcnn/tests/mpc_flat_touchdown_height_probe.py](../../Go2Pvcnn/tests/mpc_flat_touchdown_height_probe.py)
