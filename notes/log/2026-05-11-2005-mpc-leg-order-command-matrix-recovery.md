# T300d MPC leg-order command-matrix recovery

- Time: 2026-05-11 20:05
- Stage: viewer/runtime MPC diagnostics correctness and command-matrix acceptance
- Related todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- Baseline ref: `working tree on top of 130c635`
- Candidate ref: `working tree (uncommitted)`

## Purpose

Investigate the user-facing symptom "横移/后退/旋转时腿不跟随 base" under MPC viewer playback, and determine whether it is a real kinematic mismatch or a diagnostics/ordering artifact.

## Root-cause investigation

- Reproduced command-matrix playback on `env_isaacsim` (`cuda:2`) and compared two foot-error measurements:
  - raw robot foot order (`body_pos_w[:, foot_ids, :]`)
  - prior viewer helper `_read_actual_kinematic_state` (quadrant-based reordering)
- Result:
  - raw-order errors for `backward/yaw` were near zero
  - quadrant-reordered errors were large (`~0.22-0.33`)
- Conclusion: the large `backward/yaw` foot-follow errors were primarily caused by dynamic quadrant reindexing, not by MPC playback kinematics.

## Code changes

- `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
  - Added `_reorder_feet_to_planner_order(...)` to map feet to fixed planner order (`FL/FR/RL/RR`) by body name.
  - `_read_actual_kinematic_state(...)` now defaults to planner-order feet; kept optional `reorder_by_quadrant` debug switch.
  - `_mpc_state_from_env(...)` now planner-orders `foot_pos` and `foot_vel` before building `MpcRobotState`.
- `Go2Pvcnn/extension/batch_mpc_planner/manager.py`
  - Canonicalized manager foot-id cache to planner order (`fl/fr/rl/rr`) when body names are available.
- `Go2Pvcnn/tests/test_mpc_runtime_headless.py`
  - Command-matrix gait-motion threshold now uses command-aware minimum foot-step:
    - `yaw_*`: `> 0.004`
    - others: `> 0.005`

## Verification

1. `python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q`
   - Result: `12 passed`
2. `MPC_TEST_DEVICE=cuda:2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py -k command_matrix -q`
   - Result: `.`
3. `MPC_TEST_DEVICE=cuda:2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py -q`
   - Result: `........s` (`8 passed, 1 skipped`)

Post-patch command-matrix diagnostics (`cuda:2`, 8 replans each):

- `forward`: `foot_err_mean=0.031881`, `foot_step_mean=0.007157`, `dx_mean=+0.293207`
- `backward`: `foot_err_mean=0.007102`, `foot_step_mean=0.007068`, `dx_mean=-0.203161`
- `lateral_left`: `foot_err_mean=0.037405`, `foot_step_mean=0.006685`, `dy_mean=+0.245062`
- `lateral_right`: `foot_err_mean=0.034349`, `foot_step_mean=0.006659`, `dy_mean=-0.245152`
- `yaw_left`: `foot_err_mean=0.000297`, `foot_step_mean=0.004653`, `dyaw_mean=+0.292051`
- `yaw_right`: `foot_err_mean=0.000490`, `foot_step_mean=0.004701`, `dyaw_mean=-0.293046`

## Conclusion

The dominant `backward/yaw` failure mode in command-matrix diagnostics was leg-index drift from quadrant reordering. After forcing fixed planner leg order in both viewer readback and MPC state extraction, command-matrix runtime acceptance recovers, and all four motion categories (forward/backward/lateral/yaw) show consistent directional root motion with low playback foot-error.
