# Reference Contact Reward Strict MPC Current

## Purpose

Add MPC contact-state tracking to the RL reward with a small weight, strict current MPC frame alignment, and `reference_reward_mask` gating. This is intended to teach the policy to follow the MPC contact cycle instead of only matching foot positions.

## Stage

Trajectory reference reward / teacher elevation MPC semantic training cfg.

## Related Todo

- [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md)

## Implementation

- Updated [../../Go2Pvcnn/extension/mdp/rewards_reference.py](../../Go2Pvcnn/extension/mdp/rewards_reference.py):
  - `reference_contact_reward` still compares actual IsaacLab foot contact against `cache.contact_state` at `_select_reference_frame(env)`.
  - Actual contact remains `contact_forces.data.net_forces_w[:, foot_ids, 2] > 1.0`.
  - Added `manager.reference_reward_mask()` multiplication, matching `reference_foot_pos_reward`.
- Updated [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py):
  - Added `_reference_contact_reward_term()` with weight `0.05`.
  - Enabled `reference_contact` in the training reward cfg.
  - Re-enabled it for `VIEWER`, where MPC manager is attached.
  - Disabled it in no-MPC `PLAY` cfgs to avoid requiring a trajectory manager during visualization/play.
- Added tests:
  - `test_reference_contact_reward_uses_current_mpc_frame_and_reward_mask`
  - `test_teacher_mpc_semantic_cfg_enables_small_weight_reference_contact_reward`
  - `test_play_cfgs_disable_reference_contact_without_mpc_manager`

## TDD Evidence

RED:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_mpc_rl_participation.py::test_reference_contact_reward_uses_current_mpc_frame_and_reward_mask \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_teacher_mpc_semantic_cfg_enables_small_weight_reference_contact_reward -q
```

Failed for the intended reasons after import stubbing:

- `reference_contact_reward` returned `1.0` for a masked env instead of `0.0`.
- cfg source did not contain `reference_contact = _reference_contact_reward_term()`.

Additional RED:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_viewer_reset.py::test_play_cfgs_disable_reference_contact_without_mpc_manager -q
```

Failed because no-MPC PLAY cfgs did not disable `reference_contact`.

GREEN / regression:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_mpc_rl_participation.py \
  Go2Pvcnn/tests/test_viewer_reset.py::test_play_cfgs_disable_reference_contact_without_mpc_manager \
  Go2Pvcnn/tests/test_viewer_reset.py::test_play_cfgs_disable_timeout_refresh_for_visualization -q
```

Result: `7 passed`.

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_semantic_trajectory_cfg_defaults_to_mpc_and_semantic_scanner \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_teacher_mpc_semantic_cfg_enables_small_weight_reference_contact_reward -q
```

Result: `2 passed`.

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/extension/mdp/rewards_reference.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
```

Result: exit `0`.

```bash
git diff --check
```

Result: exit `0`.

## Real Smoke

Command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 240s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --headless \
  --device cuda:0 \
  --num_envs 4 \
  --mpc_num_envs 4 \
  --max_iterations 1 \
  --experiment teacher_elevation_trajectory_mpc_semantic \
  --planner-backend mpc
```

Output:

- [../../logs/mpc_policy_eval/reference_contact_train_smoke_final.log](../../logs/mpc_policy_eval/reference_contact_train_smoke_final.log)

Key evidence:

- Exit code `0`.
- Reward Manager contains `18` active terms.
- `reference_contact` appears with weight `0.05`.
- Timing output includes `reward.term.reference_contact`.
- Training completed one iteration.

## Result

Pass. Strict current-frame MPC contact reward is now active in training and viewer MPC paths, masked by valid MPC participation, and disabled in no-MPC play paths.

## Follow-Up

- Run a short resumed training from `model_14000.pt` and monitor whether contact mismatch drops without destabilizing foot tracking.
- If strict contact reward hurts old-policy warm-start too much, reduce weight below `0.05` before considering any phase offset.

## Git Refs

- Baseline Ref: current working tree on `costmap-teacher-ablation`
- Candidate Ref: local working tree
- Key Files:
  - [../../Go2Pvcnn/extension/mdp/rewards_reference.py](../../Go2Pvcnn/extension/mdp/rewards_reference.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/test_mpc_rl_participation.py](../../Go2Pvcnn/tests/test_mpc_rl_participation.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - [../../Go2Pvcnn/tests/test_viewer_reset.py](../../Go2Pvcnn/tests/test_viewer_reset.py)
