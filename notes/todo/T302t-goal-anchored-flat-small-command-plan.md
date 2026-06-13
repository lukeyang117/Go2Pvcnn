# T302t Goal-Anchored Flat-Small Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 flat-small avoidance 训练新增一个只在该 cfg 中启用的 goal-anchored body velocity command，让旧 checkpoint 继续看到 body-frame `base_velocity`，但世界运动方向由 reset 时采样的远目标锚定。

**Architecture:** 新增 `GoalAnchoredVelocityCommand` / `GoalAnchoredVelocityCommandCfg` 到现有 `go2_pvcnn.mdp.commands` 模块。flat-small cfg 把 `commands.base_velocity` 替换成新 cfg，名字仍是 `base_velocity`；reward、observation、MPC manager 不改，继续读取 body-frame `[vx_body, vy_body, yaw_rate]`。

**Tech Stack:** IsaacLab `CommandTerm`/`CommandTermCfg`、PyTorch tensor command buffers、pytest fake env 单元测试、`env_isaacsim` smoke。

---

## Source Design

- [../../docs/superpowers/specs/2026-06-13-goal-anchored-flat-small-command-design.md](../../docs/superpowers/specs/2026-06-13-goal-anchored-flat-small-command-design.md)

## Scope And Constraints

- Keep checkpoint compatibility: observation/action shape unchanged.
- Keep command contract: `command_manager.get_command("base_velocity")` returns body-frame `[vx_body, vy_body, yaw_rate]`.
- Do not change `track_lin_vel_xy_exp`, `track_ang_vel_z_exp`, observations, or MPC manager.
- Only flat-small training cfg uses the new command by default.
- Baseline semantic cfg keeps `UniformLevelVelocityCommandCfg` and velocity curriculum support.

## Files

- Modify [../../Go2Pvcnn/go2_pvcnn/mdp/commands/velocity_command.py](../../Go2Pvcnn/go2_pvcnn/mdp/commands/velocity_command.py)
  - Add `GoalAnchoredVelocityCommand` implementation.
  - Add `GoalAnchoredVelocityCommandCfg`.
  - Keep `UniformLevelVelocityCommandCfg`.
- Modify [../../Go2Pvcnn/go2_pvcnn/mdp/commands/__init__.py](../../Go2Pvcnn/go2_pvcnn/mdp/commands/__init__.py)
  - Export the new cfg/class.
- Modify [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - Wire flat-small `commands.base_velocity` to `GoalAnchoredVelocityCommandCfg`.
- Add [../../Go2Pvcnn/tests/test_goal_anchored_velocity_command.py](../../Go2Pvcnn/tests/test_goal_anchored_velocity_command.py)
  - Unit tests for reset sampling, quadrant signs, yaw clamp, and goal extension.
- Modify [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - Static cfg test: flat-small uses goal-anchored cfg, baseline does not.
- Update notes/log after verification.

## Task 1: RED Command Unit Tests

- [ ] Add a fake env/robot test module `Go2Pvcnn/tests/test_goal_anchored_velocity_command.py`.
- [ ] Test reset/resample initializes:
  - `goal_xy` at `goal_distance`.
  - `vx_abs/vy_abs` in configured ranges.
  - command tensor shape `[num_envs, 3]`.
- [ ] Test per-step quadrant mapping:
  - with root yaw `0` and goal in world `(+x,+y)`, command x/y are positive.
  - after root yaw changes so the same goal is in another body quadrant, command signs change.
- [ ] Test yaw clamp:
  - heading error larger than range clamps to `yaw_range`.
- [ ] Test reached-goal extension:
  - when root is within `goal_reached_threshold`, new goal is extended by `goal_distance`.

Run:

```bash
pytest Go2Pvcnn/tests/test_goal_anchored_velocity_command.py -q
```

Expected before implementation: import or attribute failure for missing `GoalAnchoredVelocityCommandCfg`.

## Task 2: GREEN Command Implementation

- [ ] Implement `GoalAnchoredVelocityCommand` using IsaacLab `CommandTerm` pattern.
- [ ] Buffers:
  - `vel_command_b: [num_envs, 3]`
  - `goal_xy_w: [num_envs, 2]`
  - `vx_abs: [num_envs]`
  - `vy_abs: [num_envs]`
  - `is_standing_env: [num_envs]` for compatibility, default probability `0`.
- [ ] `_resample_command(env_ids)`:
  - sample target direction uniformly in `[-pi, pi]`;
  - set `goal_xy_w`;
  - sample `vx_abs/vy_abs`;
  - sample standing envs only if `rel_standing_envs > 0`.
- [ ] `_update_command()`:
  - compute `dir_world`;
  - extend reached goals;
  - compute body-frame signs from current root yaw;
  - set fixed-magnitude x/y signs;
  - set yaw from clamped heading error;
  - zero standing env commands.
- [ ] `_update_metrics()`:
  - keep `error_vel_xy` and `error_vel_yaw` like `UniformVelocityCommand`.
- [ ] Reuse or adapt debug visualization only if needed; default can rely on command property without adding new markers.

Run:

```bash
pytest Go2Pvcnn/tests/test_goal_anchored_velocity_command.py -q
```

Expected: all new command tests pass.

## Task 3: RED/GREEN Flat-Small Cfg Wiring

- [ ] Update `test_flat_small_avoidance_cfg_static_contract`:
  - assert baseline `base.commands.base_velocity` is `UniformLevelVelocityCommandCfg`.
  - assert flat-small `cfg.commands.base_velocity` is `GoalAnchoredVelocityCommandCfg`.
  - assert `goal_distance == 10.0`.
  - assert `vx_abs_range == (0.6, 1.0)`.
  - assert `vy_abs_range == (0.6, 1.0)`.
  - assert `yaw_stiffness == 0.5`.
  - assert `yaw_range == (-0.8, 0.8)`.
  - remove old flat-small `ranges.lin_vel_x/y/ang_vel_z` expectations.
- [ ] Wire flat-small cfg in `__post_init__`.
- [ ] Keep `self.curriculum.lin_vel_cmd_levels = None`.

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract -q
```

Expected: pass after wiring.

## Task 4: Focused Compatibility Tests

- [ ] Run command/cfg/viewer static tests:

```bash
pytest \
  Go2Pvcnn/tests/test_goal_anchored_velocity_command.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract \
  Go2Pvcnn/tests/test_viewer_reset.py::test_flat_small_play_cfg_disables_training_curriculum_without_semantic_contact_sensors \
  -q
```

- [ ] Run pycompile:

```bash
python -m py_compile \
  Go2Pvcnn/go2_pvcnn/mdp/commands/velocity_command.py \
  Go2Pvcnn/go2_pvcnn/mdp/commands/__init__.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
```

## Task 5: Real IsaacLab Smoke

- [ ] Run small flat-small train smoke:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --headless \
  --num_envs 8 \
  --max_iterations 1 \
  --device cuda:0
```

- [ ] Confirm startup reaches Command Manager and Reward Manager.
- [ ] If possible, run a tiny command probe to print:
  - command shape `[8, 3]`;
  - `abs(vx)` / `abs(vy)` within `[0.6, 1.0]` for non-standing envs;
  - `yaw` within `[-0.8, 0.8]`.

## Task 6: Notes And Logs

- [ ] Create a log under `notes/log/`.
- [ ] Update `notes/log/index.md`.
- [ ] Update `notes/todo.md`.
- [ ] Update this branch page with actual command results.
- [ ] Commit the implementation after verification.

## Current Status

- [ ] Task 1
- [ ] Task 2
- [ ] Task 3
- [ ] Task 4
- [ ] Task 5
- [ ] Task 6

## Related Logs

- Pending.

## Git Refs

- Last Feature Commit: `65acd24`
- Last Verified Commit: `65acd24`
- Current Work Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/go2_pvcnn/mdp/commands/velocity_command.py](../../Go2Pvcnn/go2_pvcnn/mdp/commands/velocity_command.py)
  - [../../Go2Pvcnn/go2_pvcnn/mdp/commands/__init__.py](../../Go2Pvcnn/go2_pvcnn/mdp/commands/__init__.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/test_goal_anchored_velocity_command.py](../../Go2Pvcnn/tests/test_goal_anchored_velocity_command.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)

## Next Step

- Start Task 1 with failing command unit tests.
