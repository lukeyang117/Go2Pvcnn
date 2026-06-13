# T302t Goal-Anchored Flat-Small Command

## Purpose

Implement a flat-small-only body-frame velocity command that keeps old checkpoint command semantics while anchoring movement direction to a reset-time world target.

## Stage

Training command generation / flat-small avoidance.

## Related Todo

- [../todo/T302t-goal-anchored-flat-small-command-plan.md](../todo/T302t-goal-anchored-flat-small-command-plan.md)

## Baseline Ref

- `dd2d785`

## Candidate Ref

- Working tree after implementation.

## Key Files

- [../../Go2Pvcnn/go2_pvcnn/mdp/commands/velocity_command.py](../../Go2Pvcnn/go2_pvcnn/mdp/commands/velocity_command.py)
- [../../Go2Pvcnn/go2_pvcnn/mdp/commands/__init__.py](../../Go2Pvcnn/go2_pvcnn/mdp/commands/__init__.py)
- [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
- [../../Go2Pvcnn/tests/test_goal_anchored_velocity_command.py](../../Go2Pvcnn/tests/test_goal_anchored_velocity_command.py)
- [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
- [../../Go2Pvcnn/tests/test_viewer_reset.py](../../Go2Pvcnn/tests/test_viewer_reset.py)

## Change

Added `GoalAnchoredVelocityCommand` / `GoalAnchoredVelocityCommandCfg`.

Flat-small training now keeps command name `base_velocity` but uses the new command term:

```text
GoalAnchoredVelocityCommandCfg(
  goal_distance=10.0,
  goal_reached_threshold=1.0,
  vx_abs_range=(0.6, 1.0),
  vy_abs_range=(0.6, 1.0),
  yaw_stiffness=0.5,
  yaw_range=(-0.8, 0.8),
  rel_standing_envs=0.0,
)
```

The command tensor remains body-frame `[vx_body, vy_body, yaw_rate]`. Reset samples a world goal and fixed x/y magnitudes. Per step, x/y signs come from the target direction quadrant in current root frame, while yaw is clamped heading-error feedback.

## Verification

RED command tests:

```bash
pytest Go2Pvcnn/tests/test_goal_anchored_velocity_command.py -q
```

Observed before implementation: `4 failed`, missing `GoalAnchoredVelocityCommand`.

GREEN command tests:

```bash
pytest Go2Pvcnn/tests/test_goal_anchored_velocity_command.py -q
```

Observed: `4 passed`.

RED cfg wiring:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract -q
```

Observed before wiring: failed because flat-small still used `UniformLevelVelocityCommandCfg`.

Focused compatibility:

```bash
pytest \
  Go2Pvcnn/tests/test_goal_anchored_velocity_command.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract \
  Go2Pvcnn/tests/test_viewer_reset.py::test_flat_small_play_cfg_disables_training_curriculum_without_semantic_contact_sensors \
  -q
```

Observed: `6 passed`.

Viewer regression:

```bash
pytest Go2Pvcnn/tests/test_viewer_reset.py -q
```

Observed: `33 passed`.

Compile:

```bash
python -m py_compile \
  Go2Pvcnn/go2_pvcnn/mdp/commands/velocity_command.py \
  Go2Pvcnn/go2_pvcnn/mdp/commands/__init__.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
```

Observed: exit `0`.

Diff check:

```bash
git diff --check
```

Observed: exit `0`.

Real IsaacLab smoke:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --headless \
  --num_envs 8 \
  --max_iterations 1 \
  --device cuda:0
```

Observed exit `0`; run directory:

```text
logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-13_23-19-58
```

Key runtime evidence:

```text
Command Manager: base_velocity | GoalAnchoredVelocityCommand
policy_state shape: (45,)
critic_state shape: (48,)
action shape: 12
Reward Manager contains track_lin_vel_xy / track_ang_vel_z and semantic rewards
Curriculum Manager contains terrain_levels
```

Saved cfg evidence:

```text
env_cfg.yaml:
  commands.base_velocity.class_type = go2_pvcnn.mdp.commands.velocity_command:GoalAnchoredVelocityCommand
  goal_distance = 10.0
  vx_abs_range = (0.6, 1.0)
  vy_abs_range = (0.6, 1.0)
  yaw_stiffness = 0.5
  yaw_range = (-0.8, 0.8)
```

## Conclusion

Flat-small training now uses goal-anchored command generation without changing the policy/reward/MPC command contract. Old checkpoints still see body-frame `base_velocity` with shape `[N,3]`, but the command source should encourage longer world displacement instead of local turning.

## Follow-Up

- Start a resumed flat-small run from the intended old checkpoint.
- Watch `Train/mean_episode_length`, `Curriculum/terrain_levels`, `Episode_Reward/track_lin_vel_xy`, `Episode_Reward/track_ang_vel_z`, and semantic contact metrics.
- Re-run controlled crossing after a short warm-start to see whether path opportunities and overpass behavior improve.
