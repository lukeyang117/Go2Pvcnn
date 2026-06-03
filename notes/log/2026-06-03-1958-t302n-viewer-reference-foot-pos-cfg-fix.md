# T302n Viewer Reference Foot Pos Cfg Fix

## Purpose

Verify the viewer traceback reported by the user:

```text
AttributeError: type object 'TeacherElevationTrajectoryMpcSemanticRewardsCfg' has no attribute 'reference_foot_pos'
```

## Stage

MPC semantic viewer cfg / semantic obstacle curriculum worktree.

## Related Todo

- [../todo/T302n-semantic-obstacle-curriculum-plan.md](../todo/T302n-semantic-obstacle-curriculum-plan.md)
- [../todo/T302l-mpc-rl-participation-and-reward-plan.md](../todo/T302l-mpc-rl-participation-and-reward-plan.md)

## Command / Procedure

Cfg-only AppLauncher smoke:

```bash
CUDA_VISIBLE_DEVICES=3 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python - <<'PY'
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app
try:
    from pathlib import Path
    import sys
    repo = Path.cwd()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "Go2Pvcnn"))
    from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER
    cfg = TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER()
    print({
        "planner_owned_reference_cache": cfg.planner_owned_reference_cache,
        "use_batched_reference_trajectory": cfg.use_batched_reference_trajectory,
        "reference_foot_pos_enabled": cfg.rewards.reference_foot_pos is not None,
        "semantic_contact_collision_enabled": cfg.rewards.semantic_contact_collision is not None,
        "small_sensor_enabled": cfg.scene.semantic_contact_small is not None,
        "large_sensor_enabled": cfg.scene.semantic_contact_large is not None,
    }, flush=True)
finally:
    simulation_app.close()
PY
```

User viewer command, bounded by timeout:

```bash
CUDA_VISIBLE_DEVICES=3 timeout 120s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  --headless \
  --livestream 2 \
  --webrtc-public-ip 172.31.179.75 \
  --device cuda:0 \
  --num_envs 1 \
  --terrain task \
  --planner-backend mpc
```

## Input Conditions

- Environment: `env_isaacsim` at `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`.
- GPU selection: `CUDA_VISIBLE_DEVICES=3`.
- Candidate ref: `f23858e` plus local semantic obstacle curriculum worktree changes.

## Key Metrics

- `TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER()` constructed without `AttributeError`.
- `reference_foot_pos_enabled=True`.
- `semantic_contact_collision_enabled=True`.
- Full viewer command reached environment setup and printed:
  - `Completed setting up the environment...`
  - `Attached mpc trajectory manager`
  - active reward term `reference_foot_pos`
  - active reward term `semantic_contact_collision`

## Result

Pass for the reported traceback.

The root cause was the `VIEWER.__post_init__()` path reading `TeacherElevationTrajectoryMpcSemanticRewardsCfg.reference_foot_pos` as a class attribute after IsaacLab `@configclass` processing. The current worktree rebuilds the reward terms through `_reference_foot_pos_reward_term()` and `_semantic_contact_collision_reward_term()` instead.

## Follow-Up

The viewer process did not exit cleanly through `timeout`; the validation process was manually cleaned up. This did not affect traceback verification, because the viewer had already reached the main loop.

## Git Refs

- Baseline Ref: `f23858e`
- Candidate Ref: working tree

## Key Files

- [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
- [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
