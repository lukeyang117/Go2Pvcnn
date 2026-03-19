# Teacher Elevation — 高程图 + CNN + PPO

**Date**: 2026-03-19  
**Status**: Approved

## Summary

Add `teacher_elevation` experiment: elevation map (height map) + CNN + PPO, inheriting from teacher_without_semantic. Only adds lidar for height map; no cost map, no semantic objects (YCB/furniture).

## Decisions

| Decision | Choice |
|----------|--------|
| Base | Inherit TeacherWithoutSemanticEnvCfg |
| Scene | Inherit RobotSceneCfg, add lidar (terrain only) |
| Lidar | SemanticLidarCfg with mesh_prim_paths=["/World/ground"], return_height_map=True |
| Model | ActorCriticCNN, cost_map_channels=1, cost_map_size=15 |
| MDP | All inherited from teacher_without_semantic |

## Implementation

### 1. teacher_elevation_env_cfg.py

- `TeacherElevationSceneCfg` inherits `RobotSceneCfg` (from teacher_without_semantic), adds lidar:
  - SemanticLidarCfg, mesh_prim_paths=["/World/ground"]
  - return_height_map=True, height_map_size=(1.5, 1.5), height_map_resolution=0.1
  - Same pattern/offset as teacher_semantic
- `TeacherElevationEnvCfg` inherits `TeacherWithoutSemanticEnvCfg`:
  - Override scene = TeacherElevationSceneCfg
  - Override observations = ObservationsCfg (policy_elevation_map + policy_state, critic_elevation_map + critic_state)
- `TeacherElevationEnvCfg_PLAY` inherits `TeacherWithoutSemanticEnvCfg_PLAY`:
  - Override observations for play (disable corruption)

### 2. go2_pvcnn/mdp/observations.py

Add `elevation_map(env, lidar_cfg)`:
- Read env.scene.sensors[lidar_cfg.name].data.height_map
- Shape (batch, H, W) -> unsqueeze(1) -> (batch, 1, H, W)
- Clip (-1.0, 5.0) to match teacher_semantic

### 3. agent/train_cfg.py

Add `_teacher_elevation_train_cfg()`:
- ActorCriticCNN, cost_map_channels=1, cost_map_size=15
- actor_cnn_cfg / critic_cnn_cfg same as teacher_semantic
- obs_groups: ["policy_elevation_map", "policy_state"], ["critic_elevation_map", "critic_state"]

### 4. train.py, play.py, register_envs.py

- Add teacher_elevation to experiment choices
- EXPERIMENT_ENV_MAP / EXPERIMENT_PLAY_MAP: teacher_elevation -> (TeacherElevationEnvCfg, Isaac-Teacher-Elevation-Go2-v0)
- Gym register: Isaac-Teacher-Elevation-Go2-v0, Isaac-Teacher-Elevation-Go2-Play-v0

## Files

| File | Change |
|------|--------|
| teacher_elevation_env_cfg.py | New (rename from teacher_elevatiom) |
| go2_pvcnn/mdp/observations.py | Add elevation_map |
| agent/train_cfg.py | Add _teacher_elevation_train_cfg |
| scripts/train.py | Add teacher_elevation |
| scripts/play.py | Add teacher_elevation |
| register_envs.py | Register envs |

## Verification

- train.py --experiment teacher_elevation --num_envs 4 --headless
- play.py --experiment teacher_elevation --run_dir <dir>
