# AI Environment And Observations

## Navigation

- doc role: AI stage note
- paired human doc: [../human/human-03-environment-and-observations.md](../human/human-03-environment-and-observations.md)
- previous: [ai-02-training-and-entrypoints.md](ai-02-training-and-entrypoints.md)
- next: [ai-04-lidar-and-pvcnn.md](ai-04-lidar-and-pvcnn.md)
- master index: [../index.md](../index.md)

## Purpose

Track where task configs, scene configs, observations, rewards, and curriculum are defined and how they feed later stages.

## Code Graph

```mermaid
graph LR
    register["register_envs.py\n../../Go2Pvcnn/go2_pvcnn/tasks/register_envs.py"]
    semantic["teacher_semantic\n../../Go2Pvcnn/go2_pvcnn/tasks/teacher_semantic_env_cfg.py"]
    nosemantic["teacher_without_semantic\n../../Go2Pvcnn/go2_pvcnn/tasks/teacher_without_semantic_env_cfg.py"]
    elevation["teacher_elevation\n../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_env_cfg.py"]
    traj["teacher_elevation_trajectory\n../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py"]
    obs["observations.py\n../../Go2Pvcnn/go2_pvcnn/mdp/observations.py"]
    rewards["reward/event/termination cfgs\n../../Go2Pvcnn/go2_pvcnn/tasks/"]
    curriculum["curriculums.py\n../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py"]

    register --> semantic
    register --> nosemantic
    nosemantic --> elevation
    elevation --> traj
    semantic --> obs
    nosemantic --> obs
    elevation --> obs
    traj --> obs
    semantic --> rewards
    nosemantic --> curriculum
```

## Candidate Files

- `Go2Pvcnn/go2_pvcnn/tasks/`
- `Go2Pvcnn/go2_pvcnn/mdp/curriculums.py`

## Inputs

- selected env cfg
- robot and terrain config
- sensor config

## Outputs

- policy observations
- critic observations
- curriculum state
