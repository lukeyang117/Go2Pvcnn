# Teacher Without Semantic — Full Alignment with velocity_env_cfg

**Date**: 2026-03-19  
**Status**: Implemented

## Summary

Align `teacher_without_semantic_env_cfg.py` fully with `unitree_rl_lab`'s `velocity_env_cfg.py`:
- Scene, terrain, and MDP match velocity_env_cfg
- No unitree_rl_lab dependency
- Use go2_pvcnn.mdp (isaaclab + custom), go2_pvcnn.assets.UNITREE_GO2_CFG
- train.py, play.py, OnPolicyRunner unchanged

## Decisions

| Decision | Choice |
|----------|--------|
| Alignment scope | Full (B): Scene + Terrain + MDP |
| MDP source | No unitree_rl_lab; isaaclab + go2_pvcnn custom |
| Approach | Copy-and-adapt (方案 2) |
| Missing functions | Implement in go2_pvcnn.mdp, name with `_unitree_rl_lab` suffix |

## Implementation

### 1. go2_pvcnn/mdp/curriculums.py

Added `terrain_levels_vel_unitree_rl_lab()` — terrain curriculum based on velocity-tracking distance (from VR-Robo / isaaclab_tasks).

### 2. teacher_without_semantic_env_cfg.py — Rewrite

**Scene (RobotSceneCfg)**
- Terrain: COBBLESTONE_ROAD_CFG with full sub_terrains (flat, random_rough, slopes, boxes, stairs)
- height_scanner (RayCasterCfg)
- contact_forces
- sky_light
- No teacher objects, no furniture

**Terrain**
- ISAACLAB_NUCLEUS_DIR / ISAAC_NUCLEUS_DIR for materials
- curriculum enabled when terrain_levels curriculum is present

**MDP (aligned with velocity_env_cfg)**
- Commands: UniformLevelVelocityCommandCfg, ranges + limit_ranges
- Actions: JointPositionAction, scale=0.25, clip
- Observations: Policy (no base_lin_vel), Critic (with joint_effort)
- Rewards: track_lin_vel_xy (1.5), track_ang_vel_z (0.75), joint_vel, joint_torques, action_rate, dof_pos_limits, energy, flat_orientation, joint_position_penalty, feet_air_time, air_time_variance, feet_slide, undesired_contacts
- Terminations: time_out, base_contact, bad_orientation
- Events: physics_material (0.3–1.2), add_base_mass (-1, 3), reset_base (vel 0), reset_robot_joints (1.0, -1–1), push_robot (5–10 s)
- Curriculum: terrain_levels_vel_unitree_rl_lab, lin_vel_cmd_levels

**Play (TeacherWithoutSemanticEnvCfg_PLAY)**
- num_envs=32, terrain 2×1
- commands.ranges = limit_ranges
- No push, no observation corruption

## Files Changed

- `Go2Pvcnn/go2_pvcnn/mdp/curriculums.py` — added `terrain_levels_vel_unitree_rl_lab`
- `Go2Pvcnn/go2_pvcnn/tasks/teacher_without_semantic_env_cfg.py` — full rewrite

## Verification

- Syntax valid
- train.py / play.py / agent unchanged; use existing experiment mappings
