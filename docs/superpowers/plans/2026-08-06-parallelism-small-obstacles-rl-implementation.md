# Parallelism Small Obstacles RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `parallelism_tracking_small_obstacles` RL experiment that inherits the flat Parallelism tracking task and trains on one fixed small-obstacle subterrain.

**Architecture:** Keep the existing Parallelism planner and flat RL tracking logic. Add a focused scene-layout helper for the single fixed small-obstacle subterrain, a new env config that inherits flat, new reward/metric terms in `tracking.mdp`, and register the experiment in train/play/gym entry points.

**Tech Stack:** Python 3.10, Isaac Lab config classes, Torch tensor rewards, pytest static/unit tests, RSL-RL train/play scripts.

## Global Constraints

- Create a new file for the new task config; do not fold the task into `parallelism_tracking_env_cfg.py`.
- The small-obstacle terrain is one scene and one subterrain only: `num_rows=1`, `num_cols=1`, one `small_obstacles` subterrain.
- The obstacle patch is a square with a circular reset hole: `obstacle_patch_size_m=2.0`, `reset_clear_radius_m=0.25`, `obstacle_center_exclusion_radius_m=0.30`.
- Use `small_obstacle_count=24`, `small_obstacle_jitter_m=0.03`, `small_obstacle_min_spacing_m=0.18`, `large_obstacle_count=0`.
- Keep the current flat velocity command curriculum and only relax curriculum thresholds.
- Do not add a minimum translational speed filter.
- Do not reset on Parallelism geometry collision; use reward penalty only.
- Use Torch tensor conditionals for reward logic.
- Commit after completed, verified slices.

---

### Task 1: Single Subterrain Layout Helper

**Files:**
- Create: `Go2Pvcnn/tracking/parallelism_small_obstacles_scene.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_small_obstacles_scene.py`

**Interfaces:**
- Produces: `ParallelismSmallObstacleSceneCfg`
- Produces: `build_small_obstacle_local_xy(cfg: ParallelismSmallObstacleSceneCfg) -> tuple[tuple[float, float], ...]`
- Produces: `small_obstacles_terrain_cfg(cfg: ParallelismSmallObstacleSceneCfg) -> terrain_gen.TerrainGeneratorCfg`

- [ ] Write a failing test that imports the helper, expects exactly 24 obstacle centers, expects every center inside the 2m square, and expects every center outside radius 0.30.
- [ ] Run `pytest Go2Pvcnn/tests/tracking/test_parallelism_small_obstacles_scene.py -q` and confirm import failure.
- [ ] Implement the helper with deterministic grid-plus-jitter centers and a single-subterrain terrain generator.
- [ ] Run the new scene test and confirm pass.
- [ ] Commit with `feat: add fixed small obstacle scene helper`.

### Task 2: Experiment Config and Registration

**Files:**
- Create: `Go2Pvcnn/tracking/parallelism_small_obstacles_env_cfg.py`
- Modify: `Go2Pvcnn/tracking/register_envs.py`
- Modify: `Go2Pvcnn/scripts/train.py`
- Modify: `Go2Pvcnn/scripts/play.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_small_obstacles_env_cfg_static.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py`
- Test: `Go2Pvcnn/tests/test_train_script_static.py`

**Interfaces:**
- Consumes: `small_obstacles_terrain_cfg`
- Produces: `ParallelismTrackingSmallObstaclesEnvCfg`
- Produces: `ParallelismTrackingSmallObstaclesEnvCfg_PLAY`

- [ ] Write failing static tests that the new config uses experiment name `parallelism_tracking_small_obstacles`, one terrain row, one terrain col, and `small_obstacles` subterrain.
- [ ] Write failing static assertions that train.py, play.py, and gym registration include `parallelism_tracking_small_obstacles`.
- [ ] Run the targeted static tests and confirm failure.
- [ ] Implement the env config by inheriting `ParallelismTrackingFlatEnvCfg`, replacing terrain config with the single subterrain, enabling semantic scanner obstacles, relaxing curriculum thresholds, and preserving flat command ranges.
- [ ] Register the new experiment in train.py, play.py, and tracking/register_envs.py.
- [ ] Run the targeted static tests and confirm pass.
- [ ] Commit with `feat: register parallelism small obstacles rl task`.

### Task 3: Obstacle Rewards and Metrics

**Files:**
- Modify: `Go2Pvcnn/tracking/mdp/rewards.py`
- Modify: `Go2Pvcnn/tracking/mdp/__init__.py`
- Modify: `Go2Pvcnn/tracking/parallelism_small_obstacles_env_cfg.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_small_obstacles_rewards.py`

**Interfaces:**
- Produces: `parallelism_geometry_collision_penalty(env, asset_cfg, scanner_cfg) -> torch.Tensor`
- Produces: `active_swing_foot_on_small_obstacle_reward(env, asset_cfg, scanner_cfg) -> torch.Tensor`
- Produces: `parallelism_obstacle_episode_metrics(env) -> dict[str, torch.Tensor]`

- [ ] Write failing lightweight tests for active swing foot semantic reward and collision penalty using fake env/asset/scanner tensors.
- [ ] Run the new reward test and confirm failure.
- [ ] Implement Torch-only reward helpers that query semantic/elevation maps through already available scanner tensors when present and safely return zeros when the scanner data is absent.
- [ ] Add reward terms to `ParallelismSmallObstaclesRewardsCfg`.
- [ ] Run the new reward test and confirm pass.
- [ ] Commit with `feat: add parallelism small obstacle rewards`.

### Task 4: Verification

**Files:**
- No new files expected unless smoke-test scripts need a tiny static helper.

- [ ] Run unit/static tests:
  `pytest Go2Pvcnn/tests/tracking/test_parallelism_small_obstacles_scene.py Go2Pvcnn/tests/tracking/test_parallelism_small_obstacles_env_cfg_static.py Go2Pvcnn/tests/tracking/test_parallelism_small_obstacles_rewards.py Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py Go2Pvcnn/tests/test_train_script_static.py -q`
- [ ] Run import smoke with Isaac Python:
  `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -c "from tracking.parallelism_small_obstacles_env_cfg import ParallelismTrackingSmallObstaclesEnvCfg; cfg=ParallelismTrackingSmallObstaclesEnvCfg(); print(cfg.experiment_name, cfg.scene.terrain.terrain_generator.num_rows, cfg.scene.terrain.terrain_generator.num_cols)"`
- [ ] Run 1024-env headless smoke for 4 iterations:
  `Go2Pvcnn/scripts/train.py --experiment parallelism_tracking_small_obstacles --num_envs 1024 --max_iterations 4 --device cuda:0 --headless`
- [ ] If smoke fails, fix the root cause and rerun the failed command.
- [ ] Commit final fixes if any.
