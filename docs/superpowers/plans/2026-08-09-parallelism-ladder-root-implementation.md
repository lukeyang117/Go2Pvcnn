# Parallelism Ladder Root Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single Parallelism ladder/terrain-aware RL config where flat terrain keeps the current cross-obstacles root behavior and every non-flat terrain uses a height-map-following root trajectory.

**Architecture:** The reference manager resolves IsaacLab terrain type names from `scene.terrain.terrain_types` and `terrain_generator.sub_terrains`. It passes a per-env non-flat mask into the Parallelism planner; root rollout uses the existing stance-foot rule for flat envs and a new height-map root rule for non-flat envs. A new RL config reuses teacher semantic terrain types and applies the previous 40-small-obstacle layout only to the `flat` terrain type.

**Tech Stack:** Python 3.10, PyTorch tensor batch operations, IsaacLab config classes, pytest static/unit tests.

## Global Constraints

- Only root trajectory generation changes in the planner path.
- Foot touchdown candidates, IK, FK, collision filter, semantic filter, score, and RL tracking interface stay unchanged.
- `terrain_name == "flat"` uses the current root behavior.
- `terrain_name != "flat"` uses terrain-following root.
- New config uses teacher terrain types and puts 40 semantic small obstacles only on `flat`.
- All root logic remains torch-batched.
- Preserve existing uncommitted user config/script changes unless explicitly part of the task.

---

### Task 1: Terrain-following root rollout

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/config.py`
- Modify: `Go2Pvcnn/extension/parallelism/root.py`
- Modify: `Go2Pvcnn/extension/parallelism/planner.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_terrain_following_root.py`

**Interfaces:**
- Produces: `rollout_root(..., terrain_following_mask: Tensor | None = None) -> RootRollout`
- Produces: `soft_clamp_terrain_command(command_body: Tensor, cfg: ParallelismCfg) -> Tensor`

- [x] Write tests proving flat mask keeps old stance-foot root z.
- [x] Write tests proving non-flat mask makes root z follow height map at root xy and keeps frame 0 equal to current root z.
- [x] Write tests proving non-flat command soft clamp reduces only the excess part above soft limits.
- [x] Write tests proving non-flat roll/pitch follows local height differences with limits.
- [x] Add terrain-following config fields to `ParallelismCfg`.
- [x] Add root helpers for soft clamp, z smoothing/rate limiting, and slope roll/pitch.
- [x] Update `rollout_root` to blend old/new root output by `terrain_following_mask`.
- [x] Update `plan_trajectory` to accept and forward `terrain_following_mask`.
- [x] Run the new root tests.

### Task 2: Terrain type metadata in the reference manager

**Files:**
- Modify: `Go2Pvcnn/tracking/managers/parallelism_reference_manager.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py`

**Interfaces:**
- Produces: `ParallelismReferenceManager._terrain_following_mask(env_ids: Tensor) -> Tensor`
- Consumes: `plan_trajectory(..., terrain_following_mask=mask)`

- [x] Add tests for terrain type name resolution where col 0 is `flat` and other cols are non-flat.
- [x] Add tests for missing terrain metadata defaulting to flat behavior.
- [x] Implement terrain name resolution from `scene.terrain.terrain_types` and `scene.terrain.cfg.terrain_generator.sub_terrains`.
- [x] Pass the resolved non-flat mask into `plan_trajectory`.
- [x] Run manager tests.

### Task 3: One ladder terrain-aware RL config

**Files:**
- Create: `Go2Pvcnn/tracking/parallelism_ladder_env_cfg.py`
- Modify: `Go2Pvcnn/tracking/register_envs.py`
- Modify: `Go2Pvcnn/scripts/train.py`
- Modify: `Go2Pvcnn/scripts/play.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_ladder_env_cfg_static.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_env_cfg_static.py`

**Interfaces:**
- Produces: experiment name `parallelism_tracking_ladder`
- Produces: Gym id `Isaac-Go2-Parallelism-Tracking-Ladder-v0`
- Produces: play config `ParallelismTrackingLadderEnvCfg_PLAY`

- [x] Add static tests that the config imports teacher `SEMANTIC_TERRAIN_CFG` terrain names.
- [x] Add static tests that `flat` receives 40 small obstacles and non-flat receives 0.
- [x] Add static tests that train/play/register include the new experiment and Gym id.
- [x] Implement `ParallelismTrackingLadderEnvCfg` by inheriting current Parallelism tracking rewards/observations/curriculum and setting teacher terrain generator plus semantic course settings.
- [x] Register the new Gym task.
- [x] Add `parallelism_tracking_ladder` to train/play experiment maps and argparse choices.
- [x] Run static config tests.

### Task 4: Verification and commit

**Files:**
- Modify plan checklist in this file as tasks complete.

- [x] Run focused tests:
  `PYTHONPATH=Go2Pvcnn /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/tracking/test_parallelism_terrain_following_root.py Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py Go2Pvcnn/tests/tracking/test_parallelism_ladder_env_cfg_static.py Go2Pvcnn/tests/tracking/test_parallelism_tracking_env_cfg_static.py -q`
- [x] Run broader tracking/static tests:
  `PYTHONPATH=Go2Pvcnn /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/tracking Go2Pvcnn/tests/test_train_script_static.py -q`
- [x] Run `python -m compileall -q Go2Pvcnn/extension/parallelism Go2Pvcnn/tracking Go2Pvcnn/scripts/train.py Go2Pvcnn/scripts/play.py`.
- [x] Run `git diff --check`.
- [x] Review `git diff` and ensure unrelated user changes remain uncommitted unless intentionally staged.
- [x] Commit implementation:
  `git commit -m "feat: add terrain-following parallelism root"`
