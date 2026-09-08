# Go2 SemLoco Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Add an isolated SemLoco-style semantic Raibert foothold planner and rewards for Go2, with a runnable PPO training/play path and tests.

**Architecture:** Keep the existing Go2 CNN+MLP actor-critic and PPO untouched. Add pure batched planner/reward modules, an environment-side lifecycle adapter that owns per-environment target caches, and a separately registered SemLoco environment/config plus scripts.

**Tech Stack:** Python 3.10, PyTorch, IsaacLab ManagerBasedRLEnv/configclass, Gymnasium, pytest, shell scripts.

## Global Constraints

- Do not modify behavior of existing PPO, AMP, Distillation, Teacher, or AME paths.
- No AME integration, no Virtual Obstacle -> Rigid Obstacle staged training, and no planner-generated actions.
- Preserve current observation and action shapes and reuse current PPO hyperparameters.
- Planner operations must be batched, finite-safe, and device-preserving.
- Planner invalidity disables only foothold tracking for affected legs; clearance and base rewards continue.

### Task 1: Pure Planner Geometry

**Files:**
- Create: `Go2Pvcnn/semloco/__init__.py`
- Create: `Go2Pvcnn/semloco/semantic_raibert_planner.py`
- Test: `Go2Pvcnn/tests/semloco/test_semantic_raibert_planner.py`

**Interfaces:** `SemanticRaibertPlanner.plan(...) -> PlannerOutput`; `PlannerOutput` exposes target, nominal, valid, collision-free tensors.

- [ ] Write tests for batched shapes, Raibert vx/vy/vyaw direction, candidate selection, inflated obstacle rejection, all-invalid fallback, and finite outputs.
- [ ] Run planner tests and confirm failure before implementation.
- [ ] Implement config dataclass, trot phase mapping, nominal/Raibert target computation, 5x5 candidate search, semantic height sampling, AABB+foot-radius collision test, and invalid fallback.
- [ ] Run planner tests until passing.

### Task 2: Rewards and Lifecycle Adapter

**Files:**
- Create: `Go2Pvcnn/semloco/semloco_rewards.py`
- Create: `Go2Pvcnn/semloco/semloco_lifecycle.py`
- Test: `Go2Pvcnn/tests/semloco/test_semloco_rewards.py`
- Test: `Go2Pvcnn/tests/semloco/test_semloco_lifecycle.py`

**Interfaces:** `semantic_foothold_tracking_reward(...) -> Tensor`; `semloco_clearance_penalty(...) -> Tensor`; `SemlocoTargetCache.update_swing_targets(...)` and `.reset(...)`.

- [ ] Add failing tests for valid-mask normalization, zero-valid behavior, one-sided clearance ReLU, planner-invalid clearance behavior, and swing-start locking/touchdown replacement.
- [ ] Implement pure reward functions with explicit `[N,4]` masks and finite-safe arithmetic.
- [ ] Implement cache lifecycle that updates only newly entering swing legs and preserves targets mid-swing.
- [ ] Run reward/lifecycle tests until passing.

### Task 3: Isolated Environment Configuration and Registration

**Files:**
- Create: `Go2Pvcnn/semloco/semloco_env_cfg.py`
- Create: `Go2Pvcnn/semloco/semloco_train_cfg.py`
- Modify: `Go2Pvcnn/tracking/register_envs.py` (add only SemLoco import/registration)
- Test: `Go2Pvcnn/tests/semloco/test_semloco_env_cfg_static.py`

**Interfaces:** `SemlocoCrossLargeComplexEnvCfg`, `SemlocoCrossLargeComplexEnvCfg_PLAY`, and Gym ID `Isaac-Go2-Cross-Large-Complex-SemLoco-v0`.

- [ ] Test that the new config has experiment name, 1024 default environments, SemLoco reward terms, unchanged action/observation declarations, and independent Gym registration.
- [ ] Derive the new config from the current cross-large-complex PPO config, add planner/cache initialization and post-step reward hooks without editing the base config.
- [ ] Add independent PPO runner settings and register train/play entries.
- [ ] Run static config tests and import checks.

### Task 4: Train/Play Entrypoints

**Files:**
- Create: `Go2Pvcnn/scripts/train_cross_large_complex_semloco.py`
- Create: `Go2Pvcnn/scripts/train_cross_large_complex_semloco_headless.sh`
- Create: `Go2Pvcnn/scripts/play_cross_large_complex_semloco.py`
- Test: `Go2Pvcnn/tests/semloco/test_semloco_launchers_static.py`

**Interfaces:** CLI supports `--num-envs`, `--max-iterations`, `--headless`, `--resume`, and checkpoint path using existing project conventions.

- [ ] Add static launcher tests for Gym ID, experiment name, resume forwarding, and executable shell behavior.
- [ ] Implement training script by following existing PPO launcher patterns and loading the isolated config.
- [ ] Implement headless shell with environment activation/path setup and argument forwarding.
- [ ] Implement play script with the isolated PLAY config and checkpoint loading.
- [ ] Run launcher tests and `--help` checks.

### Task 5: Real IsaacLab Smoke

**Files:**
- Modify: `Go2Pvcnn/tests/semloco/test_semloco_smoke.py` (integration test/command documentation)

- [ ] Run the independent entrypoint with 1024 environments, headless mode, and 4 iterations.
- [ ] Verify normal exit, finite PPO tensors, unchanged action/observation shapes, and both SemLoco reward names in output/logs.
- [ ] Run focused existing tracking static tests to verify no regressions.
- [ ] Record command and observed results in the final handoff.

