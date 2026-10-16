# M1 AME Pure PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 在 m1_rl 分支新增不依赖 Parallelism 的 M1 AME 纯 PPO 训练与播放链路，使地图、地形课程、PPO 算法和 AME 网络完全复用 Go2，仅将机器人资产、16 维动作/状态接口和 M1 尺寸相关 reward 对齐到 M1。

**Architecture:** 复用 Go2 的 AmeRslRlEnvWrapper、ActorCriticAME、AmeOnPolicyRunner 和 PPO 配置。新增独立的 m1_ame 配置模块，继承 Go2 纯 PPO 地形/课程配置，但用 M1_CFG、M1 scanner body path、显式 16 关节 selector 和 M1 reward terms 覆盖机器人相关部分。训练脚本单独注册 M1 env id，避免修改 Go2 AME 入口。

**Tech Stack:** Python 3.10, Isaac Lab, Isaac Sim, Gymnasium, PyTorch, local rsl_rl, pytest。

## Global Constraints

- 地图固定为 Go2 AME 的 6 通道、16×16 下采样地图。
- terrain generator、semantic obstacle curriculum、terrain level curriculum 和 command curriculum 复用 Go2 纯 PPO 配置。
- PPO 算法参数、rollout/update 逻辑、ActorCriticAME CNN/MHA/MLP 结构全部复用 Go2。
- M1 policy observation 为 1594 维，critic observation 为 1597 维，action 为 16 维。
- M1 joint 顺序固定为 M1_USD_JOINT_NAMES，包含 12 个腿部关节和 4 个轮子关节。
- policy/critic 的 joint_pos、joint_vel 和 last_action 都包含 16 个关节对应维度。
- 轮子 state 可以进入 pure PPO policy/critic；Parallelism state、reference trajectory 和 planner 不进入本任务。
- Go2 AME 文件不得被修改来适配 M1；M1 使用独立模块和 env id。
- M1 root 初始高度使用 M1_ROOT_Z_M，轮盘尺寸使用 M1 已有半径、厚度和水平包络常量。
- reward 权重和 PPO 超参数保持 Go2；reward 的 selector、接触语义、尺寸阈值和归一化按 M1 修改。
- 每个任务先写失败测试，确认失败原因来自缺少 M1 AME 实现，再写最小实现。

---

### Task 1: 建立 M1 AME 契约测试

**Files:**
- Create: Go2Pvcnn/tests/m1_ame/test_m1_ame_contract.py
- Create: Go2Pvcnn/tests/m1_ame/__init__.py
- Create: Go2Pvcnn/m1_ame/__init__.py

**Interfaces:**
- Tests import m1_ame.m1_ame_env_cfg.M1AmeCrossLargeComplexEnvCfg.
- Tests import m1_ame.m1_ame_train_cfg.get_m1_ame_train_cfg.
- Tests expect M1_AME_ENV_ID = Isaac-M1-Cross-Large-Complex-AME-v0.
- Tests expect M1_AME_POLICY_OBS_DIM = 1594, M1_AME_CRITIC_OBS_DIM = 1597, M1_AME_ACTION_DIM = 16.

- [ ] **Step 1: Write failing tests.** Test the environment class, env id constants, explicit 16-joint ordering, observation dimension constants, unchanged Go2 AME PPO keys, and absence of planner/reference fields in the M1 AME config module.
- [ ] **Step 2: Run the contract tests and verify the expected failure.**

    pytest -q Go2Pvcnn/tests/m1_ame/test_m1_ame_contract.py

    Expected: FAIL because m1_ame and M1AmeCrossLargeComplexEnvCfg do not exist.

- [ ] **Step 3: Add package constants only.** Define the env id and dimension constants in Go2Pvcnn/m1_ame/__init__.py; do not add environment implementation in this step.
- [ ] **Step 4: Run the tests again.** Confirm the failure moves to missing environment/config symbols rather than import or test syntax errors.
- [ ] **Step 5: Commit.**

    git add Go2Pvcnn/tests/m1_ame Go2Pvcnn/m1_ame
    git commit -m "test: define M1 AME pure PPO contract"

### Task 2: Import the Go2 AME pure-PPO baseline modules

**Files:**
- Create: `Go2Pvcnn/ame_baseline/actor_critic_ame.py`
- Create: `Go2Pvcnn/ame_baseline/ame_env_cfg.py`
- Create: `Go2Pvcnn/ame_baseline/ame_env_wrapper.py`
- Create: `Go2Pvcnn/ame_baseline/ame_observations.py`
- Create: `Go2Pvcnn/ame_baseline/ame_runner.py`
- Create: `Go2Pvcnn/ame_baseline/ame_train_cfg.py`
- Create: `Go2Pvcnn/tracking/cross_large_complex_ppo_env_cfg.py`
- Create: `Go2Pvcnn/tracking/mdp/gait.py`
- Create: `Go2Pvcnn/tracking/mdp/distillation.py`
- Create: `Go2Pvcnn/tracking/mdp/policy_geometry_rewards.py`
- Modify: `Go2Pvcnn/tracking/mdp/__init__.py`

**Interfaces:**
- The imported AME files must match the Go2 reference branch byte-for-byte in algorithm and network behavior.
- `CrossLargeComplexPpoEnvCfg` must remain the planner-free Go2 pure-PPO base used by M1 inheritance.
- `tracking.mdp` must expose fixed_gait_phase, fixed_gait_contact_match_reward, and policy_geometry_collision_penalty without adding M1-specific behavior.

- [ ] **Step 1: Write a source parity test.** Compare the copied pure-PPO baseline files against the Go2 reference branch after normalization of the repository root path.
- [ ] **Step 2: Run the parity test and verify failure.**

    pytest -q Go2Pvcnn/tests/m1_ame/test_go2_ame_baseline_parity.py

- [ ] **Step 3: Copy the reference files through apply_patch without modifying algorithm or network code.** Add only files missing from m1_rl and update tracking.mdp exports to match the reference dependency graph.
- [ ] **Step 4: Run the parity test and verify pass.**
- [ ] **Step 5: Commit the imported baseline.**

    git add Go2Pvcnn/ame_baseline Go2Pvcnn/tracking
    git commit -m "feat: import Go2 AME pure PPO baseline"

### Task 3: Add the M1 AME scene, observations, and 16-dimension action wiring

**Files:**
- Create: Go2Pvcnn/m1_ame/m1_ame_env_cfg.py
- Modify: Go2Pvcnn/m1_ame/__init__.py
- Test: Go2Pvcnn/tests/m1_ame/test_m1_ame_contract.py

**Interfaces:**
- Produce M1AmeCrossLargeComplexEnvCfg and M1AmeCrossLargeComplexEnvCfg_PLAY.
- Produce M1AmeObservationsCfg with the four groups expected by AmeRslRlEnvWrapper.
- Produce M1AmeActionsCfg with all 16 M1_USD_JOINT_NAMES in USD order.
- Produce M1AmeSceneCfg with M1_CFG, scanner path {ENV_REGEX_NS}/Robot/BASE_LINK, unchanged 1.5 m scanner pattern, unchanged semantic ids, and the existing Go2 terrain generator.

- [ ] **Step 1: Extend failing tests.** Assert that the scene robot is M1, scanner path contains BASE_LINK, terrain generator and semantic curriculum are present, action total dimension is 16 when Isaac Lab is available, and observation terms contain 16-dimensional joint position, joint velocity, and last action terms.
- [ ] **Step 2: Run focused tests and verify failure.**

    pytest -q Go2Pvcnn/tests/m1_ame/test_m1_ame_contract.py

- [ ] **Step 3: Implement the scene by subclassing the Go2 pure PPO scene config.** Keep the imported terrain generator, semantic layout, scanner size, semantic ids, lighting, and curriculum values from TeacherElevationTrajectoryMpcSemanticSceneCfg; override only robot and scanner prim path. Set M1_CFG.init_state.pos.z through the existing M1 asset config.
- [ ] **Step 4: Implement observations by copying Go2 AME term structure.** Keep map terms unchanged. Keep base angular velocity, projected gravity, command, action history, gait phase, noise, and concatenation behavior unchanged. Use joint_pos_rel and joint_vel_rel with explicit SceneEntityCfg("robot", joint_names=list(M1_USD_JOINT_NAMES)) so all 16 M1 joints, including wheels, enter policy and critic state. Critic adds base_lin_vel exactly as Go2.
- [ ] **Step 5: Implement actions with the explicit 16-joint selector.** Use M1 USD order and Go2 action scale/default offset/clip semantics for the first implementation.
- [ ] **Step 6: Implement the M1 env config.** Subclass CrossLargeComplexPpoEnvCfg, replace scene/observations/actions, keep planner_owned_reference_cache=False and use_batched_reference_trajectory=False, and do not attach a trajectory manager.
- [ ] **Step 7: Run focused tests and verify pass.**

    pytest -q Go2Pvcnn/tests/m1_ame/test_m1_ame_contract.py

- [ ] **Step 8: Commit.**

    git add Go2Pvcnn/m1_ame
    git commit -m "feat: add M1 AME scene observations and 16D actions"

### Task 4: Add M1-sized reward and termination configuration

**Files:**
- Modify: Go2Pvcnn/m1_ame/m1_ame_env_cfg.py
- Create: Go2Pvcnn/m1_ame/m1_ame_rewards.py
- Create: Go2Pvcnn/tests/m1_ame/test_m1_ame_rewards.py

**Interfaces:**
- Produce M1AmeRewardsCfg and M1AmeTerminationsCfg.
- Produce explicit selector constants for M1 leg joints, wheel joints, support bodies, and non-support collision bodies.
- Keep Go2 reward keys and weights where semantics remain valid; use separate M1 terms/selectors where wheel semantics differ.

- [ ] **Step 1: Write failing reward contract tests.** Assert that joint posture and joint limit terms select only 12 leg joints; wheel state is not subject to leg posture or position-limit penalties; torque/energy/action-rate terms can cover all 16 action dimensions; contact selectors use M1 body names; root height references M1_ROOT_Z_M; wheel radius/thickness/envelope constants are used by geometry terms; no Parallelism reference reward is present.
- [ ] **Step 2: Run reward tests and verify failure.**

    pytest -q Go2Pvcnn/tests/m1_ame/test_m1_ame_rewards.py

- [ ] **Step 3: Implement selector constants and helper config.** Use M1_PLANNER_JOINT_NAMES for the 12 leg joints, M1_WHEEL_JOINT_NAMES for four wheels, and explicit M1 support body names from the asset/viewer config. Do not use unvalidated .*_foot or .* selectors for M1 reward terms.
- [ ] **Step 4: Implement M1 reward terms.** Preserve Go2 weights for command tracking, base velocity, orientation, PPO-compatible regularization, and semantic geometry collision where the term accepts M1 selectors. Restrict joint_pos, joint_vel, and dof_pos_limits to leg joints. Keep wheel torque/energy/action-rate accounting available for all 16 actions. Prevent normal wheel rolling contact from being treated as Go2 foot sliding or undesired contact. Use M1 radius and envelope in clearance/collision thresholds.
- [ ] **Step 5: Implement M1 terminations.** Keep timeout and bad orientation semantics from Go2; replace base/support selectors with M1 names and use M1 root/geometry thresholds. Do not add planner-reference termination terms.
- [ ] **Step 6: Attach rewards and terminations to M1AmeCrossLargeComplexEnvCfg.** Preserve original Go2 terrain and command curriculum values.
- [ ] **Step 7: Run reward and contract tests.**

    pytest -q Go2Pvcnn/tests/m1_ame/test_m1_ame_rewards.py Go2Pvcnn/tests/m1_ame/test_m1_ame_contract.py

- [ ] **Step 8: Commit.**

    git add Go2Pvcnn/m1_ame
    git commit -m "feat: align M1 AME rewards with wheel geometry"

### Task 5: Add M1 AME PPO config and checkpoint dimension guard

**Files:**
- Create: Go2Pvcnn/m1_ame/m1_ame_train_cfg.py
- Create: Go2Pvcnn/m1_ame/m1_ame_runner.py
- Create: Go2Pvcnn/tests/m1_ame/test_m1_ame_train_cfg.py

**Interfaces:**
- Produce get_m1_ame_train_cfg().
- Produce M1AmeOnPolicyRunner with the same runner behavior as AmeOnPolicyRunner, but an M1 architecture signature and expected dimensions (1594, 1597, 16).

- [ ] **Step 1: Write failing tests.** Assert all PPO and AME network values equal Go2 get_ame_train_cfg(); only M1 dimensions/signature differ; the M1 runner rejects a Go2 checkpoint signature or wrong dimension tuple.
- [ ] **Step 2: Run tests and verify failure.**

    pytest -q Go2Pvcnn/tests/m1_ame/test_m1_ame_train_cfg.py

- [ ] **Step 3: Implement get_m1_ame_train_cfg().** Deep-copy Go2 get_ame_train_cfg() and preserve every algorithm/policy value. Do not add M1-specific hidden layers, normalization, recurrent state, AMP, or Parallelism fields.
- [ ] **Step 4: Implement the M1 runner.** Reuse AmeOnPolicyRunner save/load behavior, change only architecture signature and dimension validation to M1 values. Keep optimizer, curriculum state, standard deviation, and checkpoint semantics unchanged.
- [ ] **Step 5: Run focused tests and verify pass.**

    pytest -q Go2Pvcnn/tests/m1_ame/test_m1_ame_train_cfg.py

- [ ] **Step 6: Commit.**

    git add Go2Pvcnn/m1_ame
    git commit -m "feat: add M1 AME PPO configuration and runner"

### Task 6: Add M1 training and play entrypoints

**Files:**
- Create: Go2Pvcnn/scripts/train_m1_cross_large_complex_ame.py
- Create: Go2Pvcnn/scripts/play_m1_cross_large_complex_ame.py
- Create: Go2Pvcnn/tests/m1_ame/test_m1_ame_entrypoints.py

**Interfaces:**
- Training accepts --num_envs, --max_iterations, --seed, --resume, --checkpoint, --log_dir, and Isaac Lab AppLauncher args.
- Play accepts --num_envs=1, --checkpoint, --seed, and Isaac Lab AppLauncher args.
- Training env id is Isaac-M1-Cross-Large-Complex-AME-v0; play env id is Isaac-M1-Cross-Large-Complex-AME-Play-v0.

- [ ] **Step 1: Write failing static entrypoint tests.** Assert each script imports M1 config/runner/train config, registers only M1 env ids, does not import or attach Parallelism/reference manager, and passes observations through AmeRslRlEnvWrapper.
- [ ] **Step 2: Run tests and verify failure.**

    pytest -q Go2Pvcnn/tests/m1_ame/test_m1_ame_entrypoints.py

- [ ] **Step 3: Implement training entrypoint by copying Go2 AME lifecycle.** Keep AppLauncher setup, CUDA/TensorFloat configuration, log metadata, YAML dumps, resume validation, wrapper, runner creation, and runner.learn behavior. Replace only config, env id, runner, train config, and experiment/log names.
- [ ] **Step 4: Implement play entrypoint using the existing play pattern.** Load the M1 runner checkpoint, call act_inference, step M1 env, and preserve visual/non-headless AppLauncher behavior.
- [ ] **Step 5: Run static entrypoint tests and all M1 pure-PPO unit tests.**

    pytest -q Go2Pvcnn/tests/m1_ame

- [ ] **Step 6: Commit.**

    git add Go2Pvcnn/scripts Go2Pvcnn/tests/m1_ame
    git commit -m "feat: add M1 AME PPO train and play entrypoints"

### Task 7: Run Isaac Lab smoke tests and protect Go2 behavior

**Files:**
- Create: Go2Pvcnn/tests/m1_ame/m1_ame_isaac_smoke.py
- Create: Go2Pvcnn/scripts/train_m1_cross_large_complex_ame_headless.sh
- Create: Go2Pvcnn/scripts/play_m1_cross_large_complex_ame.sh
- Modify: only M1-local implementation or entrypoint files if smoke evidence identifies a concrete M1 issue.

- [ ] **Step 1: Run static and unit verification.**

    pytest -q Go2Pvcnn/tests/m1_ame
    git diff --check HEAD~1..HEAD

- [ ] **Step 2: Run an 8-env, one-update M1 smoke.** Verify environment creation, observation shapes (8,1594) and (8,1597), action shape (8,16), finite reward, and one PPO update.
- [ ] **Step 3: Run a 1024-env short smoke.** Verify unchanged map/terrain curriculum initializes and completes one rollout without NaN, planner imports, CPU per-env loops, or wheel selector errors.
- [ ] **Step 4: Run existing Go2 AME static tests.** Confirm Go2 AME files and configs retain their original values and entrypoint behavior.
- [ ] **Step 5: Record smoke evidence with env id, git commit, observation/action shapes, finite reward check, and checkpoint path.
- [ ] **Step 6: Review final diff and status.** Ensure only M1-local implementation, tests, scripts, and plan files changed.

Expected final verification:

    pytest -q Go2Pvcnn/tests/m1_ame
    pytest -q Go2Pvcnn/tests/ame_baseline/test_ame_observations.py Go2Pvcnn/tests/test_actor_critic_ame.py
    git status --short --branch
