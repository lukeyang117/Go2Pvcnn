# Consecutive Standstill Termination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Parallelism 连续两次规划失败并进入 standstill 时终止当前 RL episode，同时保留第一次失败后的 `standard_state` 重规划逻辑。

**Architecture:** `ParallelismReferenceManager` 维护每个环境的连续 standstill 计数。每次 replan 成功时清零，环境 reset 和新 episode 初始化时清零；command 改变不清零。新的 MDP termination 只判断计数是否达到 2，训练配置启用该 term，play 面板显示并可单独屏蔽该 term。

**Tech Stack:** Python 3.10, PyTorch tensors, Isaac Lab ManagerBasedRLEnv termination terms, pytest.

## Global Constraints

- standstill 计数按 replan 事件更新，不按 physics step 更新。
- 只有环境 reset、Parallelism 成功规划、新 episode 初始化可以清零。
- 第一次 standstill 后，下一次规划继续使用现有 `standard_state`。
- command 改变不清零计数。
- 不修改现有 Parallelism planner 的候选、filter、score 逻辑。
- 保留工作区中用户已有的 reward、spacing、jitter、joint termination 和训练脚本修改。

---

### Task 1: Manager 连续 standstill 状态

**Files:**
- Modify: `Go2Pvcnn/tracking/managers/parallelism_reference_manager.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py`

**Interfaces:**
- Produces: `ParallelismReferenceManager.standstill_count`，形状为 `[num_envs]` 的 long tensor。
- Existing: `standstill_latched` 继续表示上一次规划是否失败，并继续驱动 `standard_state`。

- [x] 添加 manager 单测，验证首次失败后计数为 1，第二次连续失败为 2。
- [x] 添加单测验证成功规划将计数清零。
- [x] 添加单测验证 `reset()` 将计数清零。
- [x] 添加单测验证 `mark_command_changed()` 不清零计数。
- [x] 在 manager 初始化时创建 `standstill_count`。
- [x] 在 `reset()` 中只对指定 env ids 清零计数。
- [x] 在 `_plan()` 中根据 `trajectory.valid` 更新计数；成功置零，失败加一，并保留现有 `standstill_latched = ~trajectory.valid`。
- [x] 运行 manager 测试，确认通过。

### Task 2: MDP termination 和配置

**Files:**
- Modify: `Go2Pvcnn/tracking/mdp/terminations.py`
- Modify: `Go2Pvcnn/tracking/mdp/__init__.py`
- Modify: `Go2Pvcnn/tracking/parallelism_tracking_env_cfg.py`
- Modify: `Go2Pvcnn/tracking/parallelism_small_obstacles_env_cfg.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_env_cfg_static.py`

**Interfaces:**
- Produces: `parallelism_consecutive_standstill(env, threshold=2) -> torch.Tensor`.
- Produces: termination name `parallelism_consecutive_standstill`.

- [x] 添加 termination 单测：count 为 0/1 时返回 false，count 为 2 时返回 true，并保持 batch shape。
- [x] 在 `terminations.py` 中用 manager 的 `standstill_count` 做 Torch 条件判断；没有该属性时返回全 false。
- [x] 从 `tracking.mdp` 导出新 termination。
- [x] 在 flat termination config 中注册 `parallelism_consecutive_standstill`，阈值固定为 2。
- [x] small-obstacles 配置继承该 termination，不覆盖用户现有配置修改。
- [x] 更新静态测试确认 termination 已注册。
- [x] 运行 tracking MDP 和配置测试。

### Task 3: Episode diagnostics 和 play 面板

**Files:**
- Modify: `Go2Pvcnn/tracking/env.py`
- Modify: `Go2Pvcnn/scripts/play.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_policy_play_visualization.py`

**Interfaces:**
- Produces: `Episode_Termination/parallelism_consecutive_standstill`，由 Isaac Lab termination manager 自动记录。
- Produces: play 面板中的 `parallelism_consecutive_standstill` 开关和 raw diagnostic。

- [x] 将 termination 名称加入 play 面板的 termination 列表。
- [x] 在 play debug snapshot 中输出当前 `standstill_count`。
- [x] 验证 play 面板可以单独屏蔽该 termination，但仍保留 raw mask。
- [x] 运行 play visualization 静态/单元测试。

### Task 4: Full Verification and Commit

**Files:**
- No additional production files.

- [x] 运行：
  `PYTHONPATH=Go2Pvcnn python -m pytest Go2Pvcnn/tests/tracking Go2Pvcnn/tests/test_semantic_course_curriculum_layout.py Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py Go2Pvcnn/tests/test_train_script_static.py -q`
- [x] 运行 `git diff --check`。
- [x] 检查用户已有未提交修改仍然保留。
- [x] 提交：
  `git commit -m "feat: reset after consecutive parallelism standstill"`
