# Parallelism 蒸馏 Teacher/Critic 观测解耦实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 显式使用指定 teacher checkpoint，并将新 teacher 的 MLP 输入 1102 维与旧 student critic 的 MLP 输入 1099 维解耦，支持从 `model_5499.pt` 安全 resume。

**Architecture:** 蒸馏环境同时导出 `teacher` 和 `critic` 两组 privileged observation。teacher actor 使用包含 `velocity_commands` 的 1102 维输入；student critic 保持旧 1099 维输入和旧权重。resume 时只加载 student/student_critic，再加载显式 `--teacher_checkpoint`，跳过 student checkpoint 内嵌的旧 teacher。

**Tech Stack:** Python 3.10、PyTorch、Isaac Lab、RSL-RL、pytest、Bash。

## Global Constraints

- CNN 后 MLP 输入为 `student=1069`、`critic=1099`、`teacher=1102`；环境 flatten 输入对应 `student=557`、`critic=587`、`teacher=590`。
- 显式提供 `--teacher_checkpoint` 时，checkpoint 中的 `teacher.*` 永不加载。
- 不修改 Parallelism planner、reward、termination、curriculum 或 episode 接管逻辑。
- 保留 student actor、旧 student critic、student std 和 resume iteration。
- 普通 PPO 和非蒸馏路径保持兼容。
- 不覆盖工作区中与本任务无关的用户修改。

---

### Task 1: 先写失败测试

**Files:**
- Modify: `Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py`
- Create: `Go2Pvcnn/tests/tracking/test_parallelism_distillation_resume_static.py`

- [x] 添加静态断言：wrapper 导出 `critic`、模型支持 `num_critic_obs`、runner 有 `load_student_checkpoint`、resume 脚本有 `--teacher_checkpoint`。
- [x] 添加模型测试：确认 student critic 使用独立输入维度。
- [x] 添加 checkpoint 测试：student-only 加载后，student 与旧 critic 更新，嵌入 teacher 不更新。
- [x] 运行并确认失败：

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py Go2Pvcnn/tests/tracking/test_parallelism_distillation_resume_static.py
```

### Task 2: 拆分 teacher/critic 观测

**Files:**
- Modify: `Go2Pvcnn/tracking/parallelism_cross_large_complex_distillation_env_cfg.py`
- Modify: `Go2Pvcnn/scripts/train.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py`

- [x] teacher state 保留新 `velocity_commands`；新增旧版 critic map/state，不加入 command，保持 MLP 输入 1099 维。
- [ ] wrapper 返回：

```python
(student_obs, {"observations": {"teacher": teacher_obs, "critic": critic_obs}})
```

- [ ] Hybrid runner 初始化、每步更新和 bootstrap value 都分别读取 teacher/critic；普通 PPO/Distillation 路径保持原调用。

### Task 3: 保留旧 critic，增加 student-only 加载

**Files:**
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/modules/student_teacher_cnn.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/algorithms/hybrid_distillation_ppo.py`

- [x] `StudentTeacherCNN.__init__` 增加 `num_critic_obs=None`；student critic 使用它，缺省时回退到 `num_teacher_obs`。
- [x] 增加 `load_student_state_dict(state_dict, keep_std=True)`，只加载 `student.*` 和 `student_critic.*`，绝不传入 `teacher.*`。
- [x] 增加 `OnPolicyRunner.load_student_checkpoint(path, load_optimizer=True, keep_std=True)`，恢复 student、旧 critic、optimizer 和 iteration。
- [x] `HybridDistillationPPO.act(obs, teacher_obs, critic_obs)`：teacher action/imitation 使用 teacher，value/rollout 使用 critic。
- [x] 显式 teacher 继续通过现有 `load_teacher()` 单独加载。

### Task 4: 显式 teacher 覆盖 resume 中的旧 teacher

**Files:**
- Modify: `Go2Pvcnn/scripts/train.py`
- Modify: `Go2Pvcnn/scripts/train_parallelism_large_obstacles_rl_headless_distilation.sh`
- Modify: `Go2Pvcnn/scripts/train_parallelism_large_obstacles_rl_headless_distilation_resume.sh`

- [x] distillation resume 分支使用：

```python
if args_cli.teacher_checkpoint is not None:
    runner.load_student_checkpoint(checkpoint_path, keep_std=args_cli.keep_std)
    runner.load_teacher(args_cli.teacher_checkpoint, keep_std=True)
else:
    runner.load(checkpoint_path, keep_std=args_cli.keep_std)
```

- [x] 显式校验并打印 student checkpoint 和 teacher checkpoint 两条路径。
- [x] fresh launcher 指向 `.../parallelism_tracking_cross_large_complex/2026-08-18_20-30-59/6def073/model_4999.pt`。
- [x] resume launcher 指向 student `2026-08-18_21-38-36/6def073/model_5499.pt`，并增加上述 teacher path。

### Task 5: 验证与提交

- [x] 运行 focused tests 和 tracking 测试（`126 passed`）。
- [x] 执行 `py_compile` 和两个 distillation shell 的 `bash -n`。
- [x] 完成 1024 环境 4 iteration resume smoke test。
- [x] 检查 `git diff --check`，只暂存本任务文件和已审阅设计文档。
- [ ] 提交：`feat: decouple distillation teacher and critic observations`。
