# M1 Mixed Leg-Position Wheel-Velocity Action Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 `m1_rl` 分支中保持 16 维 M1 PPO/AME 接口不变，将 12 个腿部关节改为位置目标控制、4 个轮子关节改为速度目标控制，并完成 1 环境和 1024 环境 IsaacLab smoke test。

**Architecture:** 新增一个单一的 M1 mixed action term，输入仍按 `M1_USD_JOINT_NAMES` 保持 16 维和原有顺序；该 term 内部按腿/轮子索引分别生成位置目标和速度目标。M1 环境配置只替换 action term，Go2 和 parallelism 代码不修改。第一阶段不改地形、课程、PPO/AME 网络和 reward 权重，先用运行指标验证动作接口是否修复了轮子绝对位置控制问题。

**Tech Stack:** Python 3.10, Isaac Lab, Isaac Sim, PyTorch, Gymnasium, pytest, rsl_rl AME PPO。

## Global Constraints

- 只修改 `/share/home/tm884089579940000/a915071960/lhy/kinematic_m1/Go2Pvcnn_m1_rl` worktree 的 `m1_rl` 分支。
- 主 `m1` 工作树、Go2 分支和 parallelism 执行路径不得修改。
- Policy action 必须保持 16 维，顺序必须保持 `M1_USD_JOINT_NAMES`：每条腿的 ABAD/HIP/KNEE/FOOT。
- 腿部 action 使用位置目标，scale 为 `0.25`，默认关节位置为 offset。
- 轮子 action 使用速度目标，默认速度 offset 为零，初始速度 scale 固定为 `8.0 rad/s`，并在动作 term 内限幅到 `±8.0 rad/s`。
- 本阶段不修改地形生成、terrain curriculum、PPO/AME 算法、观测维度和 Go2 配置。
- 任何训练必须从新的 policy 初始化开始，不得从旧的 M1 不稳定 checkpoint resume。
- 1 环境测试至少验证动作目标写入、环境构造、若干步仿真和有限数值；1024 环境测试至少验证环境构造、一次 rollout、动作维度和有限数值。

---

### Task 1: 为 mixed action 建立失败测试

**Files:**
- Create: `Go2Pvcnn/tests/m1_ame/test_m1_mixed_actions.py`
- Modify: none

**Interfaces:**
- Test imports `M1MixedJointActionCfg`, `M1MixedJointAction`, and `split_m1_mixed_action` from `m1_ame.m1_mixed_actions`.
- `split_m1_mixed_action(actions, leg_indices, wheel_indices, leg_scale, wheel_scale, default_leg_pos)` returns `(position_targets, velocity_targets)` with shapes `(N, 12)` and `(N, 4)`.

- [ ] **Step 1: Write the failing test**

```python
import torch

from go2_pvcnn.assets.m1 import M1_USD_JOINT_NAMES
from m1_ame.m1_mixed_actions import M1MixedJointActionCfg, split_m1_mixed_action


def test_m1_mixed_action_preserves_usd_order_and_separates_control_modes():
    actions = torch.zeros((2, 16))
    actions[:, 0] = 1.0
    actions[:, 3] = 0.5
    actions[:, 15] = -0.25
    leg_indices = tuple(i for i, name in enumerate(M1_USD_JOINT_NAMES) if not name.endswith("FOOT_JOINT"))
    wheel_indices = tuple(i for i, name in enumerate(M1_USD_JOINT_NAMES) if name.endswith("FOOT_JOINT"))
    default_leg_pos = torch.arange(12, dtype=torch.float32).view(1, 12).repeat(2, 1)

    position_targets, velocity_targets = split_m1_mixed_action(
        actions, leg_indices, wheel_indices, 0.25, 8.0, default_leg_pos
    )

    assert position_targets.shape == (2, 12)
    assert velocity_targets.shape == (2, 4)
    assert torch.allclose(position_targets[:, 0], default_leg_pos[:, 0] + 0.25)
    assert torch.allclose(velocity_targets[:, 0], torch.full((2,), 4.0))
    assert torch.allclose(velocity_targets[:, 3], torch.full((2,), -2.0))


def test_m1_mixed_action_config_is_single_sixteen_dimensional_term():
    cfg = M1MixedJointActionCfg(
        asset_name="robot",
        joint_names=list(M1_USD_JOINT_NAMES),
        leg_joint_names=[name for name in M1_USD_JOINT_NAMES if not name.endswith("FOOT_JOINT")],
        wheel_joint_names=[name for name in M1_USD_JOINT_NAMES if name.endswith("FOOT_JOINT")],
    )
    assert cfg.class_type is not None
    assert len(cfg.joint_names) == 16
    assert cfg.leg_scale == 0.25
    assert cfg.wheel_scale == 8.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic_m1/Go2Pvcnn_m1_rl
PYTHONPATH="$PWD/Go2Pvcnn" pytest -q Go2Pvcnn/tests/m1_ame/test_m1_mixed_actions.py
```

Expected: FAIL because `m1_ame.m1_mixed_actions` does not exist.

---

### Task 2: 实现保持 16 维顺序的 mixed action term

**Files:**
- Create: `Go2Pvcnn/m1_ame/m1_mixed_actions.py`
- Modify: none
- Test: `Go2Pvcnn/tests/m1_ame/test_m1_mixed_actions.py`

**Interfaces:**
- `split_m1_mixed_action(...)` performs pure tensor conversion and is used by the action term.
- `M1MixedJointActionCfg` extends `ActionTermCfg` and contains `joint_names`, `leg_joint_names`, `wheel_joint_names`, `leg_scale=0.25`, `wheel_scale=8.0`, `wheel_velocity_limit=8.0`.
- `M1MixedJointAction` extends `ActionTerm`, reports `action_dim == 16`, stores raw/processed actions, and in `apply_actions()` calls `set_joint_position_target()` only for leg IDs and `set_joint_velocity_target()` only for wheel IDs.

- [ ] **Step 1: Write minimal implementation**

The term must resolve all IDs with `preserve_order=True`, build action-order indices from the configured full list, and reject duplicate or missing leg/wheel names. It must use the articulation default positions indexed by leg IDs as the leg offset. Wheel velocity targets must be clamped to `[-wheel_velocity_limit, wheel_velocity_limit]`. `reset()` must zero raw and processed action buffers.

- [ ] **Step 2: Run the focused tests**

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic_m1/Go2Pvcnn_m1_rl
PYTHONPATH="$PWD/Go2Pvcnn" pytest -q Go2Pvcnn/tests/m1_ame/test_m1_mixed_actions.py
```

Expected: PASS.

- [ ] **Step 3: Commit the isolated action implementation**

```bash
git add Go2Pvcnn/m1_ame/m1_mixed_actions.py Go2Pvcnn/tests/m1_ame/test_m1_mixed_actions.py
git commit -m "feat: add M1 mixed leg position wheel velocity action"
```

---

### Task 3: 将 M1 AME 环境切换到 mixed action

**Files:**
- Modify: `Go2Pvcnn/m1_ame/m1_ame_env_cfg.py:75-95`
- Modify: `Go2Pvcnn/tests/m1_ame/test_m1_ame_contract.py`
- Test: `Go2Pvcnn/tests/m1_ame/test_m1_ame_contract.py`

**Interfaces:**
- `M1AmeActionsCfg` contains exactly one `M1MixedJointActionCfg` term with all 16 M1 USD joints in their current order.
- Existing observation selectors remain all 16 joints; no policy/critic dimension changes are allowed.

- [ ] **Step 1: Add the failing contract assertions**

Assert the M1 environment source imports `M1MixedJointActionCfg`, instantiates it with the full USD joint list, declares both leg and wheel selectors, and does not use `JointPositionActionCfg` for the M1 action.

- [ ] **Step 2: Run the contract test to verify it fails**

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic_m1/Go2Pvcnn_m1_rl
PYTHONPATH="$PWD/Go2Pvcnn" pytest -q Go2Pvcnn/tests/m1_ame/test_m1_ame_contract.py
```

Expected: FAIL on the new mixed-action assertions.

- [ ] **Step 3: Replace the M1 action config**

Use the existing `M1_LEG_JOINT_NAMES` and `M1_WHEEL_JOINT_NAMES` selectors, import the new config, and define one mixed term with `leg_scale=0.25`, `wheel_scale=8.0`, and `wheel_velocity_limit=8.0`. Keep the action attribute name stable enough for serialized env configs, and remove the old all-position term.

- [ ] **Step 4: Run the contract and package tests**

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic_m1/Go2Pvcnn_m1_rl
PYTHONPATH="$PWD/Go2Pvcnn" pytest -q Go2Pvcnn/tests/m1_ame
```

Expected: all existing M1 AME tests plus the new assertions pass.

- [ ] **Step 5: Commit the environment wiring**

```bash
git add Go2Pvcnn/m1_ame/m1_ame_env_cfg.py Go2Pvcnn/tests/m1_ame/test_m1_ame_contract.py
git commit -m "feat: wire M1 AME mixed wheel velocity action"
```

---

### Task 4: 添加 IsaacLab 运行时动作 smoke probe

**Files:**
- Create: `Go2Pvcnn/tests/m1_ame/m1_mixed_action_runtime_probe.py`
- Create: `Go2Pvcnn/tests/m1_ame/test_m1_mixed_action_runtime_static.py`
- Modify: none

**Interfaces:**
- Probe accepts `--num_envs` and `--steps`, launches IsaacLab through `AppLauncher`, registers the M1 AME environment exactly as the training script does, and prints action manager term/dimension, finite checks, and the mean absolute wheel velocity target after a nonzero wheel action.
- Probe exits nonzero on environment construction failure, wrong action dimension, missing wheel velocity target, nonfinite tensors, or unexpected position targets on wheel joints.

- [ ] **Step 1: Write the static probe contract test**

Check that the probe uses `AppLauncher`, `num_envs`, `gym.make`, `M1AmeCrossLargeComplexEnvCfg`, `action_manager`, `joint_vel_target`, and a finite check. This test must fail before the probe exists.

- [ ] **Step 2: Run the static test to verify it fails**

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic_m1/Go2Pvcnn_m1_rl
PYTHONPATH="$PWD/Go2Pvcnn" pytest -q Go2Pvcnn/tests/m1_ame/test_m1_mixed_action_runtime_static.py
```

- [ ] **Step 3: Implement the probe**

For each step, use a deterministic action tensor with zero leg actions and a positive wheel action in the four wheel slots, call `env.step()`, inspect the articulation target buffers, and assert that leg position targets stay at the default pose while wheel velocity targets reach the configured limit. The probe must close the environment and simulation app in `finally`.

- [ ] **Step 4: Run the static test**

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic_m1/Go2Pvcnn_m1_rl
PYTHONPATH="$PWD/Go2Pvcnn" pytest -q Go2Pvcnn/tests/m1_ame/test_m1_mixed_action_runtime_static.py
```

Expected: PASS.

---

### Task 5: 执行 1 环境和 1024 环境 IsaacLab smoke test

**Files:**
- Modify: none
- Test artifacts: `Go2Pvcnn/tmp/m1_mixed_action_smoke/one_env.log`, `Go2Pvcnn/tmp/m1_mixed_action_smoke/1024_env.log`

- [ ] **Step 1: Run the 1-environment probe**

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic_m1/Go2Pvcnn_m1_rl
export DISPLAY=:1
export OMNI_KIT_ACCEPT_EULA=Y
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/share/home/tm884089579940000/a915071960/lhy/kinematic_m1/Go2Pvcnn_m1_rl/Go2Pvcnn
export ISAACLAB_PATH=/share/home/tm884089579940000/a915071960/lhy/IsaacLab
mkdir -p Go2Pvcnn/tmp/m1_mixed_action_smoke
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/m1_ame/m1_mixed_action_runtime_probe.py \
  --device cuda:0 --num_envs 1 --steps 32 \
  > Go2Pvcnn/tmp/m1_mixed_action_smoke/one_env.log 2>&1
```

Expected: exit code 0, `action_dim=16`, finite observations/rewards, nonzero wheel velocity target, no wheel position target write.

- [ ] **Step 2: Run the 1024-environment probe**

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/m1_ame/m1_mixed_action_runtime_probe.py \
  --device cuda:0 --num_envs 1024 --steps 8 \
  > Go2Pvcnn/tmp/m1_mixed_action_smoke/1024_env.log 2>&1
```

Expected: exit code 0, action/observation dimensions unchanged, all checked tensors finite, no IsaacLab environment construction error.

- [ ] **Step 3: Run the complete focused regression tests**

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic_m1/Go2Pvcnn_m1_rl
PYTHONPATH="$PWD/Go2Pvcnn" pytest -q Go2Pvcnn/tests/m1_ame Go2Pvcnn/tests/parallelism/test_m1_robot_backend.py
```

Expected: no new failures attributable to mixed action. Existing unrelated failures must be reported separately.

---

### Task 6: 提交并推送 m1_rl

**Files:**
- Modify: none beyond Tasks 1–5

- [ ] **Step 1: Review diff and repository state**

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic_m1/Go2Pvcnn_m1_rl
git diff --check
git status --short
git diff --stat origin/m1_rl...HEAD
```

- [ ] **Step 2: Commit smoke-test and regression artifacts**

```bash
git add Go2Pvcnn/m1_ame Go2Pvcnn/tests/m1_ame docs/superpowers/plans/2026-10-19-m1-mixed-wheel-action.md
git commit -m "test: validate M1 mixed wheel action in IsaacLab"
```

- [ ] **Step 3: Push the completed branch**

```bash
git push origin m1_rl
```

- [ ] **Step 4: Verify local and remote commit IDs**

```bash
git rev-parse HEAD
git ls-remote origin refs/heads/m1_rl
```

Expected: both SHA values are identical.
