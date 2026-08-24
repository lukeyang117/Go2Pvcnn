# M1 Parallelism Robot Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变 Go2 默认 Parallelism 行为的前提下，为轮式机器狗 M1 增加 12 个腿部关节的 FK/IK、碰撞几何、16 DOF Isaac viewer 资产切换和逐帧回放支持。

**Architecture:** 用机器人 backend registry 描述规划所需的 12 DOF 顺序、资产关节映射、FK/IK、限位、collision specs 与 viewer 元数据。通用 planner 接收可选 backend，未传时继续使用 Go2 backend；M1 wheel joint 只存在于 Isaac 资产回放层，规划和误差计算均不读取它。Viewer 根据 `--robot` 选择 backend、USD 配置和最小 viewer 环境，沿用现有 terrain/scanner/Parallelism 轨迹 schema。

**Tech Stack:** Python 3.10+, PyTorch, Isaac Lab/Isaac Sim, Gymnasium, pytest。

## Global Constraints

- `--robot` 默认值必须为 `go2`，默认 Go2 viewer 与既有命令等价。
- M1 planner 只处理 `FBL/FAR/RBL/RAR` 的 ABAD/HIP/KNEE 共 12 DOF；四个 `FOOT_JOINT` 不输入、不输出、不限位、不参与碰撞。
- M1 不创建 BASE collision primitive，不做 self-collision；沿用 `collision_margin_m=0.003`。
- flat/flatDense 保持 Root Z 固定；其它 terrain family 按 terrain column profile 跟随 Root，高度和姿态保持默认站姿附近。
- 第一阶段只做 Isaac Sim 运动学直接回放，不接 RL policy、轮速控制或正式物理动作控制。
- 不能修改现有 Go2 配置、Go2 FK/IK 公式和 Go2 环境默认行为；无关 dirty submodule 不得加入提交。

---

### Task 1: 建立机器人 backend 接口和 M1 静态模型

**Files:**
- Create: `Go2Pvcnn/extension/parallelism/robot_backend.py`
- Create: `Go2Pvcnn/extension/parallelism/m1_kinematics.py`
- Create: `Go2Pvcnn/extension/parallelism/m1_collision.py`
- Create: `Go2Pvcnn/go2_pvcnn/assets/m1.py`
- Modify: `Go2Pvcnn/extension/parallelism/__init__.py`
- Test: `Go2Pvcnn/tests/parallelism/test_m1_robot_backend.py`

**Interfaces:**
- `RobotBackend(name, planner_joint_names, asset_joint_names, wheel_joint_names, support_body_names, cfg, fk, ik, joint_limit_mask, collision_shapes)`。
- `get_robot_backend(name: str = "go2") -> RobotBackend`。
- `m1_fk(root_pos_w, root_rpy_w, joint_pos, *, capsule_samples=5) -> M1ParallelGeometry`。
- `m1_ik(root_pos_w, root_rpy_w, foot_target_w) -> tuple[Tensor, Tensor]`。
- `build_m1_surface_points_l(...)` 返回 16 shape 的 padded points 和 mask；box 26 点，wheel 38 点，总有效点 464。

- [ ] **Step 1: Write the failing tests**

```python
def test_m1_backend_exposes_12_planner_joints_and_16_asset_joints():
    backend = get_robot_backend("m1")
    assert len(backend.planner_joint_names) == 12
    assert len(backend.asset_joint_names) == 16
    assert backend.wheel_joint_names == (
        "FBL_FOOT_JOINT", "FAR_FOOT_JOINT", "RBL_FOOT_JOINT", "RAR_FOOT_JOINT"
    )

def test_m1_fk_neutral_pose_has_symmetric_wheel_centers():
    backend = get_robot_backend("m1")
    geometry = backend.fk(torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 12))
    expected = torch.tensor([[[.3295, .2074, -.5398], [.3295, -.2074, -.5398],
                              [-.3295, .2074, -.5398], [-.3295, -.2074, -.5398]]])
    torch.testing.assert_close(geometry.foot_pos_w, expected, atol=2e-4, rtol=0)

def test_m1_wheel_joint_is_not_in_collision_specs():
    backend = get_robot_backend("m1")
    assert len(backend.collision_shapes) == 16
    assert sum(spec.shape_type == "cylinder" for spec in backend.collision_shapes) == 4
    assert not any(spec.link_type == "base" for spec in backend.collision_shapes)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd Go2Pvcnn && pytest -q tests/parallelism/test_m1_robot_backend.py`

Expected: FAIL because the M1 backend modules and registry do not exist.

- [ ] **Step 3: Write the minimal implementation**

Keep `fk_go2` and `ik_go2` untouched. Add M1's explicit symmetric hip offsets, X-axis ABAD followed by Y-axis HIP/KNEE planar chains, neutral root height `0.6358`, limits `ABAD left [-0.523, 0.697]`, `ABAD right [-0.697, 0.523]`, `HIP [-2.443, 2.443]`, `KNEE [-2.801, 2.801]`, and the four wheel joint names. Register Go2 as the default backend wrapper and M1 as the new backend. Implement collision surface generation with 4 ABAD boxes, 4 HIP boxes, 4 KNEE boxes and 4 wheel cylinders, with no BASE spec.

- [ ] **Step 4: Run the focused tests**

Run: `cd Go2Pvcnn && pytest -q tests/parallelism/test_m1_robot_backend.py`; Expected: PASS with the 12/16 mapping, neutral symmetry, limits, and 464 valid collision points.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/parallelism Go2Pvcnn/go2_pvcnn/assets/m1.py Go2Pvcnn/tests/parallelism/test_m1_robot_backend.py
git commit -m "feat: add M1 parallelism robot backend"
```

### Task 2: 让通用 Parallelism planner 使用可选 backend

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/planner.py`
- Modify: `Go2Pvcnn/extension/parallelism/collision.py`
- Modify: `Go2Pvcnn/extension/parallelism/viewer_adapter.py`
- Modify: `Go2Pvcnn/extension/parallelism/__init__.py`
- Test: `Go2Pvcnn/tests/parallelism/test_m1_planner_backend.py`

**Interfaces:**
- `plan_trajectory(..., robot_backend: RobotBackend | None = None)`；`None` 等价于 Go2。
- `parallelism_trajectory_to_viewer_result(trajectory, robot_backend: RobotBackend | None = None)`。

- [ ] **Step 1: Write the failing tests**

```python
def test_go2_planner_default_backend_remains_go2():
    assert inspect.signature(plan_trajectory).parameters["robot_backend"].default is None
    assert get_robot_backend().name == "go2"

def test_m1_planner_uses_m1_shape_names_and_12_dof_output():
    result = plan_trajectory(make_neutral_m1_state(), torch.zeros(1, 3), make_flat_terrain(),
                             M1_CFG, robot_backend=get_robot_backend("m1"))
    assert result.joint_pos.shape[-1] == 12
    assert result.diagnostics.collision_shape_names[0].startswith("FBL_")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd Go2Pvcnn && pytest -q tests/parallelism/test_m1_planner_backend.py`; Expected: FAIL because planner and viewer adapter still call Go2 FK/IK and Go2 collision specs directly.

- [ ] **Step 3: Implement the minimal backend threading**

Use `backend = robot_backend or get_robot_backend("go2")` at the public entry point and pass it through candidate FK/IK, current foot fallback, joint limits, swing collision, touchdown collision, and trajectory assembly. Update collision helpers to accept `backend.collision_shapes` and use backend geometry link fields. Preserve existing calls by adding only optional arguments.

- [ ] **Step 4: Run focused regression**

Run: `cd Go2Pvcnn && pytest -q tests/parallelism/test_m1_planner_backend.py tests/parallelism/test_m1_robot_backend.py tests/tracking/test_parallelism_plan_valid.py`; Expected: PASS with unchanged Go2 static planner tests and new M1 planner tests.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/parallelism Go2Pvcnn/tests/parallelism/test_m1_planner_backend.py
git commit -m "feat: make parallelism planner robot backend aware"
```

### Task 3: 添加 M1 Isaac Lab asset 和 viewer 环境配置

**Files:**
- Create: `Go2Pvcnn/tracking/m1_parallelism_viewer_env_cfg.py`
- Modify: `Go2Pvcnn/go2_pvcnn/assets/m1.py`
- Test: `Go2Pvcnn/tests/tracking/test_m1_viewer_env_cfg_static.py`

**Interfaces:**
- `M1_CFG` 是 16 DOF 的 `ArticulationCfg`，USD 路径固定为 `m1/ZJ_V3_URDF_V1_0/ZJ_V3_URDF_V1_0.usd`。
- `M1ParallelismViewerEnvCfg` 提供 terrain、semantic scanner、reset、command 和 16 维 zero-action 所需的最小 viewer 配置；不实例化 Go2 reference robot 或 tracking reward/termination/curriculum。

- [ ] **Step 1: Write the failing static test**

```python
def test_m1_asset_and_viewer_cfg_use_m1_names():
    asset = (ROOT / "go2_pvcnn/assets/m1.py").read_text()
    env = (ROOT / "tracking/m1_parallelism_viewer_env_cfg.py").read_text()
    assert "ZJ_V3_URDF_V1_0.usd" in asset
    assert "BASE_LINK" in env and "FBL_FOOT_LINK" in env
    assert "reference_robot" not in env
```

- [ ] **Step 2: Run the static test to verify it fails**

Run: `cd Go2Pvcnn && pytest -q tests/tracking/test_m1_viewer_env_cfg_static.py`; Expected: FAIL because the M1 asset and environment config do not exist.

- [ ] **Step 3: Implement the M1 asset/config**

Clone only the existing terrain/scanner parameters into a dedicated viewer config, replace robot prim with `{ENV_REGEX_NS}/Robot`, scanner attachment with `/Robot/BASE_LINK`, explicit support bodies `FBL_FOOT_LINK/FAR_FOOT_LINK/RBL_FOOT_LINK/RAR_FOOT_LINK`, and action dimension 16. Disable reference robot and tracking managers. Set the initial 16 joint positions to zero; playback preserves measured wheel values.

- [ ] **Step 4: Run static tests**

Run: `cd Go2Pvcnn && pytest -q tests/tracking/test_m1_viewer_env_cfg_static.py`; Expected: PASS without importing Isaac Sim in the static test process.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/assets/m1.py Go2Pvcnn/tracking/m1_parallelism_viewer_env_cfg.py Go2Pvcnn/tests/tracking/test_m1_viewer_env_cfg_static.py
git commit -m "feat: add M1 Isaac viewer configuration"
```

### Task 4: 接入 viewer CLI、关节映射、碰撞显示和有限周期 smoke

**Files:**
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Test: `Go2Pvcnn/tests/test_m1_viewer_backend_static.py`

**Interfaces:**
- CLI: `--robot {go2,m1}` default `go2`; `--max-plan-cycles N` default `0` means interactive/unbounded。
- `merge_planner_joints_into_robot(robot_joint_pos, planner_joint_pos, backend) -> Tensor`。
- `_build_env_cfg(args_cli)` and `gym.make` select the same robot-specific asset/config/task path。

- [ ] **Step 1: Write failing parser/mapping tests**

```python
def test_viewer_defaults_to_go2_and_accepts_m1():
    parser = viewer.build_arg_parser()
    assert parser.parse_args(["--robot", "m1"]).robot == "m1"
    assert parser.parse_args([]).robot == "go2"
    assert parser.parse_args([]).max_plan_cycles == 0

def test_m1_playback_overwrites_only_12_leg_joints():
    current = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    merged = viewer.merge_planner_joints_into_robot(current, torch.full((1, 12), 9.0), get_robot_backend("m1"))
    torch.testing.assert_close(merged[:, [3, 7, 11, 15]], current[:, [3, 7, 11, 15]])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_viewer_backend_static.py`; Expected: FAIL because parser, mapping, and M1 viewer selection are not implemented.

- [ ] **Step 3: Implement selection and playback**

Add parser arguments, normalize/validate the robot name, select M1 `M1ParallelismViewerEnvCfg` and `M1_CFG`, find support bodies with explicit names, and retain Go2's `.*_foot` lookup by default. Modify direct playback to clone the current robot joint tensor and scatter only planner joint indices; wheel indices are never written. Pass backend to state/config/plan/result helpers. Make mesh visibility robot-neutral and display backend collision centers/points.

- [ ] **Step 4: Add bounded exit and run regression**

Increment `plan_cycle` after successful planning and break after `max_plan_cycles`; close the environment through the existing cleanup path. Run: `cd Go2Pvcnn && pytest -q tests/test_m1_viewer_backend_static.py tests/test_viewer_reset.py tests/parallelism tests/tracking/test_parallelism_tracking_env_cfg_static.py`; Expected: PASS and existing Go2 reset tests remain green.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_m1_viewer_backend_static.py
git commit -m "feat: add M1 parallelism viewer selection and playback"
```

### Task 5: Isaac Sim headless smoke and final regression

**Files:**
- Create: `Go2Pvcnn/tests/isaac_m1_viewer_smoke.py`
- Modify: `Go2Pvcnn/README.md`

- [ ] **Step 1: Add a bounded smoke probe**

Verify registry/asset path statically, and when Isaac Sim is available launch one M1 parallelism plan with `--robot m1 --planner-backend parallelism --terrain-col 1 --scripted-command "0.1 0 0" --scripted-command-cycles 1 --max-plan-cycles 1 --headless`; assert a 16-column robot tensor, 12-column planned trajectory, and no BASE shape.

- [ ] **Step 2: Run final checks**

Run: `cd Go2Pvcnn && pytest -q tests/parallelism tests/tracking/test_m1_viewer_env_cfg_static.py tests/test_m1_viewer_backend_static.py`, then run the bounded Isaac Sim command using the machine's actual Isaac launcher. If launcher/`pxr` is unavailable, record the exact blocker rather than claiming runtime success.

- [ ] **Step 3: Document usage and commit**

Document `python extension/viz/go2_foostep_planner.py --planner-backend parallelism` and `python extension/viz/go2_foostep_planner.py --robot m1 --planner-backend parallelism --terrain-col 1` in the README, then:

```bash
git add Go2Pvcnn/tests/isaac_m1_viewer_smoke.py Go2Pvcnn/README.md
git commit -m "test: add bounded M1 Isaac viewer smoke"
```

## Self-Review Checklist

- [ ] Go2 default path is covered by existing tests and an explicit default-backend assertion.
- [ ] M1 wheel preservation, 16 shape/464 point collision geometry, no-BASE boundary, terrain-column selection, and bounded viewer exit each have a test or smoke assertion.
- [ ] No task modifies RL training behavior or unrelated dirty submodules.
