# Human：batched planner 中 swing / stance、foothold、base 与 IK（语义、复杂度与执行地图）

## 导航

- 文档类型：`human` 专题笔记（复杂度与代码对照）
- 对应 AI 文档：[../ai/ai-13-batched-planner-swing-stance-ik-complexity.md](../ai/ai-13-batched-planner-swing-stance-ik-complexity.md)
- 上游 runtime 语境：[human-10-extension-planner-runtime.md](human-10-extension-planner-runtime.md)
- 总索引：[../index.md](../index.md)

## 约定符号

- **N**：并行环境个数，即 `batched_generate_trajectory` 里与 `states` / `commands` 对齐的 **batch 维** `batch_size`（与 Isaac 里 `num_envs` 一致）。

```135:137:Go2Pvcnn/extension/batched_planner/trajectory.py
    batch_size = int(states.root_pos.shape[0])
    if commands_t.shape[0] != batch_size:
        raise ValueError("states and commands must share batch size")
```

- **T**：一次规划结果的时间长度 `n_frames`（`min(requested_n_frames, cycle_frames)`，见 `trajectory.py:139-141`）。
- **K**：foothold 螺旋搜索里 **单腿单环境** 的平面候选格点数，由 `_precompute_spiral_offsets(search_radius, search_step)` 决定，与 `N`、`T` 独立（`foothold.py:15-31` 与 `121-163`）。
- **S**：机身净空采样点数，`BODY_COLLISION_SAMPLES.shape[0]`，常数（`base_solver.py:11-23`）。
- **腿数**：固定为 4。

**本文只写时间复杂度**（随问题规模增长的阶）。约定：

- 若代码路径中存在 **按环境或按 batch 维顺序执行的 Python 串行**（含 **`for` 遍历 `batch_size` / `.item()` 按 env 取标量**），则在该段阶中 **显式写出 N**。
- 若仅有 **张量批算、且无对 N 的 Python 串行环**，则阶中 **不写 N**（对 N 视为与渐近无关；可保留 **T**、**K** 等与 horizon / 搜索离散度相关的参数）。

---

## 1. stance / swing 各自代表什么

### 1.1 接触序列 `contact_seq`：stance = 着地，swing = 腾空

`batched_gait_schedule` 根据相位与 `duty_factor` 生成布尔接触序列：`phase < duty_factor` 的帧记为接触（stance），否则为不接触（swing）。

```173:176:Go2Pvcnn/extension/batched_planner/gait.py
    frame_index = torch.arange(int(n_frames), dtype=torch.float64, device=t0_b.device)
    t = t0_b[:, None] + frame_index[None, :] * float(dt)
    phase = torch.remainder(t[:, :, None] * step_freq_b[:, None, None] + phase_offsets_b[:, None, :], 1.0)
    return (phase < duty_factor_b[:, None, None]).to(torch.float32)
```

主入口里该序列形状为 `(N, T, 4)`，直接作为 `contact_state` 写入结果，并在 swing 与 base 求解等处复用：

```147:155:Go2Pvcnn/extension/batched_planner/trajectory.py
    contact_seq = batched_gait_schedule(
        torch.zeros(batch_size, dtype=torch.float64, device=device),
        n_frames,
        dt,
        torch.full((batch_size,), float(cfg.step_freq), dtype=torch.float64, device=device),
        torch.full((batch_size,), float(cfg.duty_factor), dtype=torch.float64, device=device),
        phase_offsets,
    )
```

语义：**1 表示该腿该帧处于支撑（stance），0 表示摆动（swing）**。

### 1.2 `stance_time`（时间秒）与 swing 几何是两条线

`batched_stance_time` 返回的是 **每条步态周期内支撑相持续时间**（秒），用于足端候选搜索等「运动学尺度」，并不在 `swing.py` 里参与插值：

```206:228:Go2Pvcnn/extension/batched_planner/gait.py
def batched_stance_time(step_freq, duty_factor) -> Tensor:
    """Return stance duration with shape (N,).

    Scalar inputs broadcast to a single batch item.
    Mixed-device tensor inputs raise ``ValueError``.
    """
    device = _resolve_input_device(step_freq, duty_factor)
    step_freq_b = _as_batch_vector(step_freq, name="step_freq", device=device)
    duty_factor_b = _as_batch_vector(duty_factor, name="duty_factor", device=device)
    ...
    return duty_factor_b / step_freq_b
```

在 `trajectory.py` 中它传入 `batched_compute_footholds(..., stance_time=st, ...)`，与 `contact_seq` 并行使用。

### 1.3 足端世界系目标：`swing.py` 如何把 stance / swing 合成 `foot_targets`

`batched_compute_swing_targets` 对每个 batch、每条腿：

1. 用 `contact_seq > 0.5` 得到逐帧 `stance` 布尔序列。
2. `_leg_swing_progress_and_stance_anchor(stance)` 计算：
   - `swing_progress`：在本段连续 swing 区间内的归一化进度 \([0,1]\)；
   - `use_touchdown`：当前 stance 帧是否应以 **touchdown** 世界坐标为锚（由抬脚/落地事件计数判定）。
3. `_swing_phase_targets` 在 swing 段生成 **抬脚点 → 顶点 → 落点** 的两段 Hermite 竖直曲线 + XY 线性插值。
4. `torch.where(stance, anchor, arc)`：**stance 用锚点（lift-off 或 touchdown），swing 用弧线**。

```148:164:Go2Pvcnn/extension/batched_planner/swing.py
    out = torch.zeros((batch_size, num_frames, 4, 3), dtype=torch.float64, device=device)
    for batch_idx in range(batch_size):
        for leg in range(4):
            lo = lift_off_pos_t[batch_idx, leg]
            td = touchdown_pos_t[batch_idx, leg]
            apex = _compute_swing_apex(lo, td, float(step_height), terrain_h[batch_idx, leg], clearance=float(clearance))
            stance = contact_seq_t[batch_idx, :, leg] > 0.5
            swing_prog, use_td = _leg_swing_progress_and_stance_anchor(stance)
            arc = _swing_phase_targets(
                swing_prog,
                lo.expand(num_frames, -1),
                td.expand(num_frames, -1),
                apex.expand(num_frames),
            )
            anchor = torch.where(use_td[:, None], td.expand(num_frames, -1), lo.expand(num_frames, -1))
            out[batch_idx, :, leg, :] = torch.where(stance[:, None], anchor, arc)
    return out
```

**要点**：swing 段的形状由 Hermite 控制（竖直两段、XY 直线），顶点高度受 `step_height` 与 `terrain_max_heights` 影响（`_compute_swing_apex`）。

---

## 2. swing / stance 路径的时间复杂度（显式含 N）

记 `batched_compute_swing_targets` 的时间维为 T、腿数为 4。实现中对 **batch 维显式 Python 循环**：

```149:164:Go2Pvcnn/extension/batched_planner/swing.py
    for batch_idx in range(batch_size):
        for leg in range(4):
            lo = lift_off_pos_t[batch_idx, leg]
            td = touchdown_pos_t[batch_idx, leg]
            apex = _compute_swing_apex(lo, td, float(step_height), terrain_h[batch_idx, leg], clearance=float(clearance))
            stance = contact_seq_t[batch_idx, :, leg] > 0.5
            swing_prog, use_td = _leg_swing_progress_and_stance_anchor(stance)
            arc = _swing_phase_targets(
                swing_prog,
                lo.expand(num_frames, -1),
                td.expand(num_frames, -1),
                apex.expand(num_frames),
            )
            anchor = torch.where(use_td[:, None], td.expand(num_frames, -1), lo.expand(num_frames, -1))
            out[batch_idx, :, leg, :] = torch.where(stance[:, None], anchor, arc)
    return out
```

### 2.1 时间复杂度（只看串行控制流）

`batched_compute_swing_targets`（`swing.py:126` 起）：

- 外层 **`for batch_idx in range(batch_size)`**、内层 **`for leg in range(4)`**（`swing.py:149-150`）→ 对 **N** 与腿数串行（腿数为常数）。
- `_leg_swing_progress_and_stance_anchor` 内两段 **`for i in range(n)`**（`n = T`）及 **`.item()`**（`swing.py:104-115`）→ 对每个 `(batch_idx, leg)` 再 **O(T)**。

故本函数时间复杂度：**O(N·T)**。

**N = 1** 时：**O(T)**（仍含锚点内两段长度 T 的 Python 环）。

### 2.2 实现层面的重要细节（常数很大）

`_leg_swing_progress_and_stance_anchor` 在计算 swing 区间长度时使用了 **Python `for i in range(n)` 循环**，并在循环内调用 `.item()` 把 GPU 标量同步到 CPU：

```102:118:Go2Pvcnn/extension/batched_planner/swing.py
    starts = torch.full((n,), -1, dtype=torch.int64, device=stance.device)
    start_idx = -1
    for i in range(n):
        if bool(swing_starts[i].item()):
            start_idx = int(idxs[i].item())
        starts[i] = start_idx
    ...
    for i in range(n - 1, -1, -1):
        if int(lengths[i].item()) > 0:
            last_len = int(lengths[i].item())
        run_len[i] = last_len
```

含义：除阶上 **O(T)** 外，**.item()** 还可能带来额外同步开销（不计入渐近阶，但影响实测）。

---

## 3. IK 的时间复杂度（无对 N 的 Python 串行）

### 3.1 调用形态

一次规划末尾把 `(N, T, …)` 展平为首维 **`N·T`** 再调 IK / FK（`ik.py` 里首维变量名写作 `n_frames`，实为 **样本条数 = N·T**）：

```250:254:Go2Pvcnn/extension/batched_planner/trajectory.py
    root_pos_flat = root_pos.reshape(batch_size * n_frames, 3)
    root_quat_flat = root_quat.reshape(batch_size * n_frames, 4)
    foot_targets_flat = foot_targets.reshape(batch_size * n_frames, 4, 3)
    joint_angles = batch_inverse_kinematics(root_pos_flat, root_quat_flat, foot_targets_flat).reshape(batch_size, n_frames, 12)
    body_pos_w = batch_forward_kinematics(root_pos_flat, root_quat_flat, joint_angles.reshape(batch_size * n_frames, 12)).reshape(batch_size, n_frames, 12, 3)
```

`_standstill_trajectory` 中同样使用 `batch_size * n_frames` 展平（`trajectory.py:74-77`）。

### 3.2 `batch_inverse_kinematics` 内部在做什么

主流程：沿 **展平后的每一行**（共 **N·T** 行）做四元数 → 旋转矩阵 → 足端机体坐标 → 每腿闭式 IK → `clamp`。

```166:185:Go2Pvcnn/extension/batched_planner/ik.py
def batch_inverse_kinematics(root_pos, root_quat, foot_targets) -> Tensor:
    ...
    n_frames = int(root_pos_t.shape[0])
    ...
    rot = _quat_to_rot_batch(root_quat_t)
    foot_body = torch.einsum("nji,nmj->nmi", rot, foot_targets_t - root_pos_t[:, None, :])
    ...
    angles = _solve_leg_ik_batch(foot_body - hip_offsets, side_signs)
    joints = angles.reshape(n_frames, 12)
    ...
    return torch.clamp(joints, min=lower, max=upper)
```

`_solve_leg_ik_batch` 对形状 `(..., 4, 3)` 的足端相对 hip 位置做逐元素闭式解；腿数 4 为常数：

```92:120:Go2Pvcnn/extension/batched_planner/ik.py
def _solve_leg_ik_batch(
    foot_pos_hip: Tensor,
    side_sign: Tensor,
    ...
) -> Tensor:
    px = foot_pos_hip[..., 0]
    py = foot_pos_hip[..., 1]
    pz = foot_pos_hip[..., 2]
    ...
    return torch.stack([hip_angle, thigh_angle, calf_angle], dim=-1)
```

### 3.3 时间复杂度

## 8. T302 MPC 碰撞与语义安全增量

T302 只改 active `extension/batch_mpc_planner` MPC 后端，不回退到旧 raw/legacy planner。

新增行为：

- `kinematics.py` 的 MPC FK 现在可返回足端、knee world、shank 采样点 world，用于 loss 侧碰撞检测。
- `terrain_clearance.py` 新增 root/body bottom、knee、shank 和 swing foot 的 height-field clearance 约束。
- stance/touchdown 的障碍禁止逻辑使用 semantic id：`0` 为地面，`1/2` 为障碍，stance 不再靠高度差判断是否允许踩。
- high-small / large 障碍会通过全 scanner cell 的风险检测降低 linear/yaw tracking pressure；yaw-only 原地转向使用 yaw swept radius。
- `planner.py` 的 `cost_breakdown` 与诊断 `loss_breakdown` 都暴露 T302 collision/risk 项，便于 viewer/headless 接受测试读取。

验证入口见 [../log/2026-05-16-2309-t302-mpc-body-leg-collision-implementation.md](../log/2026-05-16-2309-t302-mpc-body-leg-collision-implementation.md)。

`batch_inverse_kinematics` / `batch_forward_kinematics`（`ik.py:166` 起）在 IK 主路径上 **无** 对 `batch_size` 或 `n_frames` 的 Python `for`（仅常数级设备检查）。

按 §约定符号：**不在阶中写 N**；相对一次规划 horizon，记为 **O(T)**（一次 replan 内与轨迹长度相关；**N** 不进入渐近式）。

**N = 1** 时同上：**O(T)**。

---

## 4. 与「单次 `batched_generate_trajectory`」的关系

- **Swing / stance**：**O(N·T)**（§2，含对 batch 的 Python 串行）。
- **IK / FK**：**O(T)**，阶中不写 **N**（§3）。

其余模块的时间阶见 **§7**。

---

## 5. foothold 与 base：在 `trajectory.py` 中的位置

导入：

```8:14:Go2Pvcnn/extension/batched_planner/trajectory.py
from ..convention import extract_roll_pitch_batch, extract_yaw_batch, yaw_rotation_matrix_batch
from .base_solver import batched_integrate_base_planar, batched_solve_base_trajectory
from .config import BatchedTrajectoryConfig
from .foothold import batched_compute_footholds, batched_evaluate_touchdowns
from .gait import GAIT_PARAMS, batched_gait_schedule, batched_legs_requiring_touchdown, batched_next_touchdown_times, batched_stance_time
from .ik import batch_forward_kinematics, batch_inverse_kinematics
from .swing import batched_compute_swing_targets
```

主流程片段（**当前**实现：单次 `batched_compute_footholds` + swing + 近似积分 + 地形估计 + **机座求解** + IK；无多候选指令环）：

```173:254:Go2Pvcnn/extension/batched_planner/trajectory.py
    touchdowns = batched_compute_footholds(
        base_pos=states.root_pos,
        base_yaw=initial_yaw,
        base_lin_vel_xy=commands_t[:, :2],
        ref_lin_vel_xy=commands_t[:, :2],
        hip_positions=hip_positions,
        stance_time=st,
        com_height=torch.full((batch_size,), float(cfg.hip_height), dtype=torch.float64, device=device),
        terrain=terrain,
        previous_footholds=states.foot_pos,
        touchdown_times=touchdown_times,
        yaw_rate=commands_t[:, 2],
        search_radius=cfg.foothold_search_radius,
        search_step=cfg.foothold_search_step,
        max_step_down=cfg.max_foothold_step_down,
    )
    feasible, _, _ = batched_evaluate_touchdowns(
        touchdowns,
        states.foot_pos,
        contact_seq,
        touchdown_mask,
        terrain,
        states.foot_pos,
        max_reach=cfg.max_touchdown_xy_reach,
    )
    ...
    terrain_max_heights = torch.stack(
        [
            terrain.max_height_along_segment(states.foot_pos[:, leg_idx, :2], touchdowns[:, leg_idx, :2])
            for leg_idx in range(4)
        ],
        dim=1,
    )
    foot_targets = batched_compute_swing_targets(contact_seq, states.foot_pos, touchdowns, cfg.step_height, terrain_max_heights=terrain_max_heights)

    pos_xy_approx, yaw_approx = batched_integrate_base_planar(
        states.root_pos[:, :2],
        initial_yaw,
        commands_t[:, 0],
        commands_t[:, 1],
        commands_t[:, 2],
        n_frames,
        dt,
    )
    ...
    roll, pitch, height = batched_estimate_terrain(...)
    root_pos, root_quat = batched_solve_base_trajectory(
        states.root_pos,
        initial_yaw,
        commands_t[:, 0],
        commands_t[:, 1],
        commands_t[:, 2],
        n_frames,
        dt,
        terrain,
        foot_targets,
        contact_seq,
        roll,
        pitch,
        height,
        hip_height=cfg.hip_height,
        body_clearance_margin=cfg.body_clearance_margin,
    )

    root_pos_flat = root_pos.reshape(batch_size * n_frames, 3)
    ...
    joint_angles = batch_inverse_kinematics(root_pos_flat, root_quat_flat, foot_targets_flat).reshape(batch_size, n_frames, 12)
```

### 5.1 foothold：名义落点 + 螺旋网格地形搜索

`batched_compute_footholds` 将每条腿在 **下一次触地时刻** 的机体位置前推，算 hip，再用 Raibert 风格得到 **名义 XY**，`terrain.height_at` 取 Z，最后在 **预计算螺旋偏移网格** 上选可行且评分最优的格点：

```215:256:Go2Pvcnn/extension/batched_planner/foothold.py
    lead_dt = touchdown_times_t.reshape(-1)
    ...
    lead_base_xy = _predict_planar_base_xy(...)
    ...
    nominal_xy = _raibert_foothold_xy(...)
    nominal_z = terrain.height_at(nominal_xy)
    nominal = torch.cat([nominal_xy, nominal_z.unsqueeze(-1)], dim=-1)
    best = _spiral_search_safe_foothold(
        nominal,
        terrain,
        previous_footholds_t,
        search_radius=search_radius,
        grid_step=search_step,
        max_roughness=1.0,
        max_step_down=max_step_down,
    )
    return best
```

螺旋偏移个数由 `_precompute_spiral_offsets` 在 import/调用尺度上确定（与 `search_radius`、`search_step` 有关），随后在 `_spiral_search_safe_foothold` 中对 **所有腿 × 所有偏移** 做向量化 `terrain.height_at` / `roughness_at` 与 `argmin`（`foothold.py:121-163`）。

`batched_evaluate_touchdowns` 校验 `touchdown_mask` 与 `contact_seq` 推导一致，并据 XY 跨步、粗糙度等给标量分（`foothold.py:259-292`）。

### 5.2 base：`batched_solve_base_trajectory` 组合平面、高度、姿态与机身净空

```174:215:Go2Pvcnn/extension/batched_planner/base_solver.py
def batched_solve_base_trajectory(
    initial_pos,
    initial_yaw,
    vx,
    vy,
    yaw_rate,
    n_frames: int,
    dt: float,
    terrain,
    foot_targets,
    contact_seq,
    terrain_roll,
    terrain_pitch,
    terrain_height,
    hip_height: float = HIP_HEIGHT,
    body_clearance_margin: float = 0.012,
) -> tuple[Tensor, Tensor]:
    ...
    pos_xy, yaw = batched_integrate_base_planar(initial_pos_t[:, :2], initial_yaw_t, vx_t, vy_t, yaw_rate_t, n_frames, dt)
    z = batched_solve_base_height(terrain_height_t, foot_targets_t, contact_seq_t, hip_height=hip_height)
    quat = batched_solve_base_orientation(terrain_roll_t, terrain_pitch_t, yaw)
    root_pos = torch.cat([pos_xy, z.unsqueeze(-1)], dim=-1)
    z_adjustment = batched_body_clearance_adjustment(root_pos, quat, terrain, margin=body_clearance_margin)
    root_pos = root_pos.clone()
    root_pos[..., 2] = root_pos[..., 2] + z_adjustment
    return root_pos, quat
```

- **平面**：`batched_integrate_base_planar` 按常值 `vx, vy, yaw_rate` 在每一帧用机体前向速度累加位移（`base_solver.py:89-115`）。
- **高度**：`batched_solve_base_height` 用支撑足加权高度 + `hip_height` 得目标 Z，再按 `smooth_factor` 做帧间一阶平滑；实现里对 `t in range(num_frames)` 有 **Python 标量循环**（`base_solver.py:130-136`），步数为 **O(T)**。
- **姿态**：`batched_solve_base_orientation` 将地形 roll/pitch 与积分 yaw 合成四元数（`base_solver.py:140-151`）。
- **净空**：`batched_body_clearance_adjustment` 在若干预定义机体采样点上查地形，抬升 root z（`base_solver.py:154-171`；采样点数 `BODY_COLLISION_SAMPLES` 为常数）。

**注意**：`trajectory.py` 在 swing 之后还有一次 `batched_integrate_base_planar`（`211-219` 行），用于 `batched_estimate_terrain` 的近似机体轨迹；与 `batched_solve_base_trajectory` 内部的积分 **不是** 同一次调用（后者在得到 `foot_targets` 与地形 roll/pitch/height 之后执行）。

---

## 6. 代码执行地图（code-flow-guide 风格摘要）

下列大流程与 `batched_generate_trajectory` 内子步骤一一对应；训练侧入口仍见 [human-10-extension-planner-runtime.md](human-10-extension-planner-runtime.md)。

```text
batched_generate_trajectory()
Go2Pvcnn/extension/batched_planner/trajectory.py:121

↓

gait 与时间窗（contact_seq / touchdown_times / stance_time）
Go2Pvcnn/extension/batched_planner/trajectory.py:147-166

↓

foothold + touchdown 可行性
Go2Pvcnn/extension/batched_planner/trajectory.py:173-197
Go2Pvcnn/extension/batched_planner/foothold.py:166-292

↓

摆腿目标 + 地形姿态估计 + 机座轨迹 + IK/FK
Go2Pvcnn/extension/batched_planner/trajectory.py:202-254
Go2Pvcnn/extension/batched_planner/swing.py:126-164
Go2Pvcnn/extension/batched_planner/base_solver.py:174-215
Go2Pvcnn/extension/batched_planner/ik.py:166-204
```

**Call chain（foothold / base 段）**：

```text
batched_generate_trajectory
↓
batched_compute_footholds
  → _predict_planar_base_xy / _raibert_foothold_xy / _spiral_search_safe_foothold
↓
batched_evaluate_touchdowns
↓
terrain.max_height_along_segment × 4 legs
↓
batched_compute_swing_targets
↓
batched_integrate_base_planar（供 terrain 估计）
↓
batched_estimate_terrain
↓
batched_solve_base_trajectory
  → batched_integrate_base_planar
  → batched_solve_base_height
  → batched_solve_base_orientation
  → batched_body_clearance_adjustment
↓
batch_inverse_kinematics / batch_forward_kinematics
```

---

## 7. 时间复杂度总表（一次 `batched_generate_trajectory`）

约定：**仅当**存在对 **N** 的 Python 串行（或等价的按 env 顺序 `.item()`）时，阶中含 **N**；否则不写 **N**。以下针对 `trajectory.py:121` 主路径；常数腿数 4、净空采样数 **S** 吸收进常数。

| 模块 | 代码锚点 | 时间复杂度 | 是否含 N（及理由） |
| --- | --- | --- | --- |
| `batched_gait_schedule` 等 | `trajectory.py:147-166` | **O(T)** | 否：无对 `batch_size` 的 Python `for` |
| `batched_compute_footholds`（含螺旋表构造） | `trajectory.py:173`，`foothold.py:22-30`、`131-163` | **O(K)** | 否：螺旋阶与 `N` 无 Python 串行绑定（主算子批在 N 上） |
| `batched_evaluate_touchdowns` | `trajectory.py:189`，`foothold.py:259-290` | **O(1)** | 否：无对 N 的 Python `for` |
| `max_height_along_segment` ×4 | `trajectory.py:202-207`，`terrain.py:266-287` | **O(N)** | **是**：`for idx in range(self.batch_size)` + `.item()` |
| `batched_compute_swing_targets` | `trajectory.py:209`，`swing.py:149-150`、`104-115` | **O(N·T)** | **是**：对 batch/leg 的 Python 串行 + 每腿 **O(T)** |
| `batched_integrate_base_planar`（两次） | `trajectory.py:211-219`，`base_solver.py:89-115`、`208` | **O(T)** | 否 |
| `batched_estimate_terrain` | `trajectory.py:223-231`，`terrain_estimator.py:127-134` | **O(T)** | 否：阶由 `for t in range(num_frames)` 决定，不写 N |
| `batched_solve_base_trajectory`（含高度平滑） | `trajectory.py:232-248`，`base_solver.py:134-136` 等 | **O(T)** | 否：高度平滑为 `for t in range(num_frames)`，不写 N |
| IK / FK + 后处理 | `trajectory.py:250-263`，`ik.py:166` 起 | **O(T)** | 否 |

**串行段相加（同一次 replan）**：含 **N** 的项为 **O(N)**（地形沿段）与 **O(N·T)**（swing）；其余为 **O(T)**、**O(K)**、**O(1)**。总时间复杂度上界可写为：

**O(N·T + N + T + K)**（常数与 **S** 已吸收；若只关心 N、T 充分大时的主阶，则为 **O(N·T)**）。

**N = 1**：**O(T + K)**（地形沿段 **O(N)** 降为 **O(1)**；swing **O(N·T)** 降为 **O(T)**）。

---

## 8. 相关代码入口（仓库相对路径）

- `Go2Pvcnn/extension/batched_planner/gait.py`：步态相位、`contact_seq`、`stance_time`
- `Go2Pvcnn/extension/batched_planner/foothold.py`：名义落点、螺旋搜索、`touchdown` 评分
- `Go2Pvcnn/extension/batched_planner/base_solver.py`：平面积分、base 高度/姿态、机身净空
- `Go2Pvcnn/extension/batched_planner/swing.py`：swing 弧与 stance 锚点
- `Go2Pvcnn/extension/batched_planner/ik.py`：闭式 IK / FK
- `Go2Pvcnn/extension/batched_planner/trajectory.py`：组装调用链
