# Joint MPC RTI 关节顺序与 Stance 贴地修复设计

## 目标

修复 `joint_mpc_rti` viewer 与训练边界中的关节顺序错配，并保证 fixed-trot 在零速度、不同方向、不同速度大小和线速度/角速度组合下，所有标记为 stance 的足端保持地面接触。零速度允许腿继续摆动，但 root XY 和 yaw 不应产生指令外位移。

## 约束

- 保留 `H=16`、每次发布 `x1`、fixed diagonal trot 和单次 RTI。
- 不添加语义硬 gate、足端 snap、轨迹后处理投影或指定跨越腿。
- 零速度不切换为 stand gait；只要求 root 不漂移和 stance 贴地。
- 世界坐标高程场仍是唯一地面高度来源。
- 支持任意 Isaac `robot.joint_names` 排列，不能写死当前 USD 的数组顺序。

## 方案比较

### 方案 A：只提高 stance loss 权重

改动最小，但当前 `stance_xy_lock` 没有进入 RTI 的 LQ 方向，只提高 merit 权重不能稳定地产生保持世界足端的搜索方向，不采用。

### 方案 B：发布 x1 前对 stance 足端做 IK 投影

可以快速贴地，但属于优化后硬修复，会破坏 RTI merit、warm start 和连续约束一致性，不采用。

### 方案 C：公共顺序边界 + stance 世界锚点 Gauss–Newton 线性化

采用。关节顺序在共享 integration helper 中统一；每个 stance 段建立世界坐标足端锚点，现有 `stance_xy_lock` 使用锚点残差，`stance_ground_contact` 使用高程图地面残差，两者通过足端 Jacobian进入 LQ 的 joint block。line search merit 与 LQ 使用同一残差语义。

## 数据流

```text
Isaac robot-order joint_pos/joint_vel
-> robot_to_planner_joint_order()
-> measured JointMpcRtiState
-> fixed_trot_schedule()
-> nominal rollout
-> stance segment anchor XY
-> height-map query at planned foot/anchor
-> stance XY + Z residual/Jacobian
-> SQP-RTI + line search
-> planner-order x1
-> planner_to_robot_joint_order()
-> Isaac/PPO reference
```

## 关节顺序边界

新增 `integration/joint_order.py`，提供：

- `PLANNER_JOINT_ORDER`
- `joint_order_indices(source_order, target_order)`
- `reorder_joints(values, source_order, target_order)`
- `robot_to_planner_joints(values, robot_joint_names)`
- `planner_to_robot_joints(values, robot_joint_names)`

`state_from_env()` 同时转换 `joint_pos` 与 `joint_vel`。viewer 删除本地重复实现并复用公共函数。

## Stance 世界锚点

对每条腿、每个 horizon node：

- frame 0 为 contact 时，锚点取真实 measured foot 世界 XY。
- swing -> stance 转换时，锚点取 nominal touchdown 的世界 XY。
- 连续 stance 节点保持同一锚点。
- swing 节点保留 nominal foot，仅不参与 stance residual。

现有 loss 语义调整为：

```text
stance_xy_lock = ||foot_xy - stance_anchor_xy||²
stance_ground_contact = (foot_z - height_map(foot_xy))²
stance_slip_velocity = ||foot_xy[t] - foot_xy[t-1]||² / dt²
```

RTI 线性化对 XY 和 Z 残差都使用 `foot_jacobian_joint()`，形成 joint gradient 和对角 Gauss–Newton 近似。root 命令仍由现有 command residual 控制，不为零速度增加 gait gate。

## 验收

### 单元/纯张量

- 当前 Isaac joint order 转换后准确得到 planner order，位置和速度都必须通过。
- 连续 stance anchor 在 contact 段保持不变，swing->stance 时更新。
- rolling `x1 -> measured x0` 多周期测试覆盖：零速、前后、左右、快慢、yaw 和混合命令。
- 零速 root XY/yaw 漂移不超过数值容差。
- flat terrain 所有 stance foot gap 达到 `<= 0.01 m`，并检查无穿地、无非有限值。

### 真实 Isaac viewer

- adapter order error `< 1e-6 rad`。
- actual joint/foot 继续匹配 planner。
- 零速与多速度组合的 stance ground gap `<= 0.01 m`。
- 灾难性 `2.5 rad` / `0.59 m` 跳变消失。
