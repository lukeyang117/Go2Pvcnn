# Parallelism Large Obstacle Lateral Avoidance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 Parallelism root 轨迹生成前，根据速度方向上的 `semantic_id=2` 大障碍物增加 GPU 并行的横向速度，使障碍物越近横移越大，并支持可调的默认绕行方向。

**Architecture:** 在 `extension/parallelism/root.py` 增加一个纯 Torch 的批量 helper，直接从 `ParallelismTerrain` 的语义网格构造世界坐标体素，计算速度方向投影、横向投影、矩形触发 mask、加权 `mean_L` 和最近前向距离。`rollout_root()` 在现有 command clamp/terrain-following 分支之前调用该 helper，把世界横向速度转换回当前 root body frame，再复用原有 root、swing、IK 和 collision 流程。所有地形共用同一逻辑，不新增 terrain type 分支。

**Tech Stack:** Python 3.10, PyTorch/TorchScript-compatible tensor operations, pytest, existing `ParallelismCfg`, `ParallelismTerrain`, `rollout_root`.

## Global Constraints

- 只处理 `semantic_id == 2`，不改变 small obstacle 和已有碰撞筛选逻辑。
- `large_obstacle_default_side=+1` 表示向左，`-1` 表示向右。
- 检测矩形初始宽度为 `0.70m`，长度为 `1.20m`。
- 障碍物越靠近 root，横向速度绝对值越大。
- `mean_L` 只决定方向，不决定横向速度大小。
- 平地、楼梯、坡面、乱石路和其他复杂地形使用同一套速度偏移公式。
- 不使用逐环境、逐体素的 Python 循环；核心计算复杂度为 `O(B*H*W)`。
- 不修改 touchdown、swing、IK、官方碰撞点或 standstill 主逻辑。

---

### Task 1: Lock the batched avoidance behavior with tests

**Files:**
- Modify: `Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py`

**Interfaces:**
- Tests will consume `extension.parallelism.root._large_obstacle_avoidance_command`.
- The helper will accept `(state, command_body, terrain, cfg)` and return a body-frame command tensor with shape `[B, 3]`.

- [ ] **Step 1: Add a reusable semantic terrain fixture**

Add a local fixture/helper that creates a `151 x 151` or smaller test grid with origin `[-2, -2, 0]`, resolution `0.01` or `0.1`, zero height, all-valid cells, and selected cells set to semantic ID `2`.

- [ ] **Step 2: Add the no-obstacle identity test**

```python
def test_large_obstacle_avoidance_keeps_command_without_semantic_two():
    from extension.parallelism.root import _large_obstacle_avoidance_command

    command = torch.tensor([[0.6, 0.0, 0.0]])
    result = _large_obstacle_avoidance_command(_state(), command, _terrain(), ParallelismCfg())

    torch.testing.assert_close(result, command)
```

- [ ] **Step 3: Add the distance monotonicity test**

Create two batched maps with the same centered large obstacle at forward distances `0.30m` and `0.90m`. Verify both produce a nonzero lateral command and:

```python
assert result[0, 1].abs() > result[1, 1].abs()
```

- [ ] **Step 4: Add the `mean_L` direction test**

Put the large obstacle on the left (`L > 0`) and verify the command adds negative body `vy`; put it on the right (`L < 0`) and verify the command adds positive body `vy`.

- [ ] **Step 5: Add the default side sign test**

Use a symmetric semantic-two obstacle so weighted `mean_L` is approximately zero. Verify:

```python
ParallelismCfg(large_obstacle_default_side=1)   # positive body vy for +left
ParallelismCfg(large_obstacle_default_side=-1)  # negative body vy for -right
```

- [ ] **Step 6: Add the root integration test**

Call `rollout_root()` on a batched terrain with one obstacle environment and one empty environment. Verify only the obstacle environment has a changed `clamped_command_body`, while root trajectory shapes remain `[B, horizon, 3]`.

- [ ] **Step 7: Run the focused tests and confirm RED**

Run:

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn
PYTHONPATH=. pytest Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py -q
```

Expected: failures because the new config fields and helper do not exist yet.

---

### Task 2: Add explicit configuration parameters

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/config.py`
- Modify: `Go2Pvcnn/tests/parallelism/test_contracts.py`

**Interfaces:**
- `ParallelismCfg.large_obstacle_rect_width_m: float = 0.70`
- `ParallelismCfg.large_obstacle_rect_length_m: float = 1.20`
- `ParallelismCfg.large_obstacle_lateral_speed_max_mps: float = 0.25`
- `ParallelismCfg.large_obstacle_default_side: int = 1`

- [ ] **Step 1: Add config assertions to the contract test**

Assert the four defaults and document the sign convention in the test name/comments:

```python
assert cfg.large_obstacle_default_side == 1  # +1 left, -1 right
```

- [ ] **Step 2: Add the fields beside other Parallelism motion/terrain parameters**

Use succinct inline comments:

```python
large_obstacle_rect_width_m: float = 0.70
large_obstacle_rect_length_m: float = 1.20
large_obstacle_lateral_speed_max_mps: float = 0.25
large_obstacle_default_side: int = 1  # +1=left, -1=right
```

- [ ] **Step 3: Run the config-only test**

Run:

```bash
PYTHONPATH=. pytest Go2Pvcnn/tests/parallelism/test_contracts.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit the configuration change**

```bash
git add Go2Pvcnn/extension/parallelism/config.py Go2Pvcnn/tests/parallelism/test_contracts.py
git commit -m "feat: add large obstacle avoidance configuration"
```

---

### Task 3: Implement the fully batched large-obstacle command helper

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/root.py`
- Test: `Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py`

**Interfaces:**
- Add:

```python
def _large_obstacle_avoidance_command(
    state: ParallelismState,
    command_body: Tensor,
    terrain: ParallelismTerrain,
    cfg: ParallelismCfg,
) -> Tensor:
    ...
```

- It returns a new tensor `[B, 3]`; it must not mutate `command_body`.

- [ ] **Step 1: Build world-space grid points in one tensor operation**

For grid shape `[B, H, W]`, create row/column coordinates `[H, W]`, convert them to terrain-local meters, rotate by `terrain.yaw_w`, and add `terrain.origin_w[:, :2]` to produce `[B, H, W, 2]`.

- [ ] **Step 2: Convert body velocity to world velocity**

Use `state.root_rpy_w[:, 2]` and the command XY components:

```python
vx_w = cos(yaw) * vx_b - sin(yaw) * vy_b
vy_w = sin(yaw) * vx_b + cos(yaw) * vy_b
```

Normalize with `clamp_min(eps)`; environments whose command XY norm is below `eps` must not receive an avoidance velocity.

- [ ] **Step 3: Compute forward and lateral projections**

For `delta = voxel_world_xy - root_xy`:

```python
forward_distance = (delta * forward_world[:, None, None]).sum(dim=-1)
lateral_position = (delta * left_world[:, None, None]).sum(dim=-1)
```

- [ ] **Step 4: Construct the front large-obstacle mask**

```python
front_large = (
    (terrain.semantic_id == 2)
    & terrain.valid_mask
    & (forward_distance >= 0.0)
    & (forward_distance <= cfg.large_obstacle_rect_length_m)
    & (lateral_position.abs() <= 0.5 * cfg.large_obstacle_rect_width_m)
)
```

- [ ] **Step 5: Compute weighted `mean_L` and nearest distance**

Use lateral Gaussian weights:

```python
sigma = max(cfg.large_obstacle_rect_width_m * 0.5, eps)
weight = torch.exp(-lateral_position.square() / (2.0 * sigma * sigma))
weighted_count = (front_large * weight).sum(dim=(-1, -2))
mean_l = (front_large * weight * lateral_position).sum(dim=(-1, -2))
mean_l = mean_l / weighted_count.clamp_min(eps)
nearest_s = torch.where(front_large, forward_distance, torch.full_like(forward_distance, torch.inf))
nearest_s = nearest_s.amin(dim=(-1, -2))
has_front_large = front_large.any(dim=(-1, -2))
```

- [ ] **Step 6: Convert direction and distance to lateral speed**

Use `mean_L` only for sign:

```python
side = torch.where(
    mean_l > eps,
    torch.full_like(mean_l, -1.0),
    torch.where(
        mean_l < -eps,
        torch.full_like(mean_l, 1.0),
        torch.full_like(mean_l, float(cfg.large_obstacle_default_side)),
    ),
)
proximity = (1.0 - nearest_s / cfg.large_obstacle_rect_length_m).clamp(0.0, 1.0)
speed = cfg.large_obstacle_lateral_speed_max_mps * proximity
speed = torch.where(has_front_large, speed, torch.zeros_like(speed))
```

Here `side=-1` means right and `side=+1` means left. For `mean_l > 0` (obstacle left), the avoidance velocity is rightward.

- [ ] **Step 7: Convert avoidance velocity to body frame and add it**

Build `avoid_world = side * speed * left_world`, rotate with `R_yaw.T`, add only to command XY, and clamp the resulting body command with existing `clamp_command`.

- [ ] **Step 8: Run the focused tests and confirm GREEN**

Run:

```bash
PYTHONPATH=. pytest Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py Go2Pvcnn/tests/parallelism/test_contracts.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit the helper implementation**

```bash
git add Go2Pvcnn/extension/parallelism/root.py Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py
git commit -m "feat: add batched large obstacle lateral avoidance"
```

---

### Task 4: Integrate the helper before all terrain-specific root rollout branches

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/root.py`
- Test: `Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py`

**Interfaces:**
- `rollout_root()` remains the public entry point and keeps its existing signature.
- The only data-flow change is:

```text
raw command
  -> large obstacle lateral command correction
  -> flat/terrain-following clamp
  -> root XY/yaw
  -> existing root Z/RPY logic
```

- [ ] **Step 1: Call the helper once per batched rollout**

Inside `rollout_root()`, compute:

```python
command_input = torch.as_tensor(...)
command_avoid = _large_obstacle_avoidance_command(state, command_input, terrain, cfg)
flat_command = clamp_command(command_avoid, cfg)
terrain_command = soft_clamp_terrain_command(command_avoid, cfg)
```

- [ ] **Step 2: Keep terrain selection unchanged**

Preserve the existing `terrain_following_mask` selection. The helper must run before the mask so flat and non-flat terrain share identical avoidance behavior.

- [ ] **Step 3: Verify flat and terrain-following integration**

Add a test with `terrain_following_mask=torch.tensor([False, True])` and identical semantic maps. Assert that both environments receive the same lateral correction while their existing Z/RPY branches remain selected independently.

- [ ] **Step 4: Run the complete parallelism test suite**

Run:

```bash
PYTHONPATH=. pytest Go2Pvcnn/tests/parallelism -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit integration**

```bash
git add Go2Pvcnn/extension/parallelism/root.py Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py
git commit -m "feat: apply large obstacle avoidance to all terrain root rollouts"
```

---

### Task 5: Validate batched performance and regression behavior

**Files:**
- Modify: `Go2Pvcnn/tests/parallelism/parallelism_small_obstacle_runtime_probe.py` only if a reusable benchmark hook is needed.
- Create: `Go2Pvcnn/tests/parallelism/parallelism_large_obstacle_runtime_probe.py`

**Interfaces:**
- Benchmark the helper with `[1024, 151, 151]` semantic maps and `[1024, 3]` commands.
- Benchmark CPU fallback if CUDA is unavailable; use CUDA synchronization around timing when CUDA is available.

- [ ] **Step 1: Add a batched runtime probe**

The probe must:

```text
create one batched terrain
fill a configurable fraction of semantic=2 cells
call _large_obstacle_avoidance_command repeatedly
warm up CUDA
measure synchronized elapsed time
report output shape and peak memory when CUDA is available
```

- [ ] **Step 2: Run the probe at 1024 environments**

Run:

```bash
PYTHONPATH=. python Go2Pvcnn/tests/parallelism/parallelism_large_obstacle_runtime_probe.py \
  --num-envs 1024 --height 151 --width 151 --device cuda:0
```

Expected: one batched call, no Python loop over environments, output shape `[1024, 3]`.

- [ ] **Step 3: Run regression tests**

```bash
PYTHONPATH=. pytest Go2Pvcnn/tests/parallelism Go2Pvcnn/tests/tracking -q
```

- [ ] **Step 4: Commit the runtime probe and final test updates**

```bash
git add Go2Pvcnn/tests/parallelism/parallelism_large_obstacle_runtime_probe.py
git commit -m "test: benchmark batched large obstacle avoidance"
```

---

### Task 6: Final verification and working-tree check

**Files:**
- No new production files.

- [ ] **Step 1: Run syntax and diff checks**

```bash
python -m compileall -q Go2Pvcnn/extension/parallelism
git diff --check
```

- [ ] **Step 2: Confirm design and implementation consistency**

Check that the code and HTML design agree on:

```text
rect_width=0.70
rect_length=1.20
max_lateral_speed=0.25
default_side=+1 left / -1 right
mean_L only controls direction
nearest forward distance controls speed magnitude
all terrain types share the same helper
```

- [ ] **Step 3: Report commits and test results**

Report the final commit IDs, focused pytest result, complete regression result, and runtime probe output. Do not claim Isaac Sim validation unless it was actually run.
