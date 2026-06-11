# T302q Flat Small Avoidance Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This page is the implementation plan and branch memory for T302q.

**Goal:** 新增平地小障碍物避障 RL 配置，在不改变 observation/action shape、不改 MPC reference/loss 的前提下，从现有 teacher checkpoint 继续训练，并用真实 IsaacLab body pose + 当前 IsaacLab scanner semantic/elevation map 提供近场腿部避障奖励。

**Architecture:** 新配置继承当前 `TeacherElevationTrajectoryMpcSemanticEnvCfg`，只挂到新的 gym/experiment name。新增 `semantic_body_part_clearance_reward()` 读取 IsaacLab 当前 `foot/calf/thigh` body pose，直接使用当前 `semantic_height_scanner.data.elevation_map/semantic_map` 构造和 MPC 一致的 terrain query，并按 scanner 当前 pose 查询 height/semantic；当前 row-based semantic curriculum 直接改为 episode-level small-contact gate，不新建第二套 curriculum。

**Tech Stack:** IsaacLab ManagerBased RL env / `RewTerm` / `SceneEntityCfg`、`robot.data.body_pos_w`、`semantic_height_scanner`、`semantic_contact_small.data.force_matrix_w`、PyTorch batch tensor、RSL-RL resume checkpoint、`env_isaacsim`。

---

## Source Spec

- [../../docs/superpowers/specs/2026-06-10-flat-small-obstacle-avoidance-reward-design.html](../../docs/superpowers/specs/2026-06-10-flat-small-obstacle-avoidance-reward-design.html)
- Upstream curriculum: [T302n row-gated semantic obstacle curriculum](T302n-semantic-obstacle-curriculum-plan.md)
- Upstream contact route: [T302l MPC RL participation and reward](T302l-mpc-rl-participation-and-reward-plan.md)
- Eval reference: [T302o MPC policy eval](T302o-mpc-policy-eval-plan.md)

## Current State

- Design is approved and recorded in commits:
  - `1858a4a docs: design flat small avoidance reward`
  - `da46138 docs: refine flat small avoidance curriculum design`
- Existing checkpoint for warm start:

```text
logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic/2026-06-04_18-16-07
```

- Current semantic curriculum now keeps row-based static obstacle counts and uses an episode-level flat gate: a completed flat episode is successful only when it times out without small semantic contact, base contact, or bad orientation.
- Current semantic contact source is authoritative IsaacLab contact:

```python
semantic_contact_small.data.force_matrix_w  # [num_envs, num_bodies, num_small_objects, 3]
semantic_contact_large.data.force_matrix_w  # [num_envs, num_bodies, num_large_objects, 3]
```

- `mpc_policy_eval.py --mode small_collision` uses dense small-obstacle flat scenes, with default `--small-count-per-tile=80`.
- Local implementation through Task 8 is complete in the working tree:
  - new flat-small cfg and train/play/registry entries are added
  - `semantic_body_part_clearance_reward` is implemented with current IsaacLab scanner maps, current scanner pose, the shared MPC terrain query helper, and current IsaacLab body pose samples
  - current row-based semantic curriculum now has episode-level true small-contact success helpers and term tests
  - new experiment is added to `TRAJECTORY_MANAGER_EXPERIMENTS`, so planner-owned `reference_foot_pos_reward` attaches the MPC trajectory manager
  - focused local verification passed: `31 passed`; touched production `py_compile` exit `0`
  - real IsaacLab fresh train smoke passed with 16 envs / 1 iteration
  - real IsaacLab resume smoke passed from `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic/2026-06-04_18-16-07/model_14000.pt`
- Flat-small 1024-env slowdown root cause is fixed locally in `SemanticGridRayCaster`: repeated late semantic mesh refresh polling was happening because the flat-small config intentionally has no large-obstacle semantic geometry. Internal timing showed pre-fix `refresh=1684-1877ms/chunk` while `raycast=1.6-2.1ms`; after the fix `refresh=0.01-0.02ms`, steady `observation.compute=21-22ms`, and normal resumed training collection falls in the `3.768-6.857s` range after startup. See [../log/2026-06-10-2317-t302q-semantic-raycaster-refresh-fix.md](../log/2026-06-10-2317-t302q-semantic-raycaster-refresh-fix.md).
- Flat-small curriculum flat-mask bookkeeping is fixed locally: when a single flat sub-terrain generates multiple IsaacLab terrain column ids, all of those columns now count as flat. RED caught the previous column-0-only behavior, focused `21 passed`, pycompile exit `0`, and a real 64-env IsaacLab probe reports `plane_mask_count=64` / `plane_env_count=64`. See [../log/2026-06-11-1428-t302q-flat-small-plane-mask-fix.md](../log/2026-06-11-1428-t302q-flat-small-plane-mask-fix.md).
- Flat-small velocity curriculum is disabled only in `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg`: base config keeps `lin_vel_cmd_levels`, flat-small sets it to `None`, and real 8-env smoke shows Curriculum Manager active terms contain only `terrain_levels`. See [../log/2026-06-11-1621-t302q-flat-small-remove-velocity-curriculum.md](../log/2026-06-11-1621-t302q-flat-small-remove-velocity-curriculum.md).
- TensorBoard readout for `2026-06-11_17-05-24` is stable and confirms fixed bookkeeping (`plane_env_count=1024`, no non-flat move-ups, speed curriculum disabled), but `semantic_body_part_clearance` remains all zero and the semantic gate stays closed. See [../log/2026-06-11-1724-t302q-flat-small-tensorboard-readout.md](../log/2026-06-11-1724-t302q-flat-small-tensorboard-readout.md).
- Env-level collision curriculum redesign is written for review in [../../docs/superpowers/specs/2026-06-11-flat-small-env-level-collision-curriculum-design.html](../../docs/superpowers/specs/2026-06-11-flat-small-env-level-collision-curriculum-design.html). The design removes the global semantic gate from flat move-up, makes each env decide at episode end, keeps only `mean_terrain_level` as the curriculum TensorBoard metric, and uses small collision to block upgrade without first-version forced downgrade. See [../log/2026-06-11-2156-flat-small-env-level-collision-curriculum-html-design.md](../log/2026-06-11-2156-flat-small-env-level-collision-curriculum-html-design.md).
- Current implementation handoff moved to [T302s env-level collision curriculum plan](T302s-env-level-collision-curriculum-plan.md). Old T302q todo items that require global `semantic_gate_pass`, `min_completed_episodes` gating, or noisy curriculum TensorBoard metrics are closed as conflicting with the approved redesign.
- T302s implementation completed locally; use [../log/2026-06-11-2211-t302s-env-level-collision-curriculum-implementation.md](../log/2026-06-11-2211-t302s-env-level-collision-curriculum-implementation.md) as the current curriculum behavior evidence.
- Small-collision eval smoke is still pending.

## Confirmed Requirements

- [x] New experiment name: `teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance`.
- [x] New cfg class: `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg`.
- [x] Inherit current `TeacherElevationTrajectoryMpcSemanticEnvCfg`.
- [x] Keep policy/critic observation shape unchanged.
- [x] Keep action space unchanged.
- [x] Do not modify MPC planner loss, MPC reference cache ABI, command shaping, or planner targets.
- [x] Do not use SemLoco foothold search.
- [x] Do not duplicate MPC FK for reward; read IsaacLab current body pose.
- [x] Reward detects `foot`, `calf/shank`, and `thigh` clearance against the current IsaacLab scanner semantic/elevation map.
- [x] Reward no longer keeps a private scanner-map root anchor cache; it follows MPC by querying current scanner data and scanner pose.
- [x] No root displacement/yaw staleness gate; map pose ownership stays with IsaacLab scanner data and the shared MPC terrain query.
- [x] Directly modify current row-based semantic curriculum gate to episode-level small-contact success; do not create a new curriculum route.
- [x] Use real `semantic_contact_small` collision for curriculum success, not reward proxy.
- [x] Training flat-small obstacle count table defaults to `8,16,24,32,40,48,56,64,72,80`, all large counts `0`.
- [x] Final row aligns with eval default `small_count_per_tile=80`.

## File Structure

- Create `Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py`
  - Owns pure tensor helpers, current scanner terrain construction, body-part sample generation, and IsaacLab reward entry.
- Modify `Go2Pvcnn/extension/mdp/__init__.py`
  - Exports the new reward function if local mdp exports use explicit imports.
- Modify `Go2Pvcnn/extension/semantic_curriculum.py`
  - Adds episode-level state fields and helpers while preserving row-count layout helpers from T302n.
- Modify `Go2Pvcnn/go2_pvcnn/mdp/curriculums.py`
  - Replaces flat semantic gate internals with episode-level success windows based on `semantic_contact_small`.
- Modify `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
  - Adds flat-only terrain cfg, flat-small semantic curriculum counts, new reward `RewTerm`, and train/play/eval-compatible cfg class if needed.
- Modify `Go2Pvcnn/go2_pvcnn/tasks/register_envs.py`
  - Registers the new gym id and experiment name without changing existing main route.
- Modify train/play mapping files only if the current scripts require explicit cfg-name allowlists.
- Create `Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py`
  - Pure tensor reward, scanner-pose projection, and current-root wrapper tests.
- Modify `Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py`
  - Episode-level curriculum state tests.
- Modify `Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py`
  - Curriculum term tests using fake reset/time-out/termination signals and fake semantic contact.
- Modify `Go2Pvcnn/tests/test_batch_mpc_backend.py` or create a focused cfg static test file
  - New cfg inheritance, reward wiring, obstacle count table, observation/action compatibility static contracts.
- Update `notes/todo.md`, this page, `notes/log/index.md`, and create per-verification logs as tasks complete.

## Function Contracts

### Reward Entry

```python
def semantic_body_part_clearance_reward(
    env,
    *,
    asset_cfg,
    scanner_cfg,
    contact_sensor_cfg=None,
    small_semantic_ids=(1,),
    foot_margin_m=0.015,
    shank_margin_m=0.04,
    thigh_margin_m=0.04,
    foot_weight=0.5,
    shank_weight=2.0,
    thigh_weight=2.0,
    stance_foot_weight=0.5,
    swing_foot_weight=1.0,
    contact_force_threshold=1.0,
    shank_sample_count=2,
    thigh_sample_count=2,
    penalty_clip=1.0,
):
    """Return per-env negative clearance reward with shape [num_envs]."""
```

### Current Scanner Terrain

```python
def _current_scanner_terrain(
    scanner,
    *,
    device,
):
    """Build the same current scanner terrain view used by MPC-style queries."""
```

Terrain fields:

```text
elevation_map: Tensor [N,H,W]
semantic_map: Tensor [N,H,W]
sensor_pos_w: Tensor [N,3] from scanner.data.pos_w if available
sensor_yaw: Tensor [N] from scanner.data.quat_w if available
world_x_range/world_y_range: from scanner pattern size
```

### Body-Part Samples

```python
def _current_body_part_sample_points(
    robot,
    *,
    body_ids,
    shank_sample_count=2,
    thigh_sample_count=2,
):
    """Return fixed-shape current IsaacLab body samples for foot/calf/thigh."""
```

Required `body_ids` keys:

```text
foot: Tensor/List length 4
calf: Tensor/List length 4
thigh: Tensor/List length 4
```

Output groups:

```text
foot_points: [N,4,1,3]
shank_points: [N,4,S,3]
thigh_points: [N,4,T,3]
```

### Clearance From Points

```python
def _semantic_clearance_penalty_from_points(
    *,
    terrain,
    points_by_part,
    small_semantic_ids=(1,),
    margins,
    weights,
    penalty_clip=1.0,
    foot_contact_mask=None,
    stance_foot_weight=0.5,
    swing_foot_weight=1.0,
):
    """Query current MPC-compatible terrain at world points and return per-env penalty."""
```

Penalty rule:

```python
small_mask = semantic_id in small_semantic_ids
deficit = relu(terrain_z + margin_part - point_z)
point_cost = small_mask * deficit.square()
reward = -clip(weighted_reduction(point_cost), 0.0, penalty_clip)
```

## Global Constraints

- No per-env Python loop in reward hot path.
- Fixed part branches are allowed; map query and reductions must be batched tensors.
- Do not keep a private reward-side scanner root-anchor cache.
- Use the shared MPC terrain query helper for scanner pose and map coordinates.
- Keep existing `teacher_elevation_trajectory_mpc_semantic` cfg behavior unchanged.
- Do not modify `raw/` or `onlyReference/`.
- IsaacLab validation should use:

```bash
CUDA_VISIBLE_DEVICES=<card> /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python ...
```

---

### Task 1: Static Contracts For New Cfg And Registration

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/register_envs.py`
- Modify/Create tests under `Go2Pvcnn/tests/`

- [x] **Step 1: Write failing cfg tests**

Test assertions:

```python
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
    TeacherElevationTrajectoryMpcSemanticEnvCfg,
    TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg,
)

base = TeacherElevationTrajectoryMpcSemanticEnvCfg()
cfg = TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg()

assert isinstance(cfg, TeacherElevationTrajectoryMpcSemanticEnvCfg)
assert cfg.experiment_name == "teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance"
assert cfg.rewards.semantic_body_part_clearance is not None
assert getattr(base.rewards, "semantic_body_part_clearance", None) is None
assert tuple(c.small for c in cfg.semantic_obstacle_curriculum.plane_counts) == (8,16,24,32,40,48,56,64,72,80)
assert tuple(c.large for c in cfg.semantic_obstacle_curriculum.plane_counts) == (0,0,0,0,0,0,0,0,0,0)
assert set(cfg.scene.terrain.terrain_generator.sub_terrains.keys()) == {"flat"}
```

- [x] **Step 2: Run tests and confirm RED**

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract -q
```

Expected: FAIL because the cfg/reward does not exist.

- [x] **Step 3: Implement minimal cfg and registration**

Add a flat-only terrain generator by copying required scalar settings from `SEMANTIC_TERRAIN_CFG` and replacing `sub_terrains` with only `flat`.

Add `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg`:

```python
@configclass
class TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg(TeacherElevationTrajectoryMpcSemanticEnvCfg):
    experiment_name: str = "teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance"

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance"
        self.scene.terrain.terrain_generator = FLAT_SMALL_AVOIDANCE_TERRAIN_CFG
        self.semantic_obstacle_curriculum = SemanticObstacleCurriculumCfg(
            plane_counts=(
                SemanticObstacleCount(small=8, large=0),
                SemanticObstacleCount(small=16, large=0),
                SemanticObstacleCount(small=24, large=0),
                SemanticObstacleCount(small=32, large=0),
                SemanticObstacleCount(small=40, large=0),
                SemanticObstacleCount(small=48, large=0),
                SemanticObstacleCount(small=56, large=0),
                SemanticObstacleCount(small=64, large=0),
                SemanticObstacleCount(small=72, large=0),
                SemanticObstacleCount(small=80, large=0),
            ),
            non_plane_counts=(SemanticObstacleCount(small=0, large=0),),
        )
        self.rewards.semantic_body_part_clearance = RewTerm(...)
```

Register the new gym id only for this cfg. Do not repoint existing ids.

- [x] **Step 4: Run cfg tests and confirm GREEN**

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract -q
```

Expected: PASS.

### Task 2: Pure Tensor Clearance Helper

**Files:**
- Create: `Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py`
- Create: `Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py`

- [x] **Step 1: Write failing pure tensor tests**

Cover:

```python
def test_no_small_semantic_cells_returns_zero_penalty(): ...
def test_small_semantic_deficit_returns_negative_reward(): ...
def test_shank_and_thigh_weights_exceed_foot_for_same_deficit(): ...
def test_ground_semantic_id_has_no_penalty_even_when_height_close(): ...
def test_scanner_pose_projects_world_points_into_current_map_frame(): ...
```

- [x] **Step 2: Run tests and confirm RED**

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py -q
```

Expected: FAIL because helper module does not exist.

- [x] **Step 3: Implement `_semantic_clearance_penalty_from_points()`**

Implementation requirements:

```text
input world points [N,P,3]
build MpcPlannerTerrain from current scanner map and scanner pose
query elevation/semantic via height_at()/semantic_at()
mask only `small_semantic_ids`
return negative clipped per-env reward
```

Use the same batched query path as MPC so scanner pose and grid coordinates do not drift between reward and planner.

- [x] **Step 4: Run tests and confirm GREEN**

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py -q
```

Expected: PASS.

### Task 3: Body-Part Sampling From IsaacLab Pose

**Files:**
- Modify: `Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py`
- Modify: `Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py`

- [x] **Step 1: Write failing body sample tests**

Use a fake robot object with `data.body_pos_w` and body ids:

```python
body_ids = {"foot": [2,5,8,11], "calf": [1,4,7,10], "thigh": [0,3,6,9]}
points = _current_body_part_sample_points(fake_robot, body_ids=body_ids, shank_sample_count=2, thigh_sample_count=2)
assert points["foot"].shape == (N, 4, 1, 3)
assert points["shank"].shape == (N, 4, 2, 3)
assert points["thigh"].shape == (N, 4, 2, 3)
```

Assert shank samples lie between calf center and foot center; thigh samples lie between thigh center and calf center.

- [x] **Step 2: Run tests and confirm RED**

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py::test_current_body_part_sample_points_shapes -q
```

Expected: FAIL until sampling helper exists.

- [x] **Step 3: Implement `_current_body_part_sample_points()`**

Implementation requirements:

```text
no per-env loop
body ids converted once to tensors on robot body_pos device
alpha tensors cached by `(device, dtype, shank_sample_count, thigh_sample_count)`
foot uses exact foot body center
shank interpolates calf -> foot excluding or including endpoints consistently with tests
thigh interpolates thigh -> calf excluding or including endpoints consistently with tests
```

- [x] **Step 4: Run focused tests**

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py -q
```

Expected: PASS.

### Task 4: Current Scanner Terrain Query

**Files:**
- Modify: `Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py`
- Modify: `Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py`

- [x] **Step 1: Replace private cache tests with current scanner terrain tests**

Use fake scanner/root objects and assert the wrapper uses current scanner data/pose rather than a stale private cache:

```python
reward_before_move = semantic_body_part_clearance_reward(...)
robot.data.root_pos_w[:, 0] = 0.20
robot.data.body_pos_w[:, :, 0] = 0.20
reward_after_move = semantic_body_part_clearance_reward(...)
assert reward_before_move.item() < 0.0
assert reward_after_move.item() < 0.0
```

- [x] **Step 2: Run scanner terrain tests and confirm RED**

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py -q
```

Expected: FAIL while the reward still uses stale private root-anchor cache behavior.

- [x] **Step 3: Implement current scanner terrain query**

Implementation:

```text
read scanner.data.elevation_map/semantic_map
read scanner.data.pos_w/quat_w when available
construct MpcPlannerTerrain with scanner pattern size
query height_at()/semantic_at() at current body-part world XY
```

- [x] **Step 4: Run scanner terrain tests**

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py -q
```

Expected: PASS.

### Task 5: Reward Wrapper And RewTerm Wiring

**Files:**
- Modify: `Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py`
- Modify: `Go2Pvcnn/extension/mdp/__init__.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
- Modify tests from Task 1/2

- [x] **Step 1: Write failing reward wrapper test**

Use fake env/scene entries and assert:

```python
reward = semantic_body_part_clearance_reward(fake_env, asset_cfg=..., scanner_cfg=..., contact_sensor_cfg=...)
assert reward.shape == (num_envs,)
assert torch.isfinite(reward).all()
assert (reward <= 0.0).all()
```

- [x] **Step 2: Run wrapper test and confirm RED**

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py::test_reward_wrapper_returns_finite_per_env_values -q
```

Expected: FAIL until wrapper resolves scene entities and body ids.

- [x] **Step 3: Implement wrapper**

Wrapper responsibilities:

```text
resolve robot = env.scene[asset_cfg.name]
resolve scanner = env.scene[scanner_cfg.name]
resolve foot/calf/thigh body ids once and cache them
update map anchor cache
build current body-part sample points
optionally read foot contact mask from contact_sensor_cfg
call pure tensor penalty helper
return [num_envs] reward
```

- [x] **Step 4: Wire RewTerm only in new cfg**

Use:

```python
semantic_body_part_clearance = RewTerm(
    func=semantic_body_part_clearance_reward,
    weight=1.0,
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "scanner_cfg": SceneEntityCfg("semantic_height_scanner"),
        "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        "small_semantic_ids": (1,),
        "foot_margin_m": 0.015,
        "shank_margin_m": 0.04,
        "thigh_margin_m": 0.04,
        "foot_weight": 0.5,
        "shank_weight": 2.0,
        "thigh_weight": 2.0,
        "shank_sample_count": 2,
        "thigh_sample_count": 2,
        "penalty_clip": 1.0,
    },
)
```

- [x] **Step 5: Run reward/cfg tests**

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract -q
```

Expected: PASS.

### Task 6: Episode-Level Semantic Curriculum Gate

**Files:**
- Modify: `Go2Pvcnn/extension/semantic_curriculum.py`
- Modify: `Go2Pvcnn/go2_pvcnn/mdp/curriculums.py`
- Modify: `Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py`
- Modify: `Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py`

- [x] **Step 1: Write failing state tests**

Tests must assert:

```python
state.episode_had_small_collision[env_id] becomes True after any small force > threshold
state resets only for reset env ids after episode accounting
time_out and no small collision counts as success
base_contact or bad_orientation does not count as success
not enough completed episodes keeps gate_pass False
consecutive_success_required windows enable gate_pass True
```

- [x] **Step 2: Run tests and confirm RED**

```bash
pytest Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py \
  Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py -q
```

Expected: FAIL because current gate is collision-rate-window based.

- [x] **Step 3: Implement episode state**

Add state fields:

```python
episode_had_small_collision: torch.Tensor | None
completed_flat_episodes: int
successful_full_no_collision_episodes: int
last_semantic_success_rate: float
consecutive_success_count: int
```

Add helpers:

```python
def update_episode_small_collision_from_forces(state, small_force_matrix_w, threshold, *, env_ids=None) -> torch.Tensor
def record_completed_flat_episodes(state, *, reset_env_ids, flat_mask, time_out, base_contact, bad_orientation, cfg) -> dict[str, Any]
def semantic_episode_gate_pass(state, cfg) -> bool
```

- [x] **Step 4: Modify `terrain_levels_vel_semantic_plane_gate()`**

Required behavior:

```text
each curriculum call updates episode_had_small_collision from semantic_contact_small
only reset/completed envs are counted into episode statistics
success = time_out and not episode_had_small_collision and not base_contact and not bad_orientation
flat move_up = terrain_move_up and semantic_episode_gate_pass
non-flat behavior remains existing terrain logic
```

Use IsaacLab termination manager keys if available; otherwise use documented env buffers already used by current tests. Record exact key names in this page after implementation.

- [x] **Step 5: Run curriculum tests**

```bash
pytest Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py \
  Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py -q
```

Expected: PASS.

### Task 7: Focused Local Regression

**Files:**
- Test-only changes as needed.
- Create log: `notes/log/YYYY-MM-DD-HHMM-t302q-flat-small-local-regression.md`

- [x] **Step 1: Run focused suite**

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py \
  Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py \
  Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Expected: PASS.

- [x] **Step 2: Run pycompile for touched production files**

```bash
python -m py_compile \
  Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py \
  Go2Pvcnn/extension/semantic_curriculum.py \
  Go2Pvcnn/go2_pvcnn/mdp/curriculums.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py \
  Go2Pvcnn/go2_pvcnn/tasks/register_envs.py
```

Expected: exit `0`.

- [x] **Step 3: Update notes/log**

Record command outputs, candidate ref, and any remaining unverified real IsaacLab behavior.

### Task 8: IsaacLab Smoke And Checkpoint Compatibility

**Files:**
- Runtime only unless smoke exposes bug.
- Create log: `notes/log/YYYY-MM-DD-HHMM-t302q-flat-small-isaaclab-smoke.md`

- [x] **Step 1: Env creation and reward finite smoke**

Run a short headless smoke with the new cfg:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --num_envs 16 \
  --max_iterations 1 \
  --headless
```

Observed:

```text
env creates
reward manager includes semantic_body_part_clearance
observation/action shapes remain policy map [16,2,16,16], policy state [16,45], critic state [16,48], action [16,12]
fresh train smoke exits 0
```

- [x] **Step 2: Resume checkpoint smoke**

Use the existing run dir:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --num_envs 16 \
  --max_iterations 1 \
  --headless \
  --resume \
  --load_run /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic/2026-06-04_18-16-07 \
  --load_checkpoint model_14000.pt
```

Observed: exit `0`; checkpoint resume is compatible with the unchanged policy/critic/action shapes.

- [ ] **Step 3: Scanner cache smoke**

Instrument or log cache update count for a short run:

```text
cache map clone count <= scanner update count + initial clone
no per-step full map clone
```

Note: code-level cache signature tests cover scanner cache reuse locally; explicit runtime clone-count instrumentation remains optional follow-up.

### Task 9: Small-Collision Evaluation And Notes Alignment

**Files:**
- Runtime only unless evaluation exposes bug.
- Modify: `notes/todo.md`
- Modify: this page
- Modify: `notes/log/index.md`
- Create log: `notes/log/YYYY-MM-DD-HHMM-t302q-flat-small-eval-smoke.md`

- [ ] **Step 1: Run eval smoke against the new checkpoint or short trained run**

Reference command shape:

```bash
CUDA_VISIBLE_DEVICES=<card> /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode small_collision \
  --headless \
  --device cuda:0 \
  --num-envs 100 \
  --num-rounds 5 \
  --max-steps 2000 \
  --run-dir <flat-small-run-dir> \
  --checkpoint <checkpoint.pt> \
  --command-mode fixed \
  --command "1.0 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/flat_small_avoidance_smoke
```

Expected: JSON summaries include episode-level small collision metrics and no missing contact tensors.

- [ ] **Step 2: Final notes alignment**

Update:

```text
notes/todo.md
notes/todo/T302q-flat-small-avoidance-reward-plan.md
notes/log/index.md
notes/log/<verification logs>
```

Keep dashboard concise; keep detailed metrics in logs.

## Related Logs

- [../log/2026-06-11-2156-flat-small-env-level-collision-curriculum-html-design.md](../log/2026-06-11-2156-flat-small-env-level-collision-curriculum-html-design.md)
- [../log/2026-06-11-1428-t302q-flat-small-plane-mask-fix.md](../log/2026-06-11-1428-t302q-flat-small-plane-mask-fix.md)
- [../log/2026-06-11-1621-t302q-flat-small-remove-velocity-curriculum.md](../log/2026-06-11-1621-t302q-flat-small-remove-velocity-curriculum.md)
- [../log/2026-06-11-1724-t302q-flat-small-tensorboard-readout.md](../log/2026-06-11-1724-t302q-flat-small-tensorboard-readout.md)
- [../log/2026-06-11-1420-t302q-flat-small-curriculum-clearance-root-cause-probe.md](../log/2026-06-11-1420-t302q-flat-small-curriculum-clearance-root-cause-probe.md)
- [../log/2026-06-10-1945-t302q-flat-small-avoidance-plan.md](../log/2026-06-10-1945-t302q-flat-small-avoidance-plan.md)
- [../log/2026-06-10-2035-t302q-flat-small-local-implementation-and-smoke.md](../log/2026-06-10-2035-t302q-flat-small-local-implementation-and-smoke.md)

## Git Refs

- Last Feature Commit: `da46138`
- Last Verified Commit: `da46138`
- Current Work Ref: `working tree on top of da46138 (2026-06-10 20:35)`
- Key Files:
  - [../../docs/superpowers/specs/2026-06-10-flat-small-obstacle-avoidance-reward-design.html](../../docs/superpowers/specs/2026-06-10-flat-small-obstacle-avoidance-reward-design.html)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py](../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py)
  - [../../Go2Pvcnn/extension/semantic_curriculum.py](../../Go2Pvcnn/extension/semantic_curriculum.py)
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/extension/trajectory_manager_factory.py](../../Go2Pvcnn/extension/trajectory_manager_factory.py)
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)

## Next Step

- Run Task 9 small-collision eval smoke against a useful flat-small run/checkpoint.

## Node Details

### T302q.1 Static cfg and registration

- why-created: 新训练配置必须可从现有 checkpoint resume，不能污染当前 main MPC semantic cfg。
- hypothesis: 只新增继承 cfg + flat-only terrain + new reward field can preserve policy ABI.
- evidence: Design spec requires no observation/action shape change.

### T302q.2 Body-part clearance reward

- why-created: 小障碍物真实 contact 太晚，需要 contact 前连续风险反馈。
- hypothesis: IsaacLab current body pose + current scanner map via the shared MPC terrain query can provide stable low-cost clearance penalty.
- evidence: MPC heightfield clearance already uses `height_at()/semantic_at()` with scanner pose; this reward applies the same query path to current simulated body parts.

### T302q.3 Episode-level curriculum gate

- why-created: User explicitly rejected per-step statistics; upgrade should mean a complete episode without true small collision and without failure termination.
- hypothesis: `semantic_contact_small.data.force_matrix_w` can maintain `episode_had_small_collision` and produce a robust flat row gate.
- evidence: T302l verified real semantic global contact shape and performance; T302o small_collision uses the same real contact source for evaluation.
