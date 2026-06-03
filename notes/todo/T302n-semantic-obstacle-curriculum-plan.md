# T302n Row-Gated Semantic Obstacle Curriculum Plan

> **For agentic workers:** Treat this todo as the implementation plan. Use checkbox (`- [ ]` / `- [x]`) syntax for tracking. Keep edits scoped to semantic obstacle row layout, flat-only terrain curriculum gate, task cfg wiring, tests, and evidence logs.

**Goal:** Replace the old semantic-level runtime rebuild route with a static row-based semantic obstacle layout and a flat-only semantic collision gate for terrain row upgrades.

**Architecture:** Semantic obstacles are spawned once at environment creation from `terrain row + terrain type`. `plane_counts[row]` and `non_plane_counts[row]` define 10 tunable row difficulties. The terrain curriculum term follows `terrain_levels_vel_unitree_rl_lab`, but flat env `move_up` additionally requires a global flat semantic collision gate; non-flat envs keep the original terrain curriculum behavior.

**Tech Stack:** IsaacLab `CurriculumTermCfg`, `TerrainImporter.update_env_origins`, PyTorch, PhysX contact `force_matrix_w`, `env_isaacsim` (`/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`).

---

## Source Spec

- [../../docs/superpowers/specs/2026-06-03-semantic-obstacle-curriculum-design.html](../../docs/superpowers/specs/2026-06-03-semantic-obstacle-curriculum-design.html)
- Related runtime design: [../../docs/superpowers/specs/2026-05-30-mpc-rl-participation-and-runtime-design.html](../../docs/superpowers/specs/2026-05-30-mpc-rl-participation-and-runtime-design.html)

## Confirmed Requirements

- [x] Delete the old semantic-level route from production code.
- [x] Do not rebuild `/World/semantic_course` during training.
- [x] Keep semantic objects static after initialization.
- [x] `plane_counts` and `non_plane_counts` are row tables, not semantic-level tables.
- [x] Default count tables have 10 entries, aligned to terrain rows 0-9.
- [x] Flat envs can move to a harder row only when terrain progress passes and flat semantic collision rate has passed for `consecutive_success_required` calls.
- [x] Non-flat env reset / row movement must not be affected by semantic collision gate.
- [x] Collision rate uses `semantic_contact_small.data.force_matrix_w` and `semantic_contact_large.data.force_matrix_w`.
- [x] Do not add min episode/window/update interval/terrain-ready parameters.
- [x] Do not change MPC losses or low-small MPC behavior.

## Global Constraints

- Do not modify IsaacLab source.
- Preserve semantic prim path format:

```text
/World/semantic_course/small/row_XX/col_YY/slot_ZZ
/World/semantic_course/large/row_XX/col_YY/slot_ZZ
```

- IsaacLab validation uses:

```bash
CUDA_VISIBLE_DEVICES=<card> /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python ...
```

## Task 1: Replace Semantic Curriculum Data Model

**Files:**
- Modify: `Go2Pvcnn/extension/semantic_curriculum.py`
- Modify: `Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py`

- [x] **Step 1: Rewrite tests for row-count config**

Tests must assert:

```python
cfg = SemanticObstacleCurriculumCfg(
    plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(2, 0)),
    non_plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
)
assert count_for_row(cfg, row=0, terrain_name="flat") == SemanticObstacleCount(0, 0)
assert count_for_row(cfg, row=1, terrain_name="flat") == SemanticObstacleCount(2, 0)
assert count_for_row(cfg, row=99, terrain_name="flat") == SemanticObstacleCount(2, 0)
assert count_for_row(cfg, row=1, terrain_name="boxes") == SemanticObstacleCount(1, 0)
```

State tests must assert:

```python
state = SemanticObstacleCurriculumState()
assert not state.update_gate_from_plane_collision_rate(0.02, cfg, plane_env_count=8)["gate_pass"]
assert state.update_gate_from_plane_collision_rate(0.02, cfg, plane_env_count=8)["gate_pass"]
assert state.consecutive_success_count == 2
state.update_gate_from_plane_collision_rate(0.50, cfg, plane_env_count=8)
assert state.consecutive_success_count == 0
```

- [x] **Step 2: Run row-count tests and confirm RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py -q
```

Expected: FAIL because old code still exposes level helpers and level-up state.

- [x] **Step 3: Implement row-count config**

Change `SemanticObstacleCurriculumCfg`:

```python
plane_counts: tuple[SemanticObstacleCount, ...] = DEFAULT_PLANE_COUNTS
non_plane_counts: tuple[SemanticObstacleCount, ...] = DEFAULT_NON_PLANE_COUNTS
```

Default tables must contain 10 entries. Validation checks:

- both count tuples are non-empty.
- counts are non-negative integers.
- layout tuples remain finite and non-negative.
- layout tuple lengths can be either `1` or equal to `max(len(plane_counts), len(non_plane_counts))`; index by clamped row.
- `collision_force_threshold >= 0`.
- `0 <= plane_collision_rate_threshold <= 1`.
- `consecutive_success_required >= 1`.

Add helpers:

```python
def clamp_row_index(row: int, count_len: int) -> int
def count_for_row(cfg, *, row: int, terrain_name: str | None) -> SemanticObstacleCount
def layout_index_for_row(cfg, row: int) -> int
def layout_values_for_row(cfg, row: int) -> tuple[float, float, float]
```

Replace `SemanticObstacleCurriculumState.level` with only:

```python
consecutive_success_count: int = 0
last_plane_collision_rate: float = 0.0
```

Add:

```python
def update_gate_from_plane_collision_rate(rate, cfg, *, plane_env_count=None) -> dict[str, Any]
```

- [x] **Step 4: Run tests and confirm GREEN**

Run:

```bash
pytest Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py -q
```

Expected: PASS.

## Task 2: Make Semantic Course Consume Row Counts

**Files:**
- Modify: `Go2Pvcnn/extension/semantic_course.py`
- Modify: `Go2Pvcnn/tests/test_semantic_course_curriculum_layout.py`

- [x] **Step 1: Rewrite semantic course tests**

Tests must assert:

- `semantic_counts_for_tile(row=0, flat)` uses `plane_counts[0]`.
- `semantic_counts_for_tile(row=1, flat)` uses `plane_counts[1]`.
- `semantic_counts_for_tile(row=99, flat)` clamps to final plane count.
- `semantic_counts_for_tile(row=1, boxes)` uses `non_plane_counts[1]`.
- `build_course_anchors(...)` produces different counts for row 0 and row 1 without passing `semantic_curriculum_level`.

- [x] **Step 2: Run tests and confirm RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_semantic_course_curriculum_layout.py -q
```

Expected: FAIL while old `semantic_level` indexing is still active.

- [x] **Step 3: Implement row-count consumption**

Change `semantic_counts_for_tile(...)` signature to remove `semantic_level` consumption:

```python
semantic_counts_for_tile(row, col, terrain_names, curriculum_cfg, fallback_stage)
```

When enabled:

```python
count = count_for_row(curriculum_cfg, row=row, terrain_name=terrain_name)
return count_to_dict(count)
```

Change layout resolver to index by row:

```python
layout_cfg_for_row(base_layout_cfg, curriculum_cfg, row)
```

Remove `semantic_curriculum_level` from `build_course_anchors(...)`, `spawn_semantic_course_prestartup(...)`, and `SemanticCourseTerrainImporter.spawn_static_semantic_course()`.

- [x] **Step 4: Run tests and confirm GREEN**

Run:

```bash
pytest Go2Pvcnn/tests/test_semantic_course_curriculum_layout.py -q
```

Expected: PASS.

## Task 3: Replace Runtime Rebuild Curriculum With Flat-Only Terrain Gate

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/mdp/curriculums.py`
- Modify: `Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py`

- [x] **Step 1: Rewrite curriculum term tests**

Tests must use a fake terrain with:

```python
terrain_types = torch.tensor([0, 1, 0, 1])
terrain_levels = torch.tensor([0, 0, 0, 0])
env_origins = torch.zeros(4, 3)
terrain_origins = torch.zeros(10, 2, 3)
```

and fake root positions/commands so all envs satisfy `terrain_move_up=True`.

Assertions:

- when `semantic_gate_pass=False`, flat envs do not move up, non-flat envs move up.
- when `semantic_gate_pass=True`, flat and non-flat envs move up.
- a flat collision resets consecutive success count and blocks flat move-up.
- no output or state writes `semantic_obstacle_curriculum_level`.
- no call to `spawn_static_semantic_course()`.

- [x] **Step 2: Run tests and confirm RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py -q
```

Expected: FAIL while old `semantic_obstacle_levels()` upgrades semantic level and rebuilds course.

- [x] **Step 3: Implement new curriculum term**

Keep helpers:

```python
semantic_collision_mask_from_force_matrices(...)
plane_env_mask_from_terrain(...)
```

Delete production use of:

```python
_refresh_semantic_sensors_after_course_rebuild
_try_rebuild_semantic_course_for_level
semantic_obstacle_levels
```

Add:

```python
def terrain_levels_vel_semantic_plane_gate(env, env_ids, asset_cfg=SceneEntityCfg("robot"), cfg_name="semantic_obstacle_curriculum")
```

Implementation:

```python
distance = norm(root_pos_w[env_ids, :2] - env_origins[env_ids, :2])
terrain_move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
terrain_move_down = distance < norm(command[env_ids, :2]) * env.max_episode_length_s * 0.5
terrain_move_down &= ~terrain_move_up
semantic_gate_pass = state.update_gate_from_plane_collision_rate(... )["gate_pass"]
is_plane = plane_env_mask_from_terrain(terrain.terrain_types[env_ids], terrain_names, cfg.plane_terrain_names)
move_up = where(is_plane, terrain_move_up & semantic_gate_pass, terrain_move_up)
terrain.update_env_origins(env_ids, move_up, terrain_move_down)
```

Return logging dict with:

```python
mean_terrain_level
plane_collision_rate
plane_env_count
consecutive_success_count
semantic_gate_pass
flat_move_up_count
non_flat_move_up_count
```

- [x] **Step 4: Run tests and confirm GREEN**

Run:

```bash
pytest Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py -q
```

Expected: PASS.

## Task 4: Wire Task Config To New Term And Row Defaults

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
- Modify: `Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py`
- Modify: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [x] **Step 1: Write/update cfg tests**

Assert:

- `TeacherElevationTrajectoryMpcSemanticCurriculumCfg.terrain_levels.func` is `mdp.terrain_levels_vel_semantic_plane_gate`.
- there is no `semantic_obstacle_levels` curriculum term.
- cfg `plane_counts` and `non_plane_counts` each have length 10.
- `scene.terrain` has no `semantic_obstacle_curriculum_level` attribute set by task cfg.

- [x] **Step 2: Run tests and confirm RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_semantic_trajectory_cfg_defaults_to_mpc_and_semantic_scanner -q
```

- [x] **Step 3: Implement cfg wiring**

In task cfg:

- Replace `terrain_levels = CurrTerm(func=mdp.terrain_levels_vel_unitree_rl_lab)` and separate `semantic_obstacle_levels` with:

```python
terrain_levels = CurrTerm(
    func=mdp.terrain_levels_vel_semantic_plane_gate,
    params={"cfg_name": "semantic_obstacle_curriculum"},
)
```

- Set default `plane_counts` / `non_plane_counts` to 10 entries.
- Remove `self.scene.terrain.semantic_obstacle_curriculum_level = 0`.

- [x] **Step 4: Run tests and confirm GREEN**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_semantic_trajectory_cfg_defaults_to_mpc_and_semantic_scanner -q
```

Expected: PASS.

## Task 5: Local Regression And IsaacLab Smoke

**Files:**
- Modify: `notes/log/index.md`
- Modify: `notes/todo.md`
- Create: `notes/log/YYYY-MM-DD-HHMM-t302n-row-gated-semantic-curriculum.md`

- [x] **Step 1: Run focused local tests**

Run:

```bash
pytest \
  Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py \
  Go2Pvcnn/tests/test_semantic_course_curriculum_layout.py \
  Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py \
  Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_semantic_trajectory_cfg_defaults_to_mpc_and_semantic_scanner \
  -q
```

- [x] **Step 2: Run real IsaacLab probe**

Use an idle GPU:

```bash
CUDA_VISIBLE_DEVICES=<card> /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/semantic_obstacle_curriculum_isaaclab_probe.py --num-envs 64 --row 9 --force-low-collision-steps 1 --output-json /tmp/t302n_probe_row.json --trace-json /tmp/t302n_probe_trace.json
```

Expected: update probe if needed so it validates row-based counts and absence of runtime rebuild.

- [x] **Step 3: Record evidence**

Create a log with:

- commands
- pass/fail results
- row count behavior
- flat gate behavior
- remaining risk

- [x] **Step 4: Update dashboard**

Update:

- `notes/todo.md`
- `notes/log/index.md`
