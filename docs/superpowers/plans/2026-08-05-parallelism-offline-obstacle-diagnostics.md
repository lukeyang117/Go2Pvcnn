# Parallelism Offline Obstacle Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a YAML-driven, pure-Torch diagnostic harness that reproduces Parallelism small-obstacle failures without starting Isaac Sim.

**Architecture:** A loader reads a scene YAML containing terrain, obstacle, root, joints, and command state. A builder converts the obstacle description into `ParallelismTerrain` tensors and a `ParallelismState`. A runner calls the existing `plan_trajectory()` unchanged and emits JSON diagnostics including obstacle geometry in world/root frames, joint/root state, per-leg validity, rejection bits, and collision-shape counts.

**Tech Stack:** Python 3.10, PyTorch, PyYAML, pytest, existing `extension.parallelism` APIs.

## Global Constraints

- Do not modify `extension/parallelism` planner behavior.
- Do not start Isaac Sim for the offline probe or its tests.
- Keep all tensor decisions and planner execution on one Torch device.
- Preserve the user’s existing dirty `Go2Pvcnn/scripts/train_parallelism_flat_rl_headless_resume.sh`.
- YAML scenes must support future obstacle positions, shapes, heights, root poses, joint states, and commands.

---

### Task 1: Define the YAML scene and loader

**Files:**
- Create: `Go2Pvcnn/tests/parallelism/offline_obstacle_scenes.yaml`
- Create: `Go2Pvcnn/tests/parallelism/offline_obstacle_diagnostics.py`
- Test: `Go2Pvcnn/tests/parallelism/test_offline_obstacle_diagnostics.py`

**Interfaces:**
- `load_scene(path: Path, scene_name: str) -> OfflineObstacleScene`
- `OfflineObstacleScene` contains resolution, origin, yaw, grid size, obstacle description, root pose, joints, and command.
- YAML must include the deterministic failure case with center `[0.20, 0.00]`, radius `0.12`, height `0.20`, and commands `0.8` and `1.0`.

- [ ] **Step 1: Write failing loader tests**

```python
def test_yaml_scene_loads_failure_case():
    scene = load_scene(SCENE_FILE, "front_center_high_small")
    assert scene.obstacle.center_w == pytest.approx((0.20, 0.0))
    assert scene.obstacle.radius_m == pytest.approx(0.12)
    assert scene.obstacle.height_m == pytest.approx(0.20)
    assert scene.root.rpy_w == pytest.approx((0.0, 0.0, 0.0))
    assert scene.joint_pos.shape == (12,)
```

- [ ] **Step 2: Run the focused test and verify it fails because the loader is absent**

```bash
pytest Go2Pvcnn/tests/parallelism/test_offline_obstacle_diagnostics.py::test_yaml_scene_loads_failure_case -q
```

- [ ] **Step 3: Add the YAML schema and dataclasses**

The scene file must represent:

```yaml
scenes:
  front_center_high_small:
    terrain:
      resolution_m: 0.01
      size: 151
      origin_w: [-0.75, -0.75, 0.0]
      yaw_w: 0.0
    obstacle:
      shape: circle
      center_w: [0.20, 0.00]
      radius_m: 0.12
      height_m: 0.20
      semantic_id: 1
    root:
      position_w: [0.0, 0.0, 0.50]
      rpy_w: [0.0, 0.0, 0.0]
    joint_pos: [0.0, 0.8, -1.6, 0.0, 0.8, -1.6,
                0.0, 0.8, -1.6, 0.0, 0.8, -1.6]
    commands:
      - [0.8, 0.0, 0.0]
      - [1.0, 0.0, 0.0]
```

Use `yaml.safe_load`, validate shapes and positive resolution/grid size, and raise `ValueError` with field names for malformed scenes.

- [ ] **Step 4: Run the focused loader tests and verify they pass**

```bash
pytest Go2Pvcnn/tests/parallelism/test_offline_obstacle_diagnostics.py::test_yaml_scene_loads_failure_case -q
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/tests/parallelism/offline_obstacle_scenes.yaml \
        Go2Pvcnn/tests/parallelism/offline_obstacle_diagnostics.py \
        Go2Pvcnn/tests/parallelism/test_offline_obstacle_diagnostics.py
git commit -m "test: add yaml scenes for offline parallelism diagnostics"
```

### Task 2: Build deterministic terrain and robot state

**Files:**
- Modify: `Go2Pvcnn/tests/parallelism/offline_obstacle_diagnostics.py`
- Test: `Go2Pvcnn/tests/parallelism/test_offline_obstacle_diagnostics.py`

**Interfaces:**
- `build_terrain(scene: OfflineObstacleScene, device: torch.device) -> ParallelismTerrain`
- `build_state(scene: OfflineObstacleScene, device: torch.device) -> ParallelismState`
- `obstacle_geometry(scene, terrain) -> dict[str, Tensor]`

- [ ] **Step 1: Add failing terrain/state tests**

```python
def test_scene_builds_known_semantic_and_height_map():
    terrain = build_terrain(scene, torch.device("cuda:0"))
    center = torch.tensor([[0.20, 0.0]], device=terrain.height_w.device)
    query = query_height_semantic_valid(terrain, center)
    assert query.semantic.item() == 1
    assert query.height.item() == pytest.approx(0.20)
```

- [ ] **Step 2: Implement circle and cuboid rasterization**

Rasterize obstacle cells at the configured resolution. Set terrain cells to height 0 outside the obstacle and the configured obstacle height inside it. Preserve semantic id and valid mask. Return obstacle min/max/center in world coordinates and transform them into the root frame using inverse root yaw.

- [ ] **Step 3: Implement state construction**

Construct `ParallelismState` from YAML root pose and 12 planner-order joints. Compute current foot positions with `fk_go2()` and store them in the state so the offline run matches the real planner adapter’s state contract.

- [ ] **Step 4: Run terrain/state tests**

```bash
pytest Go2Pvcnn/tests/parallelism/test_offline_obstacle_diagnostics.py -k "semantic_and_height or state" -q
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/tests/parallelism/offline_obstacle_diagnostics.py \
        Go2Pvcnn/tests/parallelism/test_offline_obstacle_diagnostics.py
git commit -m "test: build deterministic parallelism terrain and state"
```

### Task 3: Run planner and emit failure diagnostics

**Files:**
- Modify: `Go2Pvcnn/tests/parallelism/offline_obstacle_diagnostics.py`
- Modify: `Go2Pvcnn/tests/parallelism/test_offline_obstacle_diagnostics.py`

**Interfaces:**
- `run_scene(scene, command_index=0, device=None) -> dict[str, object]`
- `run_yaml(path, scene_name=None, device=None) -> list[dict[str, object]]`
- CLI: `python Go2Pvcnn/tests/parallelism/offline_obstacle_diagnostics.py --scene-file ... --scene ... --output ...`

- [ ] **Step 1: Add failing reproduction assertions**

```python
def test_front_center_obstacle_reproduces_parallelism_standstill():
    report = run_scene(load_scene(SCENE_FILE, "front_center_high_small"), command_index=0)
    assert report["standstill"] is True
    assert report["per_leg_valid"] == [0, 40, 0, 40]
    assert report["obstacle_center_root"] == pytest.approx([0.20, 0.0])
```

- [ ] **Step 2: Implement planner execution without changing planner code**

Call:

```python
trajectory = plan_trajectory(
    state,
    command[None],
    terrain,
    ParallelismCfg(),
)
```

Compute `standstill` from the returned root rollout, summarize reject bits by the existing six names, summarize collision bits by `trajectory.diagnostics.collision_shape_names`, and serialize tensors with `.detach().cpu().tolist()`.

- [ ] **Step 3: Add CLI and JSON output**

The CLI must support:

```bash
python .../offline_obstacle_diagnostics.py \
  --scene-file .../offline_obstacle_scenes.yaml \
  --scene front_center_high_small \
  --device cuda:0 \
  --output /tmp/parallelism_offline_report.json
```

- [ ] **Step 4: Run the deterministic reproduction**

```bash
PYTHONPATH=Go2Pvcnn \
python Go2Pvcnn/tests/parallelism/offline_obstacle_diagnostics.py \
  --scene-file Go2Pvcnn/tests/parallelism/offline_obstacle_scenes.yaml \
  --scene front_center_high_small \
  --device cuda:0
```

Expected core result:

```text
standstill=True
per_leg_valid=[0, 40, 0, 40]
```

- [ ] **Step 5: Run focused tests and commit**

```bash
pytest Go2Pvcnn/tests/parallelism/test_offline_obstacle_diagnostics.py -q
git add Go2Pvcnn/tests/parallelism/offline_obstacle_diagnostics.py \
        Go2Pvcnn/tests/parallelism/test_offline_obstacle_diagnostics.py
git commit -m "test: diagnose parallelism obstacle candidate failures offline"
```

### Task 4: Add reusable optimization sweep

**Files:**
- Modify: `Go2Pvcnn/tests/parallelism/offline_obstacle_diagnostics.py`
- Modify: `Go2Pvcnn/tests/parallelism/offline_obstacle_scenes.yaml`
- Modify: `Go2Pvcnn/tests/parallelism/test_offline_obstacle_diagnostics.py`

**Interfaces:**
- `sweep_scene(scene, parameter_grid, device=None) -> list[dict[str, object]]`

- [ ] **Step 1: Add a test for position/pose reuse**

```python
def test_scene_sweep_accepts_different_obstacle_positions_and_root_yaw():
    reports = sweep_scene(scene, {
        "obstacle.center_w": [[0.20, 0.0], [0.20, 0.12]],
        "root.rpy_w": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]],
    })
    assert len(reports) == 4
    assert all("obstacle_center_root" in report for report in reports)
```

- [ ] **Step 2: Implement parameter overrides**

Support overrides for obstacle center, radius, height, root position, root rpy, joints, command, `candidate_radius_m`, `candidates_per_leg`, `swing_clearance_m`, and `semantic_touchdown_margin_m`. Keep the base YAML immutable and produce a report row per combination.

- [ ] **Step 3: Add baseline and optimization examples to YAML**

Include a second scene with obstacle center `[0.20, 0.12]` and a small sweep section for `swing_clearance_m` and `candidate_radius_m`.

- [ ] **Step 4: Run all offline tests and commit**

```bash
pytest Go2Pvcnn/tests/parallelism/test_offline_obstacle_diagnostics.py -q
git add Go2Pvcnn/tests/parallelism/offline_obstacle_diagnostics.py \
        Go2Pvcnn/tests/parallelism/offline_obstacle_scenes.yaml \
        Go2Pvcnn/tests/parallelism/test_offline_obstacle_diagnostics.py
git commit -m "test: add reusable parallelism obstacle parameter sweeps"
```
