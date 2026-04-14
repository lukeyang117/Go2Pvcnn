# Batched Planner Train Viewer Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `teacher_elevation_trajectory` training and the Isaac Lab viewer use the same `extension/batched_planner` runtime path, remove raw/placeholder runtime paths from training, align planner behavior to `raw`, and document the resulting CLI workflows.

**Architecture:** Introduce a planner-owned runtime boundary centered on a formal `PlannerTerrain` input object and a planner-owned cache manager. Move trajectory-cache ownership out of reward fallback logic, unify train and viewer on the same planner API, then tighten planner parity, dtype/device, and scaling behavior with tests before documenting commands.

**Tech Stack:** Python, Isaac Lab, PyTorch, Gymnasium, RSL-RL, pytest, git

---

## File Structure

### Runtime and Planner Files

- Modify: `Go2Pvcnn/extension/batched_planner/terrain.py`
  Responsibility: Define the planner-owned `PlannerTerrain` ABI, canonical scanner-to-terrain conversion, and stable query semantics for `height_at`, `roughness_at`, and `max_height_along_segment`.

- Modify: `Go2Pvcnn/extension/batched_planner/trajectory.py`
  Responsibility: Use the formal terrain ABI, preserve raw-aligned trajectory behavior, and remove normal-path per-env recursive fallback for batched training use.

- Modify: `Go2Pvcnn/extension/batched_planner/base_solver.py`
  Responsibility: Enforce explicit device/dtype behavior in the batched base solver and avoid multi-device surprises in body-clearance logic.

- Modify: `Go2Pvcnn/extension/batched_planner/foothold.py`
  Responsibility: Consume the formal terrain ABI consistently and preserve deterministic replan/tie-break semantics aligned to raw.

- Modify: `Go2Pvcnn/extension/batched_planner/manager.py`
  Responsibility: Own training-time cache build/update, reset-triggered replanning, command-change-triggered replanning, and interval fallback cadence.

- Modify: `Go2Pvcnn/extension/convention.py`
  Responsibility: Normalize planner input/output boundary conversion, including explicit output dtype/device normalization before cache conversion.

- Modify: `Go2Pvcnn/extension/reference/cache.py`
  Responsibility: Define the explicit cache layout and placement expectations used by train and reward code.

### Training and Reward Files

- Modify: `Go2Pvcnn/scripts/train.py`
  Responsibility: Remove `--use-raw-reference-trajectory`, delete raw runtime selection for `teacher_elevation_trajectory`, and wire train startup to the planner-owned runtime path only.

- Modify: `Go2Pvcnn/extension/mdp/rewards_reference.py`
  Responsibility: Stop lazily fabricating placeholder caches in the normal training path and consume the planner-managed cache only.

- Modify: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py`
  Responsibility: Keep trajectory env config aligned with the planner runtime assumptions and replan cadence settings used by train/viewer.

### Viewer Files

- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
  Responsibility: Call the shared planner-owned boundary and manager semantics only; remove viewer-local terrain/planner repair logic.

### Tests

- Modify: `Go2Pvcnn/tests/test_batched_terrain.py`
  Responsibility: Cover the formal `PlannerTerrain` ABI, scanner-hit conversion, invalid-hit handling, and query shape semantics.

- Modify: `Go2Pvcnn/tests/test_batched_trajectory.py`
  Responsibility: Keep raw parity checks for zero/low-speed/motion cases and add deterministic near-tie replan fixtures.

- Modify: `Go2Pvcnn/tests/test_batched_trajectory_batch.py`
  Responsibility: Verify multi-env parity without normal-path recursive behavior.

- Modify: `Go2Pvcnn/tests/test_batched_base_solver.py`
  Responsibility: Lock down dtype/device behavior and body-clearance execution on the shared device.

- Modify: `Go2Pvcnn/tests/test_batched_foothold.py`
  Responsibility: Lock down deterministic candidate scoring and tie-break rules.

- Modify: `Go2Pvcnn/tests/test_batched_manager.py`
  Responsibility: Cover reset-triggered replan, command-change-triggered replan, interval fallback, and cache lifecycle.

- Modify: `Go2Pvcnn/tests/test_batched_reference_integration.py`
  Responsibility: Verify reward-side cache consumption uses the planner-managed cache and not placeholder fallback.

- Create: `Go2Pvcnn/tests/test_batched_planner_runtime_path.py`
  Responsibility: End-to-end runtime-path checks for train/viewer boundary ownership and cache rebuild semantics.

### Notes

- Create: `notes/human/human-12-batched-planner-train-viewer-commands.md`
  Responsibility: Document `train`, `viewer`, and `play` launch commands, key parameters, and troubleshooting notes for the new runtime path.

- Modify: `notes/index.md`
  Responsibility: Add the new command/reference note into the notes entry structure.

## Task 1: Make Train and Viewer Import-Safe

**Files:**
- Modify: `Go2Pvcnn/scripts/train.py`
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Test: `Go2Pvcnn/tests/test_batched_planner_runtime_path.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_train_module_can_be_imported_without_launching_app():
    module = importlib.import_module("Go2Pvcnn.scripts.train")
    assert hasattr(module, "build_arg_parser")


def test_viewer_module_can_be_imported_without_launching_app():
    module = importlib.import_module("Go2Pvcnn.extension.viz.go2_foostep_planner")
    assert hasattr(module, "build_arg_parser")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_planner_runtime_path.py -k "imported_without_launching_app" -v`
Expected: FAIL because both modules currently parse CLI args and create `AppLauncher` at import time.

- [ ] **Step 3: Write minimal implementation**

Implement:
- `build_arg_parser()` helper in `Go2Pvcnn/scripts/train.py`
- `build_arg_parser()` helper in `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- move CLI parse and `AppLauncher` startup under `main()` or equivalent guarded startup helpers

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_planner_runtime_path.py -k "imported_without_launching_app" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/scripts/train.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_batched_planner_runtime_path.py
git commit -m "refactor: make train and viewer import safe"
```

## Task 2: Remove Train-Time Raw Runtime Selection

**Files:**
- Modify: `Go2Pvcnn/scripts/train.py`
- Test: `Go2Pvcnn/tests/test_batched_planner_runtime_path.py`

- [ ] **Step 1: Write the failing test**

```python
def test_train_cli_no_longer_accepts_raw_reference_flag():
    from Go2Pvcnn.scripts.train import build_arg_parser

    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--use-raw-reference-trajectory"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_planner_runtime_path.py::test_train_cli_no_longer_accepts_raw_reference_flag -v`
Expected: FAIL because the flag still exists or `build_arg_parser()` is not yet exposed.

- [ ] **Step 3: Write minimal implementation**

Implement in `Go2Pvcnn/scripts/train.py`:
- extract parser creation into a helper like `build_arg_parser()`
- delete `--use-raw-reference-trajectory`
- remove `_configure_reference_trajectory(... use_raw_reference_trajectory=...)` branching that still references the raw flag

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_planner_runtime_path.py::test_train_cli_no_longer_accepts_raw_reference_flag -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/scripts/train.py Go2Pvcnn/tests/test_batched_planner_runtime_path.py
git commit -m "refactor: remove raw trajectory train flag"
```

## Task 3: Define the Formal PlannerTerrain ABI

**Files:**
- Modify: `Go2Pvcnn/extension/batched_planner/terrain.py`
- Test: `Go2Pvcnn/tests/test_batched_terrain.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_planner_terrain_from_ray_hits_normalizes_batched_layout():
    ray_hits = torch.zeros((2, 16, 16, 3), dtype=torch.float32)
    terrain = PlannerTerrain.from_ray_hits(ray_hits)
    assert terrain.batch_size == 2


def test_planner_terrain_filters_invalid_hits_deterministically():
    ray_hits = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    ray_hits[0, 0, 0] = torch.tensor([float("nan"), 0.0, 0.0])
    terrain = PlannerTerrain.from_ray_hits(ray_hits)
    values = terrain.height_at(torch.tensor([[[0.0, 0.0]]], dtype=torch.float32))
    assert torch.isfinite(values).all()


def test_planner_terrain_supports_single_and_multi_query_contracts():
    ...


def test_planner_terrain_rejects_raw_tensor_callers_at_entry_boundary():
    ...


def test_planner_terrain_max_height_along_segment_contract_is_explicit():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_terrain.py -k "planner_terrain or invalid_hits" -v`
Expected: FAIL because `PlannerTerrain` and its ABI do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement in `Go2Pvcnn/extension/batched_planner/terrain.py`:
- a formal `PlannerTerrain` class
- `PlannerTerrain.from_ray_hits(...)`
- canonical range derivation from scanner hits
- explicit query-shape support for `height_at`, `roughness_at`, `max_height_along_segment`
- explicit dtype/device normalization rules
- explicit rejection of unsupported raw-heightmap direct entry-point usage

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_terrain.py -k "planner_terrain or invalid_hits" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/terrain.py Go2Pvcnn/tests/test_batched_terrain.py
git commit -m "feat: add formal planner terrain abi"
```

## Task 4: Define the Reference Cache ABI Explicitly

**Files:**
- Modify: `Go2Pvcnn/extension/reference/cache.py`
- Modify: `Go2Pvcnn/extension/convention.py`
- Modify: `Go2Pvcnn/tests/test_batched_reference_integration.py`

- [ ] **Step 1: Write the failing test**

```python
def test_reference_cache_layout_and_dtype_are_explicit():
    cache = build_reference_cache_fixture()
    assert cache.root_pos_w.dtype == torch.float32
    assert cache.root_pos_w.ndim == 3


def test_planner_result_to_reference_cache_follows_cache_abi():
    result = make_fake_result(dtype=torch.float64)
    cache = planner_result_to_reference_cache(result)
    assert cache.root_pos_w.dtype == torch.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_reference_integration.py -v`
Expected: FAIL because the cache ABI and normalization rules are not fully locked down yet.

- [ ] **Step 3: Write minimal implementation**

Implement:
- explicit cache layout and placement rules in `Go2Pvcnn/extension/reference/cache.py`
- explicit planner-result normalization in `Go2Pvcnn/extension/convention.py`
- test fixtures that assert the agreed cache ABI

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_reference_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/reference/cache.py Go2Pvcnn/extension/convention.py Go2Pvcnn/tests/test_batched_reference_integration.py
git commit -m "feat: define reference cache abi"
```

## Task 5: Remove Viewer-Local Terrain Repair

**Files:**
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Test: `Go2Pvcnn/tests/test_batched_planner_runtime_path.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_viewer_uses_planner_owned_terrain_boundary():
    import Go2Pvcnn.extension.viz.go2_foostep_planner as viewer

    assert hasattr(viewer, "PlannerTerrain")
    assert not hasattr(viewer, "SingleTerrainAdapter")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_planner_runtime_path.py::test_viewer_uses_planner_owned_terrain_boundary -v`
Expected: FAIL because the viewer still owns local adapter logic.

- [ ] **Step 3: Write minimal implementation**

Implement in `Go2Pvcnn/extension/viz/go2_foostep_planner.py`:
- import and use the planner-owned `PlannerTerrain`
- remove `SingleTerrainAdapter`
- route scanner-hit conversion through the shared planner-owned boundary

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_planner_runtime_path.py::test_viewer_uses_planner_owned_terrain_boundary -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_batched_planner_runtime_path.py
git commit -m "refactor: route viewer through planner terrain abi"
```

## Task 6: Make Planner Output and Base Solver Device Rules Explicit

**Files:**
- Modify: `Go2Pvcnn/extension/batched_planner/base_solver.py`
- Modify: `Go2Pvcnn/extension/convention.py`
- Test: `Go2Pvcnn/tests/test_batched_base_solver.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_batched_base_solver_rejects_mixed_devices_with_clear_error():
    # construct tensors on different devices if CUDA is available
    ...


def test_planner_result_to_reference_cache_normalizes_output_dtype():
    result = make_fake_result(dtype=torch.float64)
    cache = planner_result_to_reference_cache(result)
    assert cache.root_pos_w.dtype == torch.float32
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_base_solver.py -v`
Expected: FAIL because normalization/device rules are not fully explicit yet.

- [ ] **Step 3: Write minimal implementation**

Implement:
- explicit canonical runtime dtype for planner outputs before cache conversion
- explicit cache dtype/device normalization in `Go2Pvcnn/extension/convention.py`
- explicit no-hidden-sync behavior in the shared runtime path

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_base_solver.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/base_solver.py Go2Pvcnn/extension/convention.py Go2Pvcnn/tests/test_batched_base_solver.py
git commit -m "fix: normalize planner output device and dtype"
```

## Task 7: Lock Down Raw Parity for Replanning and Motion Branches

**Files:**
- Modify: `Go2Pvcnn/extension/batched_planner/trajectory.py`
- Modify: `Go2Pvcnn/extension/batched_planner/foothold.py`
- Test: `Go2Pvcnn/tests/test_batched_trajectory.py`
- Test: `Go2Pvcnn/tests/test_batched_foothold.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_near_tie_replan_matches_raw_candidate_choice():
    ...


def test_motion_branch_matches_raw_on_representative_stair_case():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_trajectory.py Go2Pvcnn/tests/test_batched_foothold.py -v`
Expected: FAIL on candidate choice and/or output mismatch.

- [ ] **Step 3: Write minimal implementation**

Implement:
- deterministic candidate ordering and tie-break semantics aligned to raw
- any motion-branch fixes needed for touchdown/contact/root/foot parity

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_trajectory.py Go2Pvcnn/tests/test_batched_foothold.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/trajectory.py Go2Pvcnn/extension/batched_planner/foothold.py Go2Pvcnn/tests/test_batched_trajectory.py Go2Pvcnn/tests/test_batched_foothold.py
git commit -m "fix: align replanning behavior with raw"
```

## Task 8: Remove the Normal-Path Per-Env Recursive Fallback

**Files:**
- Modify: `Go2Pvcnn/extension/batched_planner/trajectory.py`
- Modify: `Go2Pvcnn/tests/test_batched_trajectory_batch.py`

- [ ] **Step 1: Write the failing test**

```python
def test_batched_generate_trajectory_uses_vectorized_batch_path():
    # patch a sentinel so recursive single-env fallback would be detected
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_trajectory_batch.py -v`
Expected: FAIL because multi-env calls still recurse per env.

- [ ] **Step 3: Write minimal implementation**

Implement in `Go2Pvcnn/extension/batched_planner/trajectory.py`:
- remove the normal-path `batch_size > 1` recursion branch
- replace it with a vectorized path suitable for training/runtime use

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_trajectory_batch.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/trajectory.py Go2Pvcnn/tests/test_batched_trajectory_batch.py
git commit -m "perf: remove per-env recursive batch fallback"
```

## Task 9: Introduce Planner-Owned Cache Management for Training

**Files:**
- Modify: `Go2Pvcnn/extension/batched_planner/manager.py`
- Modify: `Go2Pvcnn/extension/mdp/rewards_reference.py`
- Modify: `Go2Pvcnn/scripts/train.py`
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Modify: `Go2Pvcnn/tests/test_batched_manager.py`
- Modify: `Go2Pvcnn/tests/test_batched_reference_integration.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_manager_replans_immediately_on_reset():
    ...


def test_manager_replans_immediately_on_command_change():
    ...


def test_rewards_reference_uses_manager_owned_cache_only():
    ...


def test_train_runtime_calls_manager_on_reset_and_step():
    ...


def test_viewer_runtime_calls_manager_on_reset_and_command_change():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_manager.py Go2Pvcnn/tests/test_batched_reference_integration.py -v`
Expected: FAIL because rewards still own placeholder fallback and the runtime call-sites do not yet encode the full lifecycle.

- [ ] **Step 3: Write minimal implementation**

Implement:
- planner-owned runtime cache lifecycle in `manager.py`
- immediate replan on reset
- immediate replan on command change
- interval fallback replanning
- concrete runtime call-sites in `Go2Pvcnn/scripts/train.py` and `Go2Pvcnn/extension/viz/go2_foostep_planner.py` that forward reset/step/command updates to the manager
- reward-side consumption of managed cache only
- no normal placeholder generation in `rewards_reference.py`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_manager.py Go2Pvcnn/tests/test_batched_reference_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/manager.py Go2Pvcnn/extension/mdp/rewards_reference.py Go2Pvcnn/scripts/train.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_batched_manager.py Go2Pvcnn/tests/test_batched_reference_integration.py
git commit -m "feat: move trajectory cache ownership into planner manager"
```

## Task 10: Wire Train to the Planner-Owned Runtime Path

**Files:**
- Modify: `Go2Pvcnn/scripts/train.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py`
- Test: `Go2Pvcnn/tests/test_batched_planner_runtime_path.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_teacher_elevation_trajectory_train_path_requires_batched_planner():
    ...


def test_train_runtime_no_longer_uses_placeholder_reference_generator():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_planner_runtime_path.py -v`
Expected: FAIL because the train path is not yet exclusively planner-owned.

- [ ] **Step 3: Write minimal implementation**

Implement:
- planner-only train path for `teacher_elevation_trajectory`
- any env-config alignment needed for the shared planner cadence/state assumptions
- removal of normal placeholder path from training startup and runtime wiring

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_planner_runtime_path.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/scripts/train.py Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py Go2Pvcnn/tests/test_batched_planner_runtime_path.py
git commit -m "feat: wire teacher trajectory train path to batched planner"
```

## Task 11: Restore Viewer Behavior on the Shared Runtime Path

**Files:**
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Test: `Go2Pvcnn/tests/test_batched_planner_runtime_path.py`

- [ ] **Step 1: Write the failing test**

```python
def test_viewer_replans_immediately_on_reset_and_command_change():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_planner_runtime_path.py::test_viewer_replans_immediately_on_reset_and_command_change -v`
Expected: FAIL because the viewer does not yet mirror the final shared runtime lifecycle.

- [ ] **Step 3: Write minimal implementation**

Implement in `Go2Pvcnn/extension/viz/go2_foostep_planner.py`:
- use the shared planner-owned manager semantics
- preserve keyboard teleop, marker drawing, and livestream behavior
- ensure motion commands lead to real replanning and touchdown updates

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_planner_runtime_path.py::test_viewer_replans_immediately_on_reset_and_command_change -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_batched_planner_runtime_path.py
git commit -m "fix: align viewer runtime with planner manager"
```

## Task 12: Add Performance and Smoke Verification

**Files:**
- Modify: `Go2Pvcnn/tests/test_batched_planner_runtime_path.py`

- [ ] **Step 1: Add train-side smoke and scaling checks**

Add tests or scripted assertions that cover:

```python
def test_runtime_path_exposes_no_recursive_batch_fallback():
    ...
```

- [ ] **Step 2: Run the focused automated tests**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_planner_runtime_path.py -v`
Expected: PASS

- [ ] **Step 3: Run the minimal headless train smoke**

Run: `conda run -n env_isaaclab python Go2Pvcnn/scripts/train.py --headless --num_envs 1 --max_iterations 1 --experiment teacher_elevation_trajectory`
Expected: reaches the planner-owned trajectory path without placeholder fallback

- [ ] **Step 4: Run the viewer startup smoke**

Run: `conda run -n env_isaaclab python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 2 --device cuda:0 --terrain mixed`
Expected: starts, accepts teleop input, and replans without the previous motion-branch crash

- [ ] **Step 5: Run a higher-env efficiency smoke**

Run: `conda run -n env_isaaclab python Go2Pvcnn/scripts/train.py --headless --num_envs 4096 --max_iterations 1 --experiment teacher_elevation_trajectory`
Expected: shared runtime path initializes without placeholder fallback and without normal-path per-env recursive planner behavior

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/tests/test_batched_planner_runtime_path.py
git commit -m "test: add planner runtime smoke and scaling checks"
```

## Task 13: Write Notes for Train Viewer Play Commands

**Files:**
- Create: `notes/human/human-12-batched-planner-train-viewer-commands.md`
- Modify: `notes/index.md`

- [ ] **Step 1: Write the note content**

Document:
- `train.py` command lines
- `go2_foostep_planner.py` command lines
- `play.py` command lines
- parameter explanations for the trajectory-related flags
- the constraint that `teacher_elevation_trajectory` is planner-only
- common failure signatures and first checks

- [ ] **Step 2: Verify links and relative paths**

Run: `python3 - <<'PY'\nfrom pathlib import Path\np = Path('notes/human/human-12-batched-planner-train-viewer-commands.md')\nprint(p.exists())\nPY`
Expected: `True`

- [ ] **Step 3: Commit**

```bash
git add notes/human/human-12-batched-planner-train-viewer-commands.md notes/index.md
git commit -m "docs: add batched planner train and viewer command guide"
```
