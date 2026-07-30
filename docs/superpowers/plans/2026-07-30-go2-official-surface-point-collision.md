# Go2 Official Surface Point Collision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the parallelism planner's old ellipsoid collision model with official Go2 USD collision primitives sampled by fixed surface points, while keeping active-leg-only planner behavior.

**Architecture:** The planner keeps its candidate/filter/score/standstill structure. `config.py` owns official primitive specs and surface-point density knobs, `collision.py` builds batched surface points and queries terrain/semantic maps in Torch, and viewer diagnostics consume renamed collision shape fields. `go2_foostep_planner.py` adds a small test terminal panel for speed magnitudes, swing height, standstill fallback, and three visualization toggles.

**Tech Stack:** Python 3.10, PyTorch tensor operations, IsaacLab viewer utilities, pytest.

## Global Constraints

- Do not create `extension/parallelism/official_collision_spec.py`.
- Delete old ellipsoid collision logic; do not keep compatibility branches.
- Keep current active-leg-only collision behavior.
- Do not add trot pair `50 x 50` candidate checking.
- Collision points must be batched over env, leg, candidate, time, shape, and point dimensions.
- Official contact-tolerant shapes ignore terrain-height contact but still reject semantic obstacles.
- `standstill_fallback_enabled` defaults to `True`; when `False`, invalid plans remain visibly invalid instead of being replaced with current-state hold outputs.
- `go2_foostep_planner.py` gets `--used-test-terminal`, default enabled.
- The viewer panel contains only `vx`, `vy`, `vyaw`, `swing_height`, `standstill_fallback_enabled`, `show_mesh`, `show_collision_body`, and `show_contact_points`.
- Do not touch `extension/joint_mpc_rti` or its tests.

---

## File Structure

- Modify `Go2Pvcnn/extension/parallelism/config.py`: replace `EllipsoidSpec` with official collision shape dataclasses and default Go2 primitive values.
- Modify `Go2Pvcnn/extension/parallelism/collision.py`: delete ellipsoid helpers and implement official surface-point generation plus batched terrain/semantic collision checks.
- Modify `Go2Pvcnn/extension/parallelism/types.py`: rename diagnostics from ellipsoid/probe wording to official shape/surface point wording.
- Modify `Go2Pvcnn/extension/parallelism/planner.py`: call the new collision API, use new diagnostics fields, and gate current-state fallback with `standstill_fallback_enabled`.
- Modify `Go2Pvcnn/extension/parallelism/viewer_adapter.py`: expose collision point diagnostics to the viewer result.
- Modify `Go2Pvcnn/extension/viz/go2_foostep_planner.py`: update reject formatting and add the test terminal UI/control state hooks.
- Modify tests under `Go2Pvcnn/tests/parallelism/`: replace ellipsoid tests with official collision tests, fallback tests, and viewer diagnostics tests.
- Keep the already-written spec `docs/superpowers/specs/2026-07-30-go2-official-surface-point-collision-design.html` as the source of truth.

---

### Task 1: Official Collision Config And Surface Point API

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/config.py`
- Modify: `Go2Pvcnn/extension/parallelism/collision.py`
- Test: `Go2Pvcnn/tests/parallelism/test_collision.py`

**Interfaces:**
- Produces: `OfficialCollisionShapeSpec(name: str, leg_name: str | None, link_type: str, shape_type: str, center_l: tuple[float, float, float], quat_wxyz_l: tuple[float, float, float, float], size_l: tuple[float, float, float], radius_m: float, height_m: float)`
- Produces: `build_official_surface_points_l(specs, cfg, dtype, device) -> tuple[torch.Tensor, torch.Tensor]`
- Produces: `official_collision_mask(terrain, geometry, cfg) -> tuple[torch.Tensor, torch.Tensor]`

- [ ] **Step 1: Write failing tests for official shape defaults**

Replace the old ellipsoid default test with assertions for these names:

```python
def test_default_official_collision_shapes_match_go2_usd():
    from extension.parallelism.config import ParallelismCfg

    cfg = ParallelismCfg()
    names = tuple(spec.name for spec in cfg.official_collision_shapes)
    assert names == (
        "thigh_box",
        "fl_calf_upper_cylinder",
        "calf_mid_cylinder",
        "calf_lower_cylinder",
        "foot_sphere",
    )
    assert cfg.collision_margin_m == 0.003
    assert cfg.box_surface_points == 6
    assert cfg.cylinder_layers == 1
    assert cfg.cylinder_angles == 4
    assert cfg.sphere_surface_points == 6
```

- [ ] **Step 2: Write failing tests for surface point positions**

Add a test that calls `build_official_surface_points_l()` and checks the six-point box, cylinder, and sphere samples from the design.

- [ ] **Step 3: Run red test**

Run: `PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism/test_collision.py -q`

Expected: FAIL because `official_collision_shapes` and `build_official_surface_points_l` do not exist yet.

- [ ] **Step 4: Implement config dataclasses and point builder**

Implement official shape specs in `config.py`, and implement point generation in `collision.py`. Keep points in primitive local first, then transform by each primitive's local `center_l` and `quat_wxyz_l`.

- [ ] **Step 5: Run green test**

Run: `PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism/test_collision.py -q`

Expected: PASS for config and point-shape tests; collision behavior tests may still fail until Task 2.

---

### Task 2: Batched Official Collision Mask

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/collision.py`
- Modify: `Go2Pvcnn/extension/parallelism/planner.py`
- Modify: `Go2Pvcnn/extension/parallelism/types.py`
- Test: `Go2Pvcnn/tests/parallelism/test_collision.py`
- Test: `Go2Pvcnn/tests/parallelism/test_planner.py`

**Interfaces:**
- Consumes: `build_official_surface_points_l(...)`
- Produces: `ParallelismDiagnostics.collision_shape_names: tuple[str, ...]`
- Produces: `ParallelismDiagnostics.collision_surface_point_count: int`

- [ ] **Step 1: Write failing collision behavior test**

Replace the ellipsoid height test with:

```python
def test_official_collision_uses_batched_surface_points():
    import torch
    from types import SimpleNamespace
    from extension.parallelism.collision import official_collision_mask
    from extension.parallelism.config import OfficialCollisionShapeSpec, ParallelismCfg
    from extension.parallelism.types import ParallelismTerrain

    cfg = ParallelismCfg(
        collision_margin_m=0.0,
        official_collision_shapes=(
            OfficialCollisionShapeSpec(
                name="foot_sphere",
                leg_name=None,
                link_type="foot",
                shape_type="sphere",
                center_l=(0.0, 0.0, 0.0),
                quat_wxyz_l=(1.0, 0.0, 0.0, 0.0),
                size_l=(0.0, 0.0, 0.0),
                radius_m=0.10,
                height_m=0.0,
            ),
        ),
    )
    terrain = ParallelismTerrain(
        height_w=torch.full((1, 11, 11), 0.10),
        semantic_id=torch.zeros(1, 11, 11, dtype=torch.long),
        valid_mask=torch.ones(1, 11, 11, dtype=torch.bool),
        origin_w=torch.tensor([[-0.5, -0.5, 0.0]]),
        yaw_w=torch.zeros(1),
        resolution=0.1,
    )
    geometry = SimpleNamespace(
        foot_pos_w=torch.zeros(1, 1, 1, 1, 3),
        foot_rot_w=torch.eye(3).view(1, 1, 1, 1, 3, 3),
        thigh_pos_w=torch.zeros(1, 1, 1, 1, 3),
        thigh_rot_w=torch.eye(3).view(1, 1, 1, 1, 3, 3),
        calf_pos_w=torch.zeros(1, 1, 1, 1, 3),
        calf_rot_w=torch.eye(3).view(1, 1, 1, 1, 3, 3),
    )

    ok, bits = official_collision_mask(terrain, geometry, cfg)

    assert ok.shape == (1, 1, 1)
    assert bits.shape == (1, 1, 1, 1)
    assert not bool(ok[0, 0, 0])
    assert bool(bits[0, 0, 0, 0])
```

- [ ] **Step 2: Write failing planner diagnostics test**

Update planner tests to assert `candidate_collision_bits.shape[-1] == len(cfg.official_collision_shapes)` and `diagnostics.collision_shape_names == tuple(spec.name for spec in cfg.official_collision_shapes)`.

- [ ] **Step 3: Run red tests**

Run: `PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism/test_collision.py Go2Pvcnn/tests/parallelism/test_planner.py -q`

Expected: FAIL because planner still imports `ellipsoid_collision_mask` and diagnostics still use old fields.

- [ ] **Step 4: Implement official collision API and planner wiring**

Rename the planner import to `official_collision_mask`, update `_contact_tolerant_indices()` to return an empty tensor or remove tolerant suppression, and populate `collision_shape_names` / `collision_surface_point_count`.

- [ ] **Step 5: Run green tests**

Run: `PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism/test_collision.py Go2Pvcnn/tests/parallelism/test_planner.py -q`

Expected: PASS.

---

### Task 3: Standstill Fallback Switch

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/config.py`
- Modify: `Go2Pvcnn/extension/parallelism/planner.py`
- Test: `Go2Pvcnn/tests/parallelism/test_planner.py`

**Interfaces:**
- Produces: `ParallelismCfg.standstill_fallback_enabled: bool = True`

- [ ] **Step 1: Write failing tests for both fallback modes**

Keep the existing invalid-map hold-current test for default `True`, and add:

```python
def test_invalid_plan_can_disable_standstill_fallback():
    import torch
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.planner import plan_trajectory

    state = _state()
    traj = plan_trajectory(
        state,
        torch.tensor([[0.7, 0.2, 0.3]]),
        _terrain(invalid=True),
        ParallelismCfg(standstill_fallback_enabled=False),
    )

    assert not bool(traj.valid[0])
    assert not torch.allclose(traj.root_pos_w, state.root_pos_w[:, None].expand(-1, 24, -1))
```

- [ ] **Step 2: Run red test**

Run: `PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism/test_planner.py::test_invalid_plan_can_disable_standstill_fallback -q`

Expected: FAIL because config does not accept `standstill_fallback_enabled`.

- [ ] **Step 3: Implement fallback switch**

Add config field and change the final `torch.where` fallback section so current-state replacement is gated by `selected_valid | ~cfg.standstill_fallback_enabled`.

- [ ] **Step 4: Run green test**

Run: `PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism/test_planner.py -q`

Expected: PASS.

---

### Task 4: Viewer Diagnostics And Test Terminal Panel

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/viewer_adapter.py`
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Test: `Go2Pvcnn/tests/parallelism/test_viewer_adapter.py`

**Interfaces:**
- Consumes: `ParallelismDiagnostics.collision_shape_names`
- Produces: `ViewerTestTerminalState`
- Produces: `_apply_test_terminal_command(base_command, state) -> torch.Tensor`

- [ ] **Step 1: Write failing viewer diagnostics test**

Rename the old ellipsoid formatting test to:

```python
def test_viewer_parallelism_reject_uses_collision_shape_names():
    import torch
    from types import SimpleNamespace
    from extension.viz.go2_foostep_planner import _format_parallelism_reject_diagnostics

    diagnostics = SimpleNamespace(
        candidate_reject_bits=torch.zeros(1, 4, 50, 6, dtype=torch.bool),
        candidate_valid=torch.ones(1, 4, 50, dtype=torch.bool),
        candidate_collision_bits=torch.zeros(1, 4, 50, 2, dtype=torch.bool),
        collision_shape_names=("calf_mid_cylinder", "foot_sphere"),
    )
    diagnostics.candidate_collision_bits[..., 0] = True

    text = _format_parallelism_reject_diagnostics(SimpleNamespace(parallelism_diagnostics=diagnostics))

    assert "collision_detail(calf_mid_cylinder=200 foot_sphere=0)" in text
```

- [ ] **Step 2: Write failing pure helper test for terminal command scaling**

Add a pure test that constructs `ViewerTestTerminalState(vx=0.5, vy=0.25, vyaw=0.75, enabled=True)` and asserts `_apply_test_terminal_command(torch.zeros(1, 3), state)` returns `[[0.5, 0.25, 0.75]]`.

- [ ] **Step 3: Run red tests**

Run: `PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism/test_viewer_adapter.py -q`

Expected: FAIL because new fields and helpers do not exist.

- [ ] **Step 4: Implement diagnostics rename and pure terminal state**

Update formatter to prefer `collision_shape_names`. Add `ViewerTestTerminalState` and `_apply_test_terminal_command()` as pure Python/Torch helpers before integrating UI. Add parser argument `--used-test-terminal` defaulting true plus `--no-used-test-terminal` if the parser needs an explicit off switch.

- [ ] **Step 5: Run green tests**

Run: `PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism/test_viewer_adapter.py -q`

Expected: PASS.

---

### Task 5: Viewer Marker Hooks For Collision Bodies And Points

**Files:**
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Test: `Go2Pvcnn/tests/parallelism/test_viewer_adapter.py`

**Interfaces:**
- Consumes: viewer result optional fields `parallelism_collision_surface_points_w`, `parallelism_collision_body_centers_w`
- Produces: brown point markers controlled by `show_contact_points`
- Produces: collision body markers controlled by `show_collision_body`

- [ ] **Step 1: Write failing pure visibility state test**

Add a helper test asserting `_parallelism_visualization_flags(show_mesh=True, show_collision_body=False, show_contact_points=True)` returns a simple namespace or dataclass with those booleans unchanged.

- [ ] **Step 2: Run red test**

Run: `PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism/test_viewer_adapter.py -q`

Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement marker data exposure and visibility helper**

Expose selected-plan collision surface points from diagnostics when available. Add markers in `PlannerVisualizer` using brown `(0.45, 0.24, 0.10)` spheres. Hide markers when toggles are false.

- [ ] **Step 4: Run green tests**

Run: `PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism/test_viewer_adapter.py -q`

Expected: PASS.

---

### Task 6: Verification And Real Viewer Smoke

**Files:**
- No planned source edits unless verification reveals a defect.

**Interfaces:**
- Consumes: all previous tasks.

- [ ] **Step 1: Run parallelism unit tests**

Run: `PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism -q`

Expected: PASS.

- [ ] **Step 2: Run targeted viewer import smoke**

Run: `PYTHONPATH=Go2Pvcnn python -m py_compile Go2Pvcnn/extension/viz/go2_foostep_planner.py`

Expected: PASS with no output.

- [ ] **Step 3: Run IsaacSim viewer smoke**

Run the user's viewer command with `--planner-backend parallelism`, `--used-test-terminal`, and a nonzero scripted command or panel command.

Expected log includes `valid`, `per_leg_valid`, `collision_detail`, selected touchdown information, and no traceback.

- [ ] **Step 4: Commit implementation**

Run:

```bash
git add Go2Pvcnn/extension/parallelism Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/parallelism docs/superpowers/plans/2026-07-30-go2-official-surface-point-collision.md docs/superpowers/specs/2026-07-30-go2-official-surface-point-collision-design.html
git commit -m "feat: use official go2 surface point collision"
```

Expected: commit only includes this feature and the updated design/plan docs.

---

## Self-Review

- Spec coverage: official primitive sizes, surface points, active-leg-only scope, parallel collision points, standstill fallback panel control, viewer toggles, and tests are each mapped to tasks.
- Placeholder scan: no `TBD` or unresolved optional tasks are present.
- Type consistency: plan consistently uses `OfficialCollisionShapeSpec`, `official_collision_shapes`, `official_collision_mask`, `collision_shape_names`, and `collision_surface_point_count`.
