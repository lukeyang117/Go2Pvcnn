# Go2 Ellipsoid Collision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the parallelism planner's legacy foot/knee/calf/thigh line collision checks with configurable Go2 visual-mesh ellipsoid collision checks.

**Architecture:** Add a small collision module under `Go2Pvcnn/extension/parallelism` that owns ellipsoid config packing, probe generation, link-frame transforms, and collision reductions. Extend parallelism FK to return same-leg `*_thigh`, `*_calf`, and `*_foot` link poses derived from root pose plus candidate joint angles, then call the collision module from `planner.py`.

**Tech Stack:** Python 3.10, PyTorch tensor ops, pytest, existing `extension.parallelism` dataclasses and terrain query API.

## Global Constraints

- Modify only `extension/parallelism`, `tests/parallelism`, and allowed viewer diagnostics in `extension/viz/go2_foostep_planner.py`.
- Do not modify or reuse `extension/joint_mpc_rti` code for this task.
- Collision/filter/score logic must use torch conditions and reductions, not Python per-candidate branching.
- Collision filter only checks heightmap surface points entering leg ellipsoids; valid map, semantic touchdown, landing, joint limits, and score remain separate filters.
- Ellipsoid offsets are link-local fixed offsets: `thigh_* -> *_thigh`, `calf_* -> *_calf`, `foot_* -> *_foot`.
- For 1024 envs, avoid retaining full `[B,L,C,E,P,3]` intermediates for all ellipsoids; compute by link group or chunk and reduce immediately.
- If all candidates are invalid, root and foot must hold current state.

---

## File Structure

- Modify `Go2Pvcnn/extension/parallelism/config.py`: add frozen dataclasses for ellipsoid parameters and defaults; remove reliance on legacy capsule radii in planner.
- Modify `Go2Pvcnn/extension/parallelism/kinematics.py`: extend `Go2ParallelGeometry` with link poses for thigh/calf/foot.
- Create `Go2Pvcnn/extension/parallelism/collision.py`: pack config to tensors, build 5 local probes, transform terrain points into link frames, and return collision masks.
- Modify `Go2Pvcnn/extension/parallelism/types.py`: add collision names/probe metadata to diagnostics.
- Modify `Go2Pvcnn/extension/parallelism/planner.py`: replace `_collision_mask()` internals with ellipsoid collision module and reshape new FK fields.
- Modify `Go2Pvcnn/extension/viz/go2_foostep_planner.py`: print named ellipsoid collision diagnostics.
- Modify `Go2Pvcnn/tests/parallelism/test_planner.py`: replace legacy calf margin test and update shape/contracts.
- Modify `Go2Pvcnn/tests/parallelism/test_viewer_adapter.py`: update diagnostics construction and viewer contracts.
- Create `Go2Pvcnn/tests/parallelism/test_collision.py`: focused TDD tests for ellipsoid config, probe offsets, link-local transform, and chunked torch implementation.

---

### Task 1: Config And Diagnostics Contract

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/config.py`
- Modify: `Go2Pvcnn/extension/parallelism/types.py`
- Test: `Go2Pvcnn/tests/parallelism/test_collision.py`
- Test: `Go2Pvcnn/tests/parallelism/test_planner.py`
- Test: `Go2Pvcnn/tests/parallelism/test_viewer_adapter.py`

**Interfaces:**
- Produces: `EllipsoidSpec(name: str, link_type: str, center_l: tuple[float,float,float], radii_l: tuple[float,float,float], probe_offset_l: tuple[float,float])`
- Produces: `ParallelismCfg.collision_ellipsoids: ClassVar[tuple[EllipsoidSpec, ...]]`
- Produces: `ParallelismDiagnostics.candidate_collision_bits: Tensor` with shape `[B,4,50,E]`
- Produces: `ParallelismDiagnostics.collision_ellipsoid_names: tuple[str, ...]`
- Produces: `ParallelismDiagnostics.collision_probe_count: int`

- [ ] **Step 1: Write failing tests for ellipsoid defaults**

```python
def test_default_collision_ellipsoid_specs_are_named_and_grouped():
    from extension.parallelism.config import ParallelismCfg

    cfg = ParallelismCfg()
    names = tuple(spec.name for spec in cfg.collision_ellipsoids)
    link_types = tuple(spec.link_type for spec in cfg.collision_ellipsoids)

    assert names == (
        "thigh_body_inner",
        "thigh_body_mid",
        "thigh_body_outer",
        "thigh_outer_cap",
        "calf_knee_cap",
        "calf_upper_bar",
        "calf_mid_bar",
        "calf_lower_bar",
        "calf_ankle_cap",
        "foot_pad",
    )
    assert link_types == ("thigh", "thigh", "thigh", "thigh", "calf", "calf", "calf", "calf", "calf", "foot")
    assert cfg.collision_probe_count == 5
    assert cfg.collision_margin_m == 0.003
```

- [ ] **Step 2: Update planner contract test expectation**

Change `Go2Pvcnn/tests/parallelism/test_planner.py::test_full_flat_trajectory_contract`:

```python
assert traj.diagnostics.candidate_collision_bits.shape == (1, 4, 50, 10)
assert traj.diagnostics.collision_ellipsoid_names == tuple(spec.name for spec in ParallelismCfg.collision_ellipsoids)
assert traj.diagnostics.collision_probe_count == 5
```

Remove `test_default_calf_collision_clearance_uses_margin_only`.

- [ ] **Step 3: Update viewer adapter diagnostics constructors**

In `Go2Pvcnn/tests/parallelism/test_viewer_adapter.py`, construct diagnostics with:

```python
candidate_collision_bits=torch.zeros(1, 4, 50, 10, dtype=torch.bool),
collision_ellipsoid_names=tuple(f"ellipsoid_{idx}" for idx in range(10)),
collision_probe_count=5,
```

- [ ] **Step 4: Run tests to verify RED**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism/test_collision.py::test_default_collision_ellipsoid_specs_are_named_and_grouped Go2Pvcnn/tests/parallelism/test_planner.py::test_full_flat_trajectory_contract -q
```

Expected: fail because `EllipsoidSpec`, `collision_ellipsoids`, and diagnostics fields do not exist.

- [ ] **Step 5: Implement config/dataclass fields**

Add to `Go2Pvcnn/extension/parallelism/config.py`:

```python
@dataclass(frozen=True)
class EllipsoidSpec:
    name: str
    link_type: str
    center_l: tuple[float, float, float]
    radii_l: tuple[float, float, float]
    probe_offset_l: tuple[float, float]
```

Add `collision_probe_count: int = 5` and `collision_ellipsoids: ClassVar[tuple[EllipsoidSpec, ...]] = (...)` with the 10 values from the spec.

Add to `Go2Pvcnn/extension/parallelism/types.py`:

```python
collision_ellipsoid_names: tuple[str, ...]
collision_probe_count: int
```

- [ ] **Step 6: Run tests to verify GREEN for contract fields**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism/test_collision.py::test_default_collision_ellipsoid_specs_are_named_and_grouped -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add Go2Pvcnn/extension/parallelism/config.py Go2Pvcnn/extension/parallelism/types.py Go2Pvcnn/tests/parallelism/test_collision.py Go2Pvcnn/tests/parallelism/test_planner.py Go2Pvcnn/tests/parallelism/test_viewer_adapter.py
git commit -m "feat: add parallelism ellipsoid collision config"
```

### Task 2: FK Link Poses

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/kinematics.py`
- Test: `Go2Pvcnn/tests/parallelism/test_collision.py`

**Interfaces:**
- Produces: `Go2ParallelGeometry.thigh_pos_w`, `thigh_rot_w`, `calf_pos_w`, `calf_rot_w`, `foot_rot_w`
- Existing `Go2ParallelGeometry.foot_pos_w` remains foot link/contact position used by current planner.

- [ ] **Step 1: Write failing FK pose test**

```python
def test_fk_returns_link_poses_for_collision_frames():
    import torch
    from extension.parallelism.kinematics import fk_go2

    root_pos = torch.tensor([[1.0, 2.0, 0.3]], dtype=torch.float32)
    root_rpy = torch.zeros(1, 3)
    joint = torch.tensor([[0.0, 0.8, -1.5] * 4], dtype=torch.float32)

    geometry = fk_go2(root_pos, root_rpy, joint)

    assert geometry.thigh_pos_w.shape == (1, 4, 3)
    assert geometry.thigh_rot_w.shape == (1, 4, 3, 3)
    assert geometry.calf_pos_w.shape == (1, 4, 3)
    assert geometry.calf_rot_w.shape == (1, 4, 3, 3)
    assert geometry.foot_rot_w.shape == (1, 4, 3, 3)
    eye = torch.eye(3).expand(1, 4, 3, 3)
    assert torch.allclose(geometry.thigh_rot_w @ geometry.thigh_rot_w.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(geometry.calf_rot_w @ geometry.calf_rot_w.transpose(-1, -2), eye, atol=1e-5)
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism/test_collision.py::test_fk_returns_link_poses_for_collision_frames -q
```

Expected: fail because FK geometry lacks link pose fields.

- [ ] **Step 3: Implement FK link poses**

In `kinematics.py`, add dataclass fields and compute local rotations:

```python
thigh_body_rot = root_rot @ abad_rot @ thigh_rot
calf_body_rot = root_rot @ abad_rot @ thigh_rot @ calf_rot
foot_body_rot = calf_body_rot
thigh_pos_w = upper_world
calf_pos_w = knee_world
```

Use batched torch matrices. Keep existing foot/knee/sample outputs unchanged.

- [ ] **Step 4: Update planner reshape**

When reshaping FK geometry in `planner.py`, include:

```python
thigh_pos_w=geometry.thigh_pos_w.reshape(batch, leg_count, candidate_count, 4, 3)
thigh_rot_w=geometry.thigh_rot_w.reshape(batch, leg_count, candidate_count, 4, 3, 3)
calf_pos_w=geometry.calf_pos_w.reshape(batch, leg_count, candidate_count, 4, 3)
calf_rot_w=geometry.calf_rot_w.reshape(batch, leg_count, candidate_count, 4, 3, 3)
foot_rot_w=geometry.foot_rot_w.reshape(batch, leg_count, candidate_count, 4, 3, 3)
```

- [ ] **Step 5: Run test to verify GREEN**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism/test_collision.py::test_fk_returns_link_poses_for_collision_frames -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/parallelism/kinematics.py Go2Pvcnn/extension/parallelism/planner.py Go2Pvcnn/tests/parallelism/test_collision.py
git commit -m "feat: expose go2 link poses for collision"
```

### Task 3: Ellipsoid Collision Module

**Files:**
- Create: `Go2Pvcnn/extension/parallelism/collision.py`
- Modify: `Go2Pvcnn/extension/parallelism/planner.py`
- Test: `Go2Pvcnn/tests/parallelism/test_collision.py`
- Test: `Go2Pvcnn/tests/parallelism/test_planner.py`

**Interfaces:**
- Produces: `ellipsoid_collision_mask(terrain, geometry, cfg) -> tuple[Tensor, Tensor]`
- Return `collision_ok: [B,L,C]`, `collision_bits: [B,L,C,E]`

- [ ] **Step 1: Write failing link-local inside test**

```python
def test_ellipsoid_collision_uses_link_local_height_points():
    import torch
    from types import SimpleNamespace
    from extension.parallelism.collision import ellipsoid_collision_mask
    from extension.parallelism.config import EllipsoidSpec, ParallelismCfg
    from extension.parallelism.types import ParallelismTerrain

    cfg = ParallelismCfg()
    cfg = type(cfg)(
        collision_margin_m=0.0,
        collision_ellipsoids=(EllipsoidSpec("foot_pad", "foot", (0.0, 0.0, 0.0), (0.10, 0.10, 0.10), (0.05, 0.05)),),
    )
    terrain = ParallelismTerrain(
        height_w=torch.full((1, 11, 11), 0.05),
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

    ok, bits = ellipsoid_collision_mask(terrain, geometry, cfg)

    assert ok.shape == (1, 1, 1)
    assert bits.shape == (1, 1, 1, 1)
    assert not bool(ok[0, 0, 0])
    assert bool(bits[0, 0, 0, 0])
```

- [ ] **Step 2: Write failing probe offset test**

```python
def test_probe_offset_checks_four_neighbors_and_center():
    import torch
    from extension.parallelism.collision import build_ellipsoid_probe_l
    from extension.parallelism.config import EllipsoidSpec

    specs = (EllipsoidSpec("e", "foot", (1.0, 2.0, 3.0), (0.1, 0.2, 0.3), (0.4, 0.5)),)

    probes = build_ellipsoid_probe_l(specs, dtype=torch.float32, device=torch.device("cpu"))

    assert probes.shape == (1, 5, 3)
    assert torch.allclose(
        probes[0],
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [1.4, 2.0, 3.0],
                [0.6, 2.0, 3.0],
                [1.0, 2.5, 3.0],
                [1.0, 1.5, 3.0],
            ]
        ),
    )
```

- [ ] **Step 3: Run tests to verify RED**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism/test_collision.py::test_ellipsoid_collision_uses_link_local_height_points Go2Pvcnn/tests/parallelism/test_collision.py::test_probe_offset_checks_four_neighbors_and_center -q
```

Expected: fail because `collision.py` does not exist.

- [ ] **Step 4: Implement collision module**

Create `collision.py` with:

```python
def build_ellipsoid_probe_l(specs, *, dtype, device) -> Tensor: ...
def ellipsoid_collision_mask(terrain, geometry, cfg) -> tuple[Tensor, Tensor]: ...
```

Implementation requirements:
- Pack centers/radii/probes to tensors.
- Group specs by `link_type` using tiny Python loops over E only.
- For each group, gather same-leg link pose by active leg index.
- Compute `probe_w = link_pos + link_rot @ probe_l`.
- Query height with existing `query_height_semantic_valid`.
- Transform `terrain_p_w` back to link local.
- Compute ellipsoid equation with `radii + cfg.collision_margin_m`.
- Reduce with `torch.any` over probe and group ellipsoid dims.

- [ ] **Step 5: Replace planner collision call**

In `planner.py`, import:

```python
from extension.parallelism.collision import ellipsoid_collision_mask
```

Replace `_collision_mask()` body or call with:

```python
collision_ok, collision_bits = ellipsoid_collision_mask(terrain, geometry, cfg)
```

Pass diagnostics names/probe count into `ParallelismDiagnostics`.

- [ ] **Step 6: Run tests to verify GREEN**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism/test_collision.py Go2Pvcnn/tests/parallelism/test_planner.py -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add Go2Pvcnn/extension/parallelism/collision.py Go2Pvcnn/extension/parallelism/planner.py Go2Pvcnn/tests/parallelism/test_collision.py Go2Pvcnn/tests/parallelism/test_planner.py
git commit -m "feat: use ellipsoid collision for parallelism planner"
```

### Task 4: Viewer Named Collision Diagnostics

**Files:**
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Test: `Go2Pvcnn/tests/parallelism/test_viewer_adapter.py`

**Interfaces:**
- Consumes: `diagnostics.collision_ellipsoid_names`
- Consumes: `diagnostics.candidate_collision_bits: [B,4,50,E]`

- [ ] **Step 1: Write failing viewer formatting test**

Add to `test_viewer_adapter.py`:

```python
def test_viewer_parallelism_reject_uses_ellipsoid_names():
    import torch
    from types import SimpleNamespace
    from extension.viz.go2_foostep_planner import _format_parallelism_reject_diagnostics

    diagnostics = SimpleNamespace(
        candidate_reject_bits=torch.zeros(1, 4, 50, 6, dtype=torch.bool),
        candidate_valid=torch.ones(1, 4, 50, dtype=torch.bool),
        candidate_collision_bits=torch.zeros(1, 4, 50, 2, dtype=torch.bool),
        collision_ellipsoid_names=("calf_mid_bar", "foot_pad"),
    )
    diagnostics.candidate_collision_bits[..., 0] = True

    text = _format_parallelism_reject_diagnostics(SimpleNamespace(parallelism_diagnostics=diagnostics))

    assert "collision_detail(calf_mid_bar=200 foot_pad=0)" in text
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism/test_viewer_adapter.py::test_viewer_parallelism_reject_uses_ellipsoid_names -q
```

Expected: fail because viewer only recognizes 4 legacy collision names.

- [ ] **Step 3: Implement viewer formatting**

In `_format_parallelism_reject_diagnostics`, replace fixed 4-name branch with:

```python
collision_names = tuple(getattr(diagnostics, "collision_ellipsoid_names", ()))
if collision_t.ndim == 4 and collision_names and collision_t.shape[-1] == len(collision_names):
    ...
elif collision_t.ndim == 4 and collision_t.shape[-1] == 4:
    collision_names = ("foot", "knee", "calf", "thigh")
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism/test_viewer_adapter.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/parallelism/test_viewer_adapter.py
git commit -m "chore: print ellipsoid collision diagnostics"
```

### Task 5: Verification And IsaacLab Smoke Test

**Files:**
- No code changes unless verification exposes a bug.

**Interfaces:**
- Consumes all previous tasks.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest Go2Pvcnn/tests/parallelism -q
```

Expected: all parallelism tests pass.

- [ ] **Step 2: Run source scan for forbidden legacy collision dependence**

Run:

```bash
rg -n "calf_radius_m|thigh_radius_m|knee_radius_m|capsule_samples|collision_names = \\(\"foot\", \"knee\", \"calf\", \"thigh\"\\)" Go2Pvcnn/extension/parallelism Go2Pvcnn/extension/viz/go2_foostep_planner.py
```

Expected: no legacy radius use inside active parallelism collision path; viewer may retain legacy fallback only.

- [ ] **Step 3: Run IsaacLab viewer smoke test**

Run from repo root:

```bash
export DISPLAY=:1
export OMNI_KIT_ACCEPT_EULA=Y
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$PWD/../miniconda-placeholder:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/lib/python3.10/site-packages/torch/lib:/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/lib/python3.10/site-packages/nvidia/cudnn/lib:/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

/opt/VirtualGL/bin/vglrun -d egl \
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python \
Go2Pvcnn/extension/viz/go2_foostep_planner.py \
--device cuda:0 \
--num_envs 1 \
--terrain task \
--planner-backend parallelism \
--n-frames 30 \
--plan-dt 0.02 \
--terrain-row 0 \
--terrain-col 0 \
--warmup-steps 0 \
--key-hold-timeout 3.0
```

Expected: viewer starts. If the robot initializes floating, use the viewer grounded reset equivalent before sending velocity. With a nonzero command, diagnostics should show named ellipsoid collision detail and valid candidates on flat terrain.

- [ ] **Step 4: Final status**

Report:
- Unit test command and result.
- Whether IsaacLab smoke test ran; if it did not, report why.
- Any remaining dirty files that are unrelated and intentionally untouched.
