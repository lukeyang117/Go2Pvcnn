# T302 MPC Body/Leg Height-Field Collision Safety

## Current State

- T302 is a new design branch related to [T300e](T300e-mpc-continuous-swing-window-plan.md).
- Purpose: add body/leg/foot height-field collision safety, semantic touchdown/stance obstacle rejection, and high-obstacle speed/yaw risk scaling to the active `batch_mpc_planner` MPC backend.
- Written design: [../../docs/superpowers/specs/2026-05-16-mpc-body-leg-height-field-collision-safety-design.md](../../docs/superpowers/specs/2026-05-16-mpc-body-leg-height-field-collision-safety-design.md)
- Status: design recorded; subagent requirements-coverage review found no P0 gaps; implementation plan is now written below.
- Subagent coverage review found no P0 gaps; P1 clarifications were folded into the spec for semantic-small classification, all-direction scanner-mask risk detection, and T300e regression baseline reuse.
- Implementation has not started. The current gate is choosing an execution mode for the written TDD plan.

## Open Children

| Child | Status | Priority | Purpose | Primary Files |
| --- | --- | --- | --- | --- |
| T302a | done | P0 | User spec review gate and subagent requirement coverage review are clean | `docs/superpowers/specs/2026-05-16-mpc-body-leg-height-field-collision-safety-design.md`, this page |
| T302b | verify | P0 | TDD implementation plan is written in this branch page | `Go2Pvcnn/tests/`, `Go2Pvcnn/extension/batch_mpc_planner/` |
| T302c | pending | P0 | Implement GPU kinematics outputs for knee/shank world samples without adding production files | `Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py` |
| T302d | pending | P0 | Implement height-field collision and semantic touchdown/stance losses | `terrain_clearance.py`, `registry.py`, `terrain.py`, `config.py` |
| T302e | pending | P0 | Implement high-small/large command corridor and yaw-swept risk scaling for tracking losses | `tracking.py`, `registry.py`, `planner.py`, `config.py` |
| T302f | pending | P0 | Add headless `env_isaacsim` acceptance for COBBLESTONE and flat semantic obstacles while preserving T300e metrics | `Go2Pvcnn/tests/` |

## Closed Children Archive

- None yet.

## Related Logs

- [../log/2026-05-16-2200-t302-mpc-body-leg-collision-design.md](../log/2026-05-16-2200-t302-mpc-body-leg-collision-design.md)
- [../log/2026-05-16-2231-t302-implementation-plan.md](../log/2026-05-16-2231-t302-implementation-plan.md)
- T300e baseline acceptance: [../log/2026-05-15-2001-mpc-contact-support-touchdown-anchor-acceptance.md](../log/2026-05-15-2001-mpc-contact-support-touchdown-anchor-acceptance.md)

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: `pending`
- Current Work Ref: `working tree on top of 3843555`
- Key Files:
  - [../../docs/superpowers/specs/2026-05-16-mpc-body-leg-height-field-collision-safety-design.md](../../docs/superpowers/specs/2026-05-16-mpc-body-leg-height-field-collision-safety-design.md)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py](../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py)
  - [../../Go2Pvcnn/tests](../../Go2Pvcnn/tests)

## Next Step

- Choose implementation mode and execute the plan below with TDD. Recommended: subagent-driven execution with one worker per independent slice and main-agent integration review.

## T302 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GPU-batched MPC body/leg height-field collision safety, semantic touchdown/stance obstacle rejection, and high-obstacle speed/yaw risk scaling while preserving T300e behavior.

**Architecture:** Extend the existing `Go2Pvcnn/extension/batch_mpc_planner` backend in place. The implementation reuses current scanner terrain helpers and IK/FK passes, adds batched collision/risk losses, and reports diagnostics through existing loss breakdown/result paths. Runtime code remains GPU tensor based; test files may be added under `Go2Pvcnn/tests/`, while production code only modifies existing files.

**Tech Stack:** PyTorch GPU tensors, existing `MpcPlannerTerrain` `height_at`/`semantic_at`, existing Go2 MPC IK/FK helpers, pytest backend tests, IsaacLab headless runtime under `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`.

---

### File Structure And Ownership

- Modify [../../Go2Pvcnn/extension/batch_mpc_planner/kinematics.py](../../Go2Pvcnn/extension/batch_mpc_planner/kinematics.py)
  - Add FK helper that returns foot, knee, and shank sample world positions from planned root pose and joint angles.
  - Keep the existing `fk_feet_from_joint_angles` contract by delegating to the richer helper or preserving its output.
- Modify [../../Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py)
  - Add one shared kinematics evaluation helper for losses so IK is not solved twice inside T302 collision terms.
- Modify [../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py)
  - Add height-field clearance losses for root/body, knee/shank, swing foot, and semantic touchdown/stance penalties.
  - Add all-scanner obstacle risk-mask helpers for high-small/large detection.
- Modify [../../Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py)
  - Let command tracking accept per-env linear/yaw scale tensors.
- Modify [../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py)
  - Wire new loss terms and expose diagnostics in `breakdown`.
- Modify [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - Add loss config dataclasses and task-cfg overrides for margins, semantic ids, threshold `0.3m`, and risk scale `0.5`.
- Modify [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - Add result `cost_breakdown` diagnostics for collision and risk scaling while preserving existing trajectory tensors.
- Modify [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - Expose MPC `loss_breakdown` through `ViewerTrajectoryResult` and support a `cobblestone` fixture terrain option.
- Modify [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
  - Add deterministic headless fixture switches for `cobblestone` terrain and test-controlled semantic small-obstacle height.
- Modify [../../Go2Pvcnn/extension/semantic_course.py](../../Go2Pvcnn/extension/semantic_course.py)
  - Add scale-profile overrides used by tests to make semantic-small obstacles low or high without new production files.
- Modify [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_semantic_viewer_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_semantic_viewer_env_cfg.py)
  - Pass semantic-course scale overrides through the existing prestartup event params.
- Add [../../Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py](../../Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py)
  - Real IsaacLab headless acceptance for COBBLESTONE and flat semantic obstacle cases.
- Modify [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - Backend TDD coverage for kinematics, loss terms, config overrides, registry breakdown, and GPU shape contracts.

### Task T302c.1: GPU FK Returns Knee And Shank Samples

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/kinematics.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Add the failing backend test imports**

Add these imports to `Go2Pvcnn/tests/test_batch_mpc_backend.py`:

```python
from extension.batch_mpc_planner.kinematics import (
    fk_feet_from_joint_angles,
    fk_leg_points_from_joint_angles,
)
```

Expected initial failure after adding the test in Step 2: `ImportError` or `AttributeError` because `fk_leg_points_from_joint_angles` does not exist.

- [ ] **Step 2: Add failing test `test_mpc_fk_leg_points_exposes_knee_and_shank_samples`**

Add this test near existing IK/FK tests:

```python
def test_mpc_fk_leg_points_exposes_knee_and_shank_samples() -> None:
    root = torch.zeros((2, 3, 3), dtype=torch.float32)
    root[..., 2] = 0.30
    rpy = torch.zeros_like(root)
    joint = torch.zeros((2, 3, 12), dtype=torch.float32)

    leg_points = fk_leg_points_from_joint_angles(root, rpy, joint, shank_sample_count=2)

    assert leg_points.foot_pos_world.shape == (2, 3, 4, 3)
    assert leg_points.knee_pos_world.shape == (2, 3, 4, 3)
    assert leg_points.shank_sample_world.shape == (2, 3, 4, 2, 3)
    torch.testing.assert_close(
        leg_points.foot_pos_world,
        fk_feet_from_joint_angles(root, rpy, joint),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert torch.isfinite(leg_points.knee_pos_world).all()
    assert torch.isfinite(leg_points.shank_sample_world).all()
```

- [ ] **Step 3: Run the failing test**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_fk_leg_points_exposes_knee_and_shank_samples -q
```

Expected: FAIL because `fk_leg_points_from_joint_angles` is missing.

- [ ] **Step 4: Implement `MpcLegPoints` and `fk_leg_points_from_joint_angles`**

In `Go2Pvcnn/extension/batch_mpc_planner/kinematics.py`, add:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class MpcLegPoints:
    foot_pos_world: Tensor
    knee_pos_world: Tensor
    shank_sample_world: Tensor
```

Add a function with this contract:

```python
def fk_leg_points_from_joint_angles(
    root_pos: Tensor,
    root_rpy: Tensor,
    joint_angles: Tensor,
    *,
    shank_sample_count: int = 2,
) -> MpcLegPoints:
    """Return foot, knee, and shank sample world points from planner-order joints."""
```

Implementation rules:

- Reuse the same leg-angle math already in `fk_feet_from_joint_angles`.
- Compute `knee_body` at the thigh end.
- Compute `foot_body` exactly as current FK computes it.
- Compute shank samples with evenly spaced `alpha` values between knee and foot, excluding endpoints:

```python
alpha = torch.linspace(
    0.0,
    1.0,
    steps=int(shank_sample_count) + 2,
    dtype=root_pos.dtype,
    device=root_pos.device,
)[1:-1]
shank_body = knee_body.unsqueeze(-2) * (1.0 - alpha.view(1, 1, 1, -1, 1)) + foot_body.unsqueeze(-2) * alpha.view(1, 1, 1, -1, 1)
```

- Transform body points to world with `_rpy_to_rot_matrix(root_rpy)` using `torch.einsum`.
- Keep `fk_feet_from_joint_angles` returning only `leg_points.foot_pos_world`.
- Export `MpcLegPoints` and `fk_leg_points_from_joint_angles` in `__all__`.

- [ ] **Step 5: Add shared loss kinematics helper**

In `Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py`, add:

```python
from dataclasses import dataclass
from ..kinematics import MpcLegPoints, fk_leg_points_from_joint_angles


@dataclass(frozen=True)
class MpcKinematicsForLoss:
    joint_angles: Tensor
    leg_points: MpcLegPoints


def evaluate_kinematics_for_loss(
    root_pos: Tensor,
    root_rpy: Tensor,
    foot_pos: Tensor,
    *,
    clamp_to_limits: bool,
    shank_sample_count: int,
) -> MpcKinematicsForLoss:
    joint_angles = solve_joint_angles_from_trajectory(
        root_pos,
        root_rpy,
        foot_pos,
        clamp_to_limits=bool(clamp_to_limits),
    )
    leg_points = fk_leg_points_from_joint_angles(
        root_pos,
        root_rpy,
        joint_angles,
        shank_sample_count=int(shank_sample_count),
    )
    return MpcKinematicsForLoss(joint_angles=joint_angles, leg_points=leg_points)
```

Export `MpcKinematicsForLoss` and `evaluate_kinematics_for_loss`.

- [ ] **Step 6: Run the focused test**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_fk_leg_points_exposes_knee_and_shank_samples -q
```

Expected: PASS.

- [ ] **Step 7: Commit the kinematics slice**

```bash
git add \
  Go2Pvcnn/extension/batch_mpc_planner/kinematics.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "feat: expose mpc knee and shank fk samples"
```

### Task T302d.1: Height-Field Collision Losses

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/config.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Add failing imports for new losses**

Add to the terrain-clearance import block in `test_batch_mpc_backend.py`:

```python
body_heightfield_collision_loss,
knee_shank_heightfield_collision_loss,
stance_semantic_obstacle_loss,
```

Expected after Step 2: import failure because functions do not exist.

- [ ] **Step 2: Add failing test `test_mpc_heightfield_collision_losses_penalize_body_knee_and_shank`**

```python
def test_mpc_heightfield_collision_losses_penalize_body_knee_and_shank() -> None:
    height = torch.zeros((1, 5, 5), dtype=torch.float32)
    height[:, 2, 2] = 0.20
    terrain = MpcPlannerTerrain(
        height_map=height,
        semantic_map=torch.zeros((1, 5, 5), dtype=torch.long),
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    root = torch.zeros((1, 2, 3), dtype=torch.float32)
    rpy = torch.zeros_like(root)
    root[..., 2] = 0.22
    knee = torch.zeros((1, 2, 4, 3), dtype=torch.float32)
    knee[..., 2] = 0.21
    shank = knee.unsqueeze(-2).expand(1, 2, 4, 2, 3).clone()

    body_loss = body_heightfield_collision_loss(
        terrain,
        root,
        rpy,
        bottom_offset_z=-0.18,
        margin_m=0.04,
        stencil_xy=((0.0, 0.0),),
    )
    leg_loss = knee_shank_heightfield_collision_loss(
        terrain,
        knee,
        shank,
        knee_margin_m=0.04,
        shank_margin_m=0.04,
    )

    assert body_loss.shape == (1,)
    assert leg_loss.shape == (1,)
    assert float(body_loss[0]) > 0.0
    assert float(leg_loss[0]) > 0.0
```

- [ ] **Step 3: Run the failing collision test**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_heightfield_collision_losses_penalize_body_knee_and_shank -q
```

Expected: FAIL because new loss functions are missing.

- [ ] **Step 4: Implement body and leg collision losses**

In `terrain_clearance.py`, add:

```python
def body_heightfield_collision_loss(
    terrain: MpcPlannerTerrain,
    root_pos: Tensor,
    root_rpy: Tensor,
    *,
    bottom_offset_z: float,
    margin_m: float,
    stencil_xy: tuple[tuple[float, float], ...],
) -> Tensor:
```

Implementation rules:

- Convert `stencil_xy` to a tensor on `root_pos.device`.
- Rotate offsets by planned yaw `root_rpy[..., 2]`.
- Build body sample xyz as root xy plus rotated offsets and z `root_pos[..., 2] + bottom_offset_z`.
- Query `height_at(terrain, sample_xy)`.
- Return per-env mean squared clearance deficit with shape `[B]`.

Add:

```python
def knee_shank_heightfield_collision_loss(
    terrain: MpcPlannerTerrain,
    knee_pos_world: Tensor,
    shank_sample_world: Tensor,
    *,
    knee_margin_m: float,
    shank_margin_m: float,
) -> Tensor:
```

Implementation rules:

- Query `height_at` on knee xy and shank xy.
- Penalize `relu(terrain_z + margin - point_z)^2`.
- Mean over time, leg, and shank sample dimensions.
- Return shape `[B]`.

- [ ] **Step 5: Add config dataclass and defaults**

In `config.py`, add:

```python
@dataclass
class MpcBodyCollisionLossCfg(MpcLossTermCfg):
    bottom_offset_z_m: float = -0.18
    margin_m: float = 0.04
    stencil_xy_m: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.18, 0.0),
        (-0.18, 0.0),
        (0.0, 0.10),
        (0.0, -0.10),
    )


@dataclass
class MpcLegCollisionLossCfg(MpcLossTermCfg):
    knee_margin_m: float = 0.04
    shank_margin_m: float = 0.04
    shank_sample_count: int = 2
```

Add fields in `MpcLossesCfg`:

```python
body_collision: MpcBodyCollisionLossCfg = field(default_factory=lambda: MpcBodyCollisionLossCfg(enabled=True, weight=2.0))
leg_collision: MpcLegCollisionLossCfg = field(default_factory=lambda: MpcLegCollisionLossCfg(enabled=True, weight=2.0))
```

Add task-cfg overrides:

```python
_override_loss_term(task_cfg, prefix="mpc_loss_body_collision", loss_term=losses.body_collision)
_set_if_has(task_cfg, "mpc_loss_body_collision_bottom_offset_z_m", float, losses.body_collision, "bottom_offset_z_m")
_set_if_has(task_cfg, "mpc_loss_body_collision_margin_m", float, losses.body_collision, "margin_m")
_override_loss_term(task_cfg, prefix="mpc_loss_leg_collision", loss_term=losses.leg_collision)
_set_if_has(task_cfg, "mpc_loss_leg_collision_knee_margin_m", float, losses.leg_collision, "knee_margin_m")
_set_if_has(task_cfg, "mpc_loss_leg_collision_shank_margin_m", float, losses.leg_collision, "shank_margin_m")
_set_if_has(task_cfg, "mpc_loss_leg_collision_shank_sample_count", int, losses.leg_collision, "shank_sample_count")
```

- [ ] **Step 6: Wire collision losses into registry**

In `registry.py`:

- Call `evaluate_kinematics_for_loss(..., clamp_to_limits=True, shank_sample_count=losses.leg_collision.shank_sample_count)` once before joint-limit / IK-FK / collision loss terms.
- Use its `leg_points.knee_pos_world` and `leg_points.shank_sample_world` for `knee_shank_heightfield_collision_loss`.
- Add `_weighted` terms named:

```text
body_collision
leg_collision
```

- Keep `joint_limit_loss_from_root_foot` and `ik_fk_residual_loss` behavior unchanged until a later cleanup task proves they can safely share the same solved joint tensors.

- [ ] **Step 7: Run focused collision tests**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_heightfield_collision_losses_penalize_body_knee_and_shank \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_loss_breakdown_exposes_continuous_window_terms -q
```

Expected: PASS after updating the breakdown test to include `body_collision` and `leg_collision`.

- [ ] **Step 8: Commit the collision-loss slice**

```bash
git add \
  Go2Pvcnn/extension/batch_mpc_planner/config.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "feat: add mpc heightfield body leg collision losses"
```

### Task T302d.2: Touchdown And Stance Semantic Obstacle Loss

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/config.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Add failing test `test_mpc_stance_semantic_loss_penalizes_obstacle_contact_frames`**

```python
def test_mpc_stance_semantic_loss_penalizes_obstacle_contact_frames() -> None:
    height = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic = torch.zeros((1, 5, 5), dtype=torch.long)
    semantic[:, 2, 2] = 1
    semantic[:, 2, 3] = 2
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-1.0, 1.0), world_y_range=(-1.0, 1.0))
    foot = torch.tensor(
        [[
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
        ]],
        dtype=torch.float32,
    )
    contact = torch.zeros((1, 2, 4), dtype=torch.float32)
    contact[:, :, 0] = 1.0
    contact[:, :, 1] = 1.0

    loss = stance_semantic_obstacle_loss(
        terrain,
        foot,
        contact,
        ground_ids=(0,),
        small_ids=(1,),
        large_ids=(2,),
        small_weight=10.0,
        large_weight=50.0,
    )

    assert loss.shape == (1,)
    assert float(loss[0]) > 10.0
```

- [ ] **Step 2: Run the failing stance semantic test**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_stance_semantic_loss_penalizes_obstacle_contact_frames -q
```

Expected: FAIL because `stance_semantic_obstacle_loss` does not exist.

- [ ] **Step 3: Implement `stance_semantic_obstacle_loss`**

In `terrain_clearance.py`, add:

```python
def stance_semantic_obstacle_loss(
    terrain: MpcPlannerTerrain,
    foot_pos: Tensor,
    contact_prob: Tensor,
    *,
    ground_ids: tuple[int, ...],
    small_ids: tuple[int, ...],
    large_ids: tuple[int, ...],
    small_weight: float,
    large_weight: float,
) -> Tensor:
```

Implementation rules:

- Use `semantic_at(terrain, foot_pos[..., :2])`.
- Build small/large masks with vectorized equality against configured ids.
- Weight by `contact_prob`.
- Normalize by active contact mass `torch.clamp(contact_prob.sum(dim=(1, 2)), min=1.0)`.
- Return shape `[B]`.
- Do not use height difference to decide whether stance is on an obstacle.

- [ ] **Step 4: Add config ids and override support**

In `MpcTouchdownSemanticLossCfg`, add:

```python
ground_ids: tuple[int, ...] = (0,)
small_ids: tuple[int, ...] = (1,)
large_ids: tuple[int, ...] = (2,)
```

If the implementation needs a separate stance config, add:

```python
@dataclass
class MpcStanceSemanticLossCfg(MpcTouchdownSemanticLossCfg):
    pass
```

and:

```python
stance_semantic: MpcStanceSemanticLossCfg = field(default_factory=lambda: MpcStanceSemanticLossCfg(enabled=True, weight=2.0))
```

For tuple overrides, add helper:

```python
def _tuple_ints_if_has(cfg, attr: str, target, target_attr: str) -> None:
    value = getattr(cfg, attr, None)
    if value is not None:
        setattr(target, target_attr, tuple(int(v) for v in value))
```

Wire overrides for `mpc_loss_touchdown_semantic_*_ids` and `mpc_loss_stance_semantic_*_ids`.

- [ ] **Step 5: Wire stance semantic into registry**

In `registry.py`, add a weighted term after `touchdown_semantic`:

```text
stance_semantic
```

Use `decoded.contact_prob` and `decoded.foot_pos`; this covers all stance frames, including post-touchdown stance frames.

- [ ] **Step 6: Update registry breakdown test**

In `test_mpc_loss_breakdown_exposes_continuous_window_terms`, add:

```python
"stance_semantic",
"body_collision",
"leg_collision",
```

Expected: the breakdown includes all new terms when diagnostics are enabled.

- [ ] **Step 7: Run focused semantic tests**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_touchdown_semantic_loss_penalizes_small_and_large_obstacles \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_stance_semantic_loss_penalizes_obstacle_contact_frames \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_loss_breakdown_exposes_continuous_window_terms -q
```

Expected: PASS.

- [ ] **Step 8: Commit semantic stance slice**

```bash
git add \
  Go2Pvcnn/extension/batch_mpc_planner/config.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "feat: penalize mpc stance on semantic obstacles"
```

### Task T302e.1: High-Obstacle Linear/Yaw Risk Scaling

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/config.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Add failing imports for risk helper**

Add to `test_batch_mpc_backend.py`:

```python
from extension.batch_mpc_planner.losses.terrain_clearance import obstacle_risk_scales
```

Expected after Step 2: import failure.

- [ ] **Step 2: Add failing test `test_mpc_obstacle_risk_scales_use_all_scanner_obstacle_cells`**

```python
def test_mpc_obstacle_risk_scales_use_all_scanner_obstacle_cells() -> None:
    height = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic = torch.zeros((1, 5, 5), dtype=torch.long)
    height[:, 2, 4] = 0.45
    semantic[:, 2, 4] = 2
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-1.0, 1.0), world_y_range=(-1.0, 1.0))
    root = torch.zeros((1, 4, 3), dtype=torch.float32)
    root[..., 2] = 0.30
    rpy = torch.zeros_like(root)
    command = torch.tensor([[0.4, 0.0, 0.0]], dtype=torch.float32)

    scales = obstacle_risk_scales(
        terrain,
        root,
        rpy,
        command,
        small_ids=(1,),
        large_ids=(2,),
        high_small_relative_height_m=0.30,
        linear_corridor_width_m=0.35,
        linear_forward_distance_m=1.0,
        yaw_swept_radius_m=0.35,
        linear_scale_when_blocked=0.5,
        yaw_scale_when_blocked=0.5,
        linear_speed_eps=1.0e-4,
        yaw_speed_eps=1.0e-4,
    )

    assert scales.linear_scale.shape == (1,)
    assert scales.yaw_scale.shape == (1,)
    assert float(scales.linear_scale[0]) == pytest.approx(0.5)
    assert int(scales.linear_trigger_count[0]) > 0
    assert int(scales.trigger_semantic_class[0]) == 2
```

- [ ] **Step 3: Add failing yaw-only test**

```python
def test_mpc_obstacle_risk_scales_handle_yaw_only_swept_region() -> None:
    height = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic = torch.zeros((1, 5, 5), dtype=torch.long)
    height[:, 2, 3] = 0.45
    semantic[:, 2, 3] = 2
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-1.0, 1.0), world_y_range=(-1.0, 1.0))
    root = torch.zeros((1, 4, 3), dtype=torch.float32)
    root[..., 2] = 0.30
    rpy = torch.zeros_like(root)
    command = torch.tensor([[0.0, 0.0, 0.4]], dtype=torch.float32)

    scales = obstacle_risk_scales(
        terrain,
        root,
        rpy,
        command,
        small_ids=(1,),
        large_ids=(2,),
        high_small_relative_height_m=0.30,
        linear_corridor_width_m=0.35,
        linear_forward_distance_m=1.0,
        yaw_swept_radius_m=0.60,
        linear_scale_when_blocked=0.5,
        yaw_scale_when_blocked=0.5,
        linear_speed_eps=1.0e-4,
        yaw_speed_eps=1.0e-4,
    )

    assert float(scales.linear_scale[0]) == pytest.approx(1.0)
    assert float(scales.yaw_scale[0]) == pytest.approx(0.5)
    assert int(scales.yaw_trigger_count[0]) > 0
```

- [ ] **Step 4: Run failing risk tests**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_obstacle_risk_scales_use_all_scanner_obstacle_cells \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_obstacle_risk_scales_handle_yaw_only_swept_region -q
```

Expected: FAIL because `obstacle_risk_scales` is missing.

- [ ] **Step 5: Implement risk scale dataclass and helper**

In `terrain_clearance.py`, add:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ObstacleRiskScales:
    linear_scale: Tensor
    yaw_scale: Tensor
    linear_trigger_count: Tensor
    yaw_trigger_count: Tensor
    trigger_horizon_index: Tensor
    trigger_semantic_class: Tensor
```

Implement:

```python
def obstacle_risk_scales(
    terrain: MpcPlannerTerrain,
    root_pos: Tensor,
    root_rpy: Tensor,
    command: Tensor,
    *,
    small_ids: tuple[int, ...],
    large_ids: tuple[int, ...],
    high_small_relative_height_m: float,
    linear_corridor_width_m: float,
    linear_forward_distance_m: float,
    yaw_swept_radius_m: float,
    linear_scale_when_blocked: float,
    yaw_scale_when_blocked: float,
    linear_speed_eps: float,
    yaw_speed_eps: float,
) -> ObstacleRiskScales:
```

Implementation rules:

- Use all scanner grid cells from `terrain.height_map` and `terrain.semantic_map`.
- If `semantic_map is None`, return scales of `1.0` and zero trigger counts.
- Build local grid xy from `world_x_range/world_y_range`; if `sensor_pos_w`/`sensor_yaw` exist, transform local grid cells into world xy for each env.
- Classify small cells only from configured small semantic ids; split high small by `height - height_at(root_xy) > high_small_relative_height_m`.
- Classify large cells from configured large ids.
- Compute translation corridor using root frame command `[Vx,Vy]` rotated by root yaw at frame 0.
- Compute yaw-only/mixed yaw risk using distance from root xy to obstacle cells within `yaw_swept_radius_m`.
- Return per-env scales and trigger counts without `.cpu()` or `.item()`.

- [ ] **Step 6: Let `command_tracking_loss` accept per-env scales**

Change signature in `tracking.py`:

```python
def command_tracking_loss(
    root_pos: Tensor,
    root_rpy: Tensor,
    command: Tensor,
    dt: float,
    *,
    vel_weight: float = 1.0,
    yaw_weight: float = 1.0,
    vel_scale: Tensor | None = None,
    yaw_scale: Tensor | None = None,
) -> Tensor:
```

Implementation:

```python
vel_term = torch.linalg.vector_norm(xy_err, dim=-1).mean(dim=1)
yaw_term = torch.abs(yaw_err).mean(dim=1)
if vel_scale is not None:
    vel_term = vel_term * vel_scale.to(dtype=root_pos.dtype, device=root_pos.device)
if yaw_scale is not None:
    yaw_term = yaw_term * yaw_scale.to(dtype=root_pos.dtype, device=root_pos.device)
return float(vel_weight) * vel_term + float(yaw_weight) * yaw_term
```

Existing tests must still pass when scales are `None`.

- [ ] **Step 7: Add config for risk scaling**

In `config.py`, add:

```python
@dataclass
class MpcObstacleRiskCfg(MpcLossTermCfg):
    high_small_relative_height_m: float = 0.30
    linear_corridor_width_m: float = 0.35
    linear_forward_distance_m: float = 1.0
    yaw_swept_radius_m: float = 0.60
    linear_scale_when_blocked: float = 0.5
    yaw_scale_when_blocked: float = 0.5
    linear_speed_eps: float = 1.0e-4
    yaw_speed_eps: float = 1.0e-4
```

Add to `MpcLossesCfg`:

```python
obstacle_risk: MpcObstacleRiskCfg = field(default_factory=lambda: MpcObstacleRiskCfg(enabled=True, weight=1.0))
```

Wire overrides for all fields using `_set_if_has`.

- [ ] **Step 8: Wire risk scaling into registry**

In `registry.py`:

- Compute `risk = obstacle_risk_scales(...)` before `command_tracking_loss`.
- Pass `vel_scale=risk.linear_scale` and `yaw_scale=risk.yaw_scale` when `losses.obstacle_risk.enabled`; otherwise pass `None`.
- Add non-loss diagnostics into `breakdown`:

```text
obstacle_risk_linear_scale
obstacle_risk_yaw_scale
obstacle_risk_linear_trigger_count
obstacle_risk_yaw_trigger_count
obstacle_risk_trigger_horizon_index
obstacle_risk_trigger_semantic_class
```

Use tensors shaped `[B]`; do not multiply them into `per_env` with `_weighted`.

- [ ] **Step 9: Run risk and tracking tests**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_tracking_loss_honors_velocity_and_yaw_weights \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_obstacle_risk_scales_use_all_scanner_obstacle_cells \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_obstacle_risk_scales_handle_yaw_only_swept_region \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_loss_breakdown_exposes_continuous_window_terms -q
```

Expected: PASS.

- [ ] **Step 10: Commit risk-scaling slice**

```bash
git add \
  Go2Pvcnn/extension/batch_mpc_planner/config.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "feat: scale mpc tracking near high obstacles"
```

### Task T302d.3: Config Overrides And Static Guardrails

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/config.py`
- Modify: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Extend config override test**

In `test_mpc_planner_cfg_from_task_cfg_applies_runtime_and_loss_overrides`, add task cfg overrides:

```python
mpc_loss_body_collision_weight=2.7,
mpc_loss_body_collision_margin_m=0.055,
mpc_loss_leg_collision_weight=3.1,
mpc_loss_leg_collision_knee_margin_m=0.045,
mpc_loss_leg_collision_shank_sample_count=2,
mpc_loss_stance_semantic_weight=2.2,
mpc_loss_touchdown_semantic_small_ids=(1,),
mpc_loss_touchdown_semantic_large_ids=(2,),
mpc_loss_obstacle_risk_high_small_relative_height_m=0.30,
mpc_loss_obstacle_risk_linear_scale_when_blocked=0.5,
mpc_loss_obstacle_risk_yaw_scale_when_blocked=0.5,
```

Add assertions:

```python
assert cfg.losses.body_collision.weight == pytest.approx(2.7)
assert cfg.losses.body_collision.margin_m == pytest.approx(0.055)
assert cfg.losses.leg_collision.weight == pytest.approx(3.1)
assert cfg.losses.leg_collision.knee_margin_m == pytest.approx(0.045)
assert cfg.losses.leg_collision.shank_sample_count == 2
assert cfg.losses.stance_semantic.weight == pytest.approx(2.2)
assert cfg.losses.touchdown_semantic.small_ids == (1,)
assert cfg.losses.touchdown_semantic.large_ids == (2,)
assert cfg.losses.obstacle_risk.high_small_relative_height_m == pytest.approx(0.30)
assert cfg.losses.obstacle_risk.linear_scale_when_blocked == pytest.approx(0.5)
assert cfg.losses.obstacle_risk.yaw_scale_when_blocked == pytest.approx(0.5)
```

- [ ] **Step 2: Add static guardrail test for CPU hot-path patterns**

Add:

```python
def test_mpc_t302_losses_do_not_introduce_cpu_hot_path_patterns() -> None:
    root = GO2PVCNN_ROOT / "extension" / "batch_mpc_planner"
    files = [
        root / "kinematics.py",
        root / "losses" / "kinematics.py",
        root / "losses" / "terrain_clearance.py",
        root / "losses" / "tracking.py",
        root / "losses" / "registry.py",
        root / "planner.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in files)

    forbidden = [
        ".cpu().numpy(",
        ".numpy()",
        "for env_id in range(",
        "for batch_idx in range(",
    ]
    for token in forbidden:
        assert token not in source
```

- [ ] **Step 3: Run config and guardrail tests**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_planner_cfg_from_task_cfg_applies_runtime_and_loss_overrides \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_t302_losses_do_not_introduce_cpu_hot_path_patterns -q
```

Expected: PASS.

- [ ] **Step 4: Commit config guardrails**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "test: guard mpc t302 config and gpu hot path"
```

### Task T302f.1: Backend Regression Suite

**Files:**
- Modify: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Run full backend suite**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Expected: all tests pass. The latest T300e baseline was `43 passed`; the new total should include the added T302 tests.

- [ ] **Step 2: Run targeted py_compile**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/extension/batch_mpc_planner/config.py \
  Go2Pvcnn/extension/batch_mpc_planner/kinematics.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py \
  Go2Pvcnn/extension/batch_mpc_planner/planner.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py
```

Expected: exit code `0`.

- [ ] **Step 3: Run CUDA backend path when available**

```bash
MPC_TEST_DEVICE=cuda:2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_plan_segment_cuda_path_when_available -q
```

Expected: PASS or pytest skip if CUDA is unavailable.

- [ ] **Step 4: Commit verification-only note updates**

After logs are created, commit notes/log updates separately:

```bash
git add notes/todo.md notes/todo/T302-mpc-body-leg-height-field-collision-safety.md notes/log/index.md notes/log/
git commit -m "docs: record t302 backend verification"
```

### Task T302f.2: Headless COBBLESTONE Complex-Terrain Acceptance

**Files:**
- Add: `Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py`
- Modify: `Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py`
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`

- [ ] **Step 1: Add the new headless test file skeleton**

Create `Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py` with imports and skip gates:

```python
from __future__ import annotations

import os

import pytest
import torch

from Go2Pvcnn.tests.fixtures import viewer_runtime_diagnostics as viewer_diag


def _device() -> str:
    return os.environ.get("MPC_TEST_DEVICE", "cuda:0")


def _enable_collision_headless() -> bool:
    return os.environ.get("MPC_T302_HEADLESS", "0") == "1"


pytestmark = pytest.mark.skipif(
    not _enable_collision_headless(),
    reason="Set MPC_T302_HEADLESS=1 to run T302 IsaacLab headless collision acceptance.",
)
```

- [ ] **Step 2: Add COBBLESTONE fixture support**

In `Go2Pvcnn/extension/viz/go2_foostep_planner.py`, extend the parser terrain choices:

```python
parser.add_argument(
    "--terrain",
    type=str,
    default="task",
    choices=["task", "cobblestone"],
    help="Use semantic task terrain or the teacher_without_semantic COBBLESTONE_ROAD_CFG terrain.",
)
```

In `_build_env_cfg`, select a non-semantic trajectory config for `cobblestone` and override the terrain generator with the exact `teacher_without_semantic_env_cfg.py` constant:

```python
def _build_env_cfg(args_cli: argparse.Namespace):
    if str(getattr(args_cli, "terrain", "task")) == "cobblestone":
        from go2_pvcnn.tasks.teacher_elevation_trajectory_env_cfg import TeacherElevationTrajectoryEnvCfg_PLAY
        from go2_pvcnn.tasks.teacher_without_semantic_env_cfg import COBBLESTONE_ROAD_CFG

        env_cfg = TeacherElevationTrajectoryEnvCfg_PLAY()
        env_cfg.scene.terrain.terrain_generator = copy.deepcopy(COBBLESTONE_ROAD_CFG)
    else:
        from go2_pvcnn.tasks.teacher_elevation_trajectory_semantic_viewer_env_cfg import (
            TeacherElevationTrajectorySemanticViewerEnvCfg_PLAY,
        )

        env_cfg = TeacherElevationTrajectorySemanticViewerEnvCfg_PLAY()
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.scene.env_spacing = 6.0
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.events.push_robot = None
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.ranges = env_cfg.commands.base_velocity.limit_ranges
    env_cfg.planner_backend = str(args_cli.planner_backend)
    env_cfg.reference_trajectory_horizon = int(args_cli.n_frames)
    env_cfg.reference_replan_interval_steps = int(args_cli.n_frames)
    env_cfg.plan_dt = float(args_cli.plan_dt)
    reset_base = env_cfg.events.reset_base
    reset_base.params["pose_range"]["x"] = (0.0, 0.0)
    reset_base.params["pose_range"]["y"] = (0.0, 0.0)
    reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
    return env_cfg
```

In `Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py`, keep `terrain` passed through `args_cli`. Add this exact post-build clamp before `gym.make` so headless startup remains compact:

```python
terrain_gen = getattr(getattr(self.env_cfg.scene, "terrain", None), "terrain_generator", None)
if self.terrain == "cobblestone" and terrain_gen is not None:
    terrain_gen.num_rows = 2
    terrain_gen.num_cols = 1
    terrain_gen.curriculum = False
```

- [ ] **Step 3: Add helper to compute planned clearance metrics**

In the same test file, add:

```python
def _height_at(terrain, points_xy: torch.Tensor) -> torch.Tensor:
    from extension.batch_mpc_planner.terrain import height_at

    return height_at(terrain, points_xy).to(dtype=points_xy.dtype, device=points_xy.device)


def _planned_collision_metrics(result, terrain, viewer_module) -> dict[str, float]:
    from extension.batch_mpc_planner.kinematics import fk_leg_points_from_joint_angles

    root = torch.as_tensor(result.root_pos_w, dtype=torch.float32).contiguous()
    quat = torch.as_tensor(result.root_quat_w, dtype=torch.float32, device=root.device).contiguous()
    rpy = viewer_module._quat_wxyz_to_rpy(quat.reshape(-1, 4)).to(dtype=root.dtype, device=root.device).reshape_as(root)
    joints = torch.as_tensor(result.joint_angles, dtype=torch.float32, device=root.device).contiguous()
    foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float32, device=root.device).contiguous()
    contact = torch.as_tensor(result.contact_state, dtype=torch.bool, device=root.device)
    swing = torch.logical_not(contact)
    leg_points = fk_leg_points_from_joint_angles(root, rpy, joints, shank_sample_count=2)
    foot_clearance = foot[..., 2] - _height_at(terrain, foot[..., :2])
    knee_clearance = leg_points.knee_pos_world[..., 2] - _height_at(terrain, leg_points.knee_pos_world[..., :2])
    shank_clearance = leg_points.shank_sample_world[..., 2] - _height_at(terrain, leg_points.shank_sample_world[..., :2])
    root_bottom = root.clone()
    root_bottom[..., 2] = root_bottom[..., 2] - 0.18
    root_bottom_clearance = root_bottom[..., 2] - _height_at(terrain, root_bottom[..., :2])
    swing_mass = torch.clamp(swing.to(dtype=torch.float32).sum(), min=1.0)
    return {
        "root_bottom_min_clearance": float(root_bottom_clearance.min().item()),
        "swing_foot_min_clearance": float(foot_clearance[swing].min().item()) if bool(swing.any().item()) else 1.0,
        "knee_min_clearance": float(knee_clearance.min().item()),
        "shank_min_clearance": float(shank_clearance.min().item()),
        "swing_foot_collision_ratio": float(((foot_clearance < 0.0) & swing).to(torch.float32).sum().item() / float(swing_mass.item())),
        "knee_collision_ratio": float((knee_clearance < 0.0).to(torch.float32).mean().item()),
        "shank_collision_ratio": float((shank_clearance < 0.0).to(torch.float32).mean().item()),
        "joint_finite": float(torch.isfinite(joints).all().item()),
    }
```

- [ ] **Step 4: Add COBBLESTONE acceptance test**

```python
def test_t302_cobblestone_mpc_headless_collision_metrics() -> None:
    runtime = viewer_diag.make_real_runtime_fixture(
        num_envs=2,
        planner_backend="mpc",
        terrain="cobblestone",
        device=_device(),
        warmup_steps=6,
    )
    try:
        commands = ("forward", "backward", "lateral_left", "lateral_right", "yaw_left", "yaw_right")
        rows: list[dict[str, float]] = []
        for name in commands:
            plan = runtime.plan_case(name)
            terrain = runtime._single_env_terrain()
            rows.append(_planned_collision_metrics(plan.result, terrain, runtime._viewer))

        assert rows
        assert max(row["swing_foot_collision_ratio"] for row in rows) <= 0.02
        assert max(row["knee_collision_ratio"] for row in rows) <= 0.02
        assert max(row["shank_collision_ratio"] for row in rows) <= 0.02
        assert min(row["root_bottom_min_clearance"] for row in rows) > -0.02
        assert min(row["joint_finite"] for row in rows) == 1.0
    finally:
        runtime.close()
```

- [ ] **Step 5: Run the headless COBBLESTONE test**

```bash
MPC_T302_HEADLESS=1 MPC_TEST_DEVICE=cuda:2 \
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py::test_t302_cobblestone_mpc_headless_collision_metrics -q
```

Expected: PASS.

- [ ] **Step 6: Commit headless COBBLESTONE test**

```bash
git add \
  Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py \
  Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py
git commit -m "test: add mpc cobblestone collision headless acceptance"
```

### Task T302f.3: Headless Flat Semantic Obstacle Acceptance

**Files:**
- Modify: `Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py`
- Modify: `Go2Pvcnn/extension/semantic_course.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_semantic_viewer_env_cfg.py`
- Modify: `Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py`
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`

- [ ] **Step 1: Add semantic sampling helper**

Add to `test_mpc_body_leg_collision_headless.py`:

```python
def _semantic_at(terrain, points_xy: torch.Tensor) -> torch.Tensor:
    from extension.batch_mpc_planner.terrain import semantic_at

    return semantic_at(terrain, points_xy)


def _stance_semantic_count(result, terrain) -> int:
    foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float32)
    contact = torch.as_tensor(result.contact_state, dtype=torch.bool)
    sem = _semantic_at(terrain, foot[..., :2])
    obstacle = torch.logical_and(contact, sem > 0)
    return int(obstacle.to(torch.int64).sum().item())
```

- [ ] **Step 2: Add deterministic semantic small-height override support**

In `Go2Pvcnn/extension/semantic_course.py`, change `semantic_scale_profile` to accept overrides:

```python
def semantic_scale_profile(
    semantic_class: str,
    *,
    scale_profile_overrides: dict[str, tuple[float, float]] | None = None,
) -> tuple[float, float]:
    if scale_profile_overrides is not None and semantic_class in scale_profile_overrides:
        diameter, height = scale_profile_overrides[semantic_class]
        return float(diameter), float(height)
    if semantic_class == "small":
        return SMALL_OBSTACLE_DIAMETER, SMALL_OBSTACLE_HEIGHT
    if semantic_class == "large":
        return LARGE_OBSTACLE_DIAMETER, LARGE_OBSTACLE_HEIGHT
    raise ValueError(f"Unsupported semantic class {semantic_class!r}.")
```

Thread the same `scale_profile_overrides` parameter through `build_course_anchors(...)` and `spawn_semantic_course_prestartup(...)`; replace each call site with:

```python
target_diameter, target_height = semantic_scale_profile(
    semantic_class,
    scale_profile_overrides=scale_profile_overrides,
)
```

In `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_semantic_viewer_env_cfg.py`, keep the event params mutable:

```python
params={
    "default_stage": DEFAULT_VIEWER_REPRESENTATIVE_STAGE.value,
    "scale_profile_overrides": None,
},
```

In `Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py`, add `semantic_small_height_m: float | None = None` to `RealViewerRuntimeFixture.__init__`. After `self.env_cfg = self._viewer._build_env_cfg(args_cli)`, add:

```python
if semantic_small_height_m is not None:
    from extension.semantic_course import SMALL_OBSTACLE_DIAMETER

    event = getattr(self.env_cfg.events, "generate_semantic_course", None)
    if event is None:
        raise RuntimeError("semantic_small_height_m requires semantic-course event support")
    event.params["scale_profile_overrides"] = {
        "small": (float(SMALL_OBSTACLE_DIAMETER), float(semantic_small_height_m)),
    }
```

- [ ] **Step 3: Expose MPC loss breakdown in viewer results**

In `Go2Pvcnn/extension/viz/go2_foostep_planner.py`, add this field to `ViewerTrajectoryResult`:

```python
loss_breakdown: dict[str, torch.Tensor] | None = None
```

In `_adapt_mpc_result_for_viewer`, copy the planner loss diagnostics:

```python
loss_breakdown = getattr(result, "loss_breakdown", None)
if loss_breakdown is not None:
    loss_breakdown = {
        str(name): torch.as_tensor(value, device=root_pos_w.device).detach().contiguous()
        for name, value in loss_breakdown.items()
    }
```

Pass it into `ViewerTrajectoryResult(loss_breakdown=loss_breakdown, ...)`.

- [ ] **Step 4: Add low-small crossing test**

```python
def test_t302_low_small_obstacle_crosses_without_stance_on_obstacle() -> None:
    runtime = viewer_diag.make_real_runtime_fixture(
        num_envs=2,
        planner_backend="mpc",
        device=_device(),
        warmup_steps=6,
    )
    try:
        plan = runtime.plan_case_near_s4_anchor_command_relative(
            "small",
            command_name="forward",
            longitudinal_offset_m=-0.35,
            lateral_offset_m=0.0,
            z_clearance=0.65,
        )
        terrain = runtime._single_env_terrain()
        root = torch.as_tensor(plan.result.root_pos_w, dtype=torch.float32)
        anchor = runtime.s4_semantic_course_anchor("small")
        obstacle_xy = torch.tensor(anchor.world_xy, dtype=torch.float32, device=root.device)
        start_side = root[0, 0, 0] - obstacle_xy[0]
        end_side = root[0, -1, 0] - obstacle_xy[0]
        min_dist = torch.linalg.vector_norm(root[0, :, :2] - obstacle_xy, dim=-1).min()

        assert float(start_side * end_side) < 0.0
        assert float(min_dist.item()) < 0.35
        assert _stance_semantic_count(plan.result, terrain) == 0
    finally:
        runtime.close()
```

This test uses the existing small obstacle height (`SMALL_OBSTACLE_HEIGHT = 0.16m`), which is below the `0.3m` crossing threshold.

- [ ] **Step 5: Add large avoidance / yaw-risk test**

```python
def test_t302_large_obstacle_avoids_or_scales_tracking_near_yaw() -> None:
    runtime = viewer_diag.make_real_runtime_fixture(
        num_envs=2,
        planner_backend="mpc",
        device=_device(),
        warmup_steps=6,
    )
    try:
        runtime.mpc_planner_cfg.diagnostics.enabled = True
        plan = runtime.plan_case_near_s4_anchor_command_relative(
            "large",
            command_name="yaw_left",
            longitudinal_offset_m=0.0,
            lateral_offset_m=0.25,
            z_clearance=0.65,
        )
        terrain = runtime._single_env_terrain()
        breakdown = plan.result.loss_breakdown or {}
        assert _stance_semantic_count(plan.result, terrain) == 0
        assert "obstacle_risk_yaw_scale" in breakdown
        assert float(torch.as_tensor(breakdown["obstacle_risk_yaw_scale"]).min().item()) <= 0.5
    finally:
        runtime.close()
```

- [ ] **Step 6: Add high-small avoidance test**

```python
def test_t302_high_small_obstacle_avoids_and_scales_linear_tracking() -> None:
    runtime = viewer_diag.make_real_runtime_fixture(
        num_envs=2,
        planner_backend="mpc",
        device=_device(),
        warmup_steps=6,
        semantic_small_height_m=0.42,
    )
    try:
        runtime.mpc_planner_cfg.diagnostics.enabled = True
        plan = runtime.plan_case_near_s4_anchor_command_relative(
            "small",
            command_name="forward",
            longitudinal_offset_m=-0.35,
            lateral_offset_m=0.0,
            z_clearance=0.65,
        )
        breakdown = plan.result.loss_breakdown or {}
        assert _stance_semantic_count(plan.result, runtime._single_env_terrain()) == 0
        assert "obstacle_risk_linear_scale" in breakdown
        assert float(torch.as_tensor(breakdown["obstacle_risk_linear_scale"]).min().item()) <= 0.5
    finally:
        runtime.close()
```

- [ ] **Step 7: Run semantic obstacle headless tests**

```bash
MPC_T302_HEADLESS=1 MPC_TEST_DEVICE=cuda:2 \
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py::test_t302_low_small_obstacle_crosses_without_stance_on_obstacle \
  Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py::test_t302_large_obstacle_avoids_or_scales_tracking_near_yaw \
  Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py::test_t302_high_small_obstacle_avoids_and_scales_linear_tracking -q
```

Expected: PASS after T302 losses are implemented and tuned.

- [ ] **Step 8: Commit semantic headless tests**

```bash
git add \
  Go2Pvcnn/extension/semantic_course.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_semantic_viewer_env_cfg.py \
  Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py \
  Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py
git commit -m "test: add mpc semantic obstacle collision acceptance"
```

### Task T302f.4: Final T300e/T302 Verification And Notes

**Files:**
- Modify: `notes/todo.md`
- Modify: `notes/todo/T302-mpc-body-leg-height-field-collision-safety.md`
- Modify: `notes/log/index.md`
- Add: one verification log under `notes/log/`

- [ ] **Step 1: Run full backend verification**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Expected: all backend tests pass.

- [ ] **Step 2: Run T300e command-matrix regression**

```bash
MPC_TEST_DEVICE=cuda:2 \
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_runtime_command_matrix_tracks_motion_and_limits_drift -q
```

Expected: PASS. This preserves the latest accepted T300e gait/grounding baseline.

- [ ] **Step 3: Run T300e root-cause acceptance subset**

```bash
MPC_TEST_DEVICE=cuda:2 MPC_ROOT_CAUSE_CYCLES=4 \
MPC_ROOT_CAUSE_SEQUENCES='backward_speeds:backward_slow,backward,backward_fast;mixed_yaw:forward_yaw_right,lateral_left_yaw_right,lateral_right_yaw_left' \
MPC_ROOT_CAUSE_VARIANTS='default:8.0' \
MPC_ROOT_CAUSE_OUTPUT=/tmp/t302_t300e_regression_probe.jsonl \
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_root_cause_probe.py
```

Expected: exit code `0`; `backward_fast` and mixed-yaw stance airborne metrics remain clean relative to [../log/2026-05-15-2001-mpc-contact-support-touchdown-anchor-acceptance.md](../log/2026-05-15-2001-mpc-contact-support-touchdown-anchor-acceptance.md).

- [ ] **Step 4: Run T302 headless acceptance**

```bash
MPC_T302_HEADLESS=1 MPC_TEST_DEVICE=cuda:2 \
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py -q
```

Expected: PASS.

- [ ] **Step 5: Run compile and diff checks**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/extension/batch_mpc_planner/config.py \
  Go2Pvcnn/extension/batch_mpc_planner/kinematics.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py \
  Go2Pvcnn/extension/batch_mpc_planner/planner.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py
git diff --check
```

Expected: both commands exit `0`.

- [ ] **Step 6: Create final verification log**

Create `notes/log/YYYY-MM-DD-HHMM-t302-mpc-body-leg-collision-acceptance.md` with:

```text
purpose
stage
related todo
commands
input conditions
T300e regression metrics
T302 collision metrics
result
conclusion
follow-up
git refs
```

- [ ] **Step 7: Update todo/log dashboard**

Update:

- `notes/todo.md`
- `notes/todo/T302-mpc-body-leg-height-field-collision-safety.md`
- `notes/log/index.md`

Record final status and any remaining open child with its exact failing metric and next command.

- [ ] **Step 8: Commit final notes**

```bash
git add notes/todo.md notes/todo/T302-mpc-body-leg-height-field-collision-safety.md notes/log/index.md notes/log/
git commit -m "docs: record t302 mpc collision acceptance"
```

### Plan Self-Review

- Spec coverage: T302c covers knee/shank FK, T302d covers height-field and semantic stance/touchdown losses, T302e covers high-small/large linear/yaw risk scaling, T302f covers COBBLESTONE and flat semantic obstacle headless acceptance, and final verification preserves T300e baseline.
- Placeholder scan: no implementation step depends on an undefined future file outside allowed `Go2Pvcnn/tests/`; production changes modify existing files only.
- Type consistency: kinematics outputs use `MpcLegPoints`; risk outputs use `ObstacleRiskScales`; registry diagnostics use `[B]` tensors in `breakdown`.
- Execution risk: high-small height control is part of T302f.3 and must be implemented through existing-file scale-profile overrides before semantic headless acceptance is considered complete.

## Node Details

### T302a Design/Spec Review Gate

- why-created: user wants the MPC redesign to include body/leg collision safety, low-small crossing, high-small/large avoidance, real IsaacLab headless tests, GPU-only implementation, TDD flow, and no loss of T300e behavior.
- design basis:
  - collisions for root/body/knee/shank/swing foot use height map clearance;
  - touchdown and stance use semantic ids, with ground `0` allowed and obstacle ids such as `1/2` penalized;
  - low small obstacles are crossable when obstacle top is within `0.3m` of root-projected ground height;
  - high small and large obstacles can reduce linear/yaw tracking weight when they affect command direction or yaw swept region;
  - tests must include `COBBLESTONE_ROAD_CFG` and flat semantic obstacle scenes;
  - all runtime planner logic must stay GPU-batched.
- evidence:
  - design spec written at [../../docs/superpowers/specs/2026-05-16-mpc-body-leg-height-field-collision-safety-design.md](../../docs/superpowers/specs/2026-05-16-mpc-body-leg-height-field-collision-safety-design.md)
  - subagent review: no P0 gaps; P1 clarifications integrated.
- next:
  - request user review.

### T302b TDD Implementation Plan

- why-created: implementation must follow TDD and preserve future RL throughput.
- plan contents:
  - failing backend tests for knee/shank kinematics outputs;
  - failing loss tests for height-field collision and semantic stance/touchdown penalties;
  - failing tests for high-obstacle linear/yaw risk scaling;
  - headless `env_isaacsim` probes for COBBLESTONE and semantic obstacle scenes;
  - runtime metrics and vectorization guardrails.
- status: plan is written in `## T302 Implementation Plan`; implementation has not started.

### T302c GPU Kinematics For Knee/Shank

- why-created: knee/shank collisions need future-horizon positions, so they must come from planned `root + foot` IK/FK, not IsaacLab current link poses.
- constraints:
  - no duplicate IK pass;
  - no CPU geometry loop;
  - outputs remain batched `[B,T,4,3]` and `[B,T,4,K,3]`.

### T302d Height-Field And Semantic Collision Losses

- why-created: current T300e covers foot/terrain grounding but not body/knee/shank swept collisions.
- constraints:
  - height map for root/body/knee/shank/swing-foot clearance;
  - semantic map for touchdown and stance obstacle rejection;
  - no privileged obstacle prim positions in planner runtime.

### T302e High-Obstacle Tracking Weight Scaling

- why-created: high small obstacles and large obstacles should let MPC reduce speed/yaw tracking pressure rather than hard-follow commands into collisions.
- constraints:
  - translation corridor handles nonzero `[Vx,Vy]`;
  - yaw swept region handles yaw-only and mixed-yaw commands;
  - scale tracking losses inside optimization, not after trajectory generation.

### T302f Headless Acceptance Matrix

- why-created: user requires real IsaacLab headless evidence under `env_isaacsim`, not only unit tests.
- cases:
  - `COBBLESTONE_ROAD_CFG` complex terrain with multiple command combinations;
  - flat semantic course with low-small crossing;
  - flat semantic course with high-small avoidance;
  - flat semantic course with large avoidance;
  - yaw-only near obstacle.
- required metrics:
  - T300e gait/grounding regression metrics;
  - T302 collision/semantic/cross/avoid/risk-scale/runtime metrics.
