# MPC Semantic RL Training Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent MPC + semantic scanner RL task that trains from MPC foot trajectories only, adds a high-resolution swing/leg collision reward, and passes a 4096 IsaacLab collect-data timing gate without regressing T302 MPC quality.

**Architecture:** Keep existing `teacher_elevation_trajectory` / `together` untouched. Add focused observation/reward helpers, a new task config file, new Gym/CLI registrations, and tests that prove semantic pooling, foot-only imitation, dirty-subset MPC replanning, and 4096 runtime timing. MPC continues to read current IsaacLab robot state and replan only selected dirty env rows.

**Tech Stack:** IsaacLab ManagerBasedRLEnv configs, `SemanticGridRayCasterCfg`, PyTorch tensor helpers, existing `MpcTrajectoryManager`, pytest, `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`.

---

## File Structure And Ownership

- Create `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
  - New train/play configs only.
  - Must not move or edit existing `TeacherElevationTrajectoryEnvCfg` defaults.
- Modify `Go2Pvcnn/extension/mdp/observations.py`
  - Add high-resolution semantic scanner downsampling helpers.
- Modify `Go2Pvcnn/extension/mdp/rewards_reference.py`
  - Add `swing_leg_collision_reward`.
- Modify `Go2Pvcnn/extension/mdp/__init__.py`
  - Export new helper names.
- Modify `Go2Pvcnn/go2_pvcnn/tasks/register_envs.py`
  - Import and register new train/play env classes.
- Modify `Go2Pvcnn/scripts/train.py`
  - Add `teacher_elevation_trajectory_mpc_semantic` experiment mapping.
- Modify `Go2Pvcnn/scripts/play.py`
  - Add `teacher_elevation_trajectory_mpc_semantic` experiment mapping and allow `mpc` in `--planner-backend`.
- Modify `Go2Pvcnn/tests/test_batch_mpc_backend.py`
  - Add fast helper/config/manager contract tests.
- Modify `Go2Pvcnn/tests/test_mpc_runtime_headless.py`
  - Add opt-in real 4096 timing/counter test.
- Modify notes:
  - `notes/todo.md`
  - `notes/todo/T302-mpc-body-leg-height-field-collision-safety.md`
  - `notes/todo/T302g-mpc-semantic-rl-training-config.md`
  - `notes/log/index.md`
  - one log per verification pass.

## Task 1: Add Semantic Priority Downsampling Helpers

**Files:**
- Modify: `Go2Pvcnn/extension/mdp/observations.py`
- Modify: `Go2Pvcnn/extension/mdp/__init__.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Add failing imports to the test**

In `Go2Pvcnn/tests/test_batch_mpc_backend.py`, add:

```python
from extension.mdp.observations import (
    downsample_height_map,
    downsample_semantic_priority_map,
    downsampled_semantic_height_scan,
)
```

Run:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_semantic_priority_pool_prefers_large_over_small -q
```

Expected: fail because the test does not exist yet.

- [ ] **Step 2: Add failing semantic priority pooling test**

Append near other lightweight helper tests in `Go2Pvcnn/tests/test_batch_mpc_backend.py`:

```python
def test_semantic_priority_pool_prefers_large_over_small() -> None:
    semantic = torch.zeros((1, 4, 4), dtype=torch.long)
    semantic[:, 0:2, 0:2] = 1
    semantic[:, 2:4, 2:4] = 2
    semantic[:, 0, 3] = 2
    semantic[:, 1, 3] = 1

    pooled = downsample_semantic_priority_map(
        semantic,
        target_size=2,
        small_ids=(1,),
        large_ids=(2,),
    )

    expected = torch.tensor([[[1, 2], [0, 2]]], dtype=torch.long)
    torch.testing.assert_close(pooled, expected)
```

Run:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_semantic_priority_pool_prefers_large_over_small -q
```

Expected: import failure for `downsample_semantic_priority_map`.

- [ ] **Step 3: Implement `downsample_semantic_priority_map`**

In `Go2Pvcnn/extension/mdp/observations.py`, add after `downsample_height_map`:

```python
def _semantic_mask_pool(mask: torch.Tensor, target_size: int) -> torch.Tensor:
    pooled = F.adaptive_max_pool2d(mask.to(dtype=torch.float32).unsqueeze(1), (target_size, target_size)).squeeze(1)
    return pooled > 0.5


def _semantic_id_mask(semantic_map: torch.Tensor, ids: tuple[int, ...]) -> torch.Tensor:
    if len(ids) == 0:
        return torch.zeros_like(semantic_map, dtype=torch.bool)
    id_tensor = torch.as_tensor(ids, dtype=semantic_map.dtype, device=semantic_map.device)
    return (semantic_map.unsqueeze(-1) == id_tensor.view(*([1] * semantic_map.ndim), -1)).any(dim=-1)


def downsample_semantic_priority_map(
    semantic_map: torch.Tensor,
    target_size: int,
    *,
    small_ids: tuple[int, ...] = (1,),
    large_ids: tuple[int, ...] = (2,),
) -> torch.Tensor:
    """Downsample semantic ids with large > small > terrain priority."""
    if semantic_map.ndim != 3:
        raise ValueError(f"Expected (batch, H, W), got shape {tuple(semantic_map.shape)}")
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    semantic = semantic_map.to(dtype=torch.long)
    large = _semantic_mask_pool(_semantic_id_mask(semantic, large_ids), target_size)
    small = _semantic_mask_pool(_semantic_id_mask(semantic, small_ids), target_size)
    out = torch.zeros((semantic.shape[0], target_size, target_size), dtype=torch.long, device=semantic.device)
    out = torch.where(small, torch.ones_like(out), out)
    out = torch.where(large, torch.full_like(out, 2), out)
    return out
```

- [ ] **Step 4: Run semantic priority pooling test**

Run:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_semantic_priority_pool_prefers_large_over_small -q
```

Expected: pass.

- [ ] **Step 5: Add failing dual-map helper test**

Add:

```python
class _FakeSemanticScannerData:
    def __init__(self) -> None:
        self.elevation_map = torch.arange(16, dtype=torch.float32).view(1, 4, 4)
        self.semantic_map = torch.zeros((1, 4, 4), dtype=torch.long)
        self.semantic_map[:, 0:2, 0:2] = 1
        self.semantic_map[:, 2:4, 2:4] = 2


class _FakeSemanticScanner:
    def __init__(self) -> None:
        self.data = _FakeSemanticScannerData()


class _FakeSensorContainer:
    def __init__(self) -> None:
        self._sensors = {"semantic_height_scanner": _FakeSemanticScanner()}

    def __getitem__(self, name: str):
        return self._sensors[name]


class _FakeScene:
    def __init__(self) -> None:
        self.sensors = _FakeSensorContainer()


class _FakeEnvForSemanticObs:
    num_envs = 1
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.scene = _FakeScene()


def test_downsampled_semantic_height_scan_returns_two_channel_map() -> None:
    out = downsampled_semantic_height_scan(
        _FakeEnvForSemanticObs(),
        sensor_cfg=SceneEntityCfg("semantic_height_scanner"),
        target_size=2,
    )

    assert out.shape == (1, 2, 2, 2)
    torch.testing.assert_close(out[:, 0], downsample_height_map(torch.arange(16, dtype=torch.float32).view(1, 4, 4), 2))
    expected_sem = torch.tensor([[[1, 0], [0, 2]]], dtype=torch.float32)
    torch.testing.assert_close(out[:, 1], expected_sem)
```

Run:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_downsampled_semantic_height_scan_returns_two_channel_map -q
```

Expected: import failure for `downsampled_semantic_height_scan`.

- [ ] **Step 6: Implement `downsampled_semantic_height_scan`**

Add to `Go2Pvcnn/extension/mdp/observations.py`:

```python
def downsampled_semantic_height_scan(
    env,
    sensor_cfg,
    target_size: int = 16,
    *,
    small_ids: tuple[int, ...] = (1,),
    large_ids: tuple[int, ...] = (2,),
    semantic_scale: float = 1.0,
) -> torch.Tensor:
    """Return height + priority semantic channels from a high-resolution semantic scanner."""
    try:
        sensor = env.scene.sensors[sensor_cfg.name]
    except Exception as exc:  # noqa: BLE001 - Isaac containers vary by version
        raise RuntimeError(f"Missing semantic scanner {sensor_cfg.name!r}") from exc
    data = sensor.data
    if getattr(data, "elevation_map", None) is None or getattr(data, "semantic_map", None) is None:
        raise RuntimeError(f"Semantic scanner {sensor_cfg.name!r} has no elevation_map/semantic_map")
    height = downsample_height_map(data.elevation_map.to(dtype=torch.float32), target_size=target_size)
    semantic = downsample_semantic_priority_map(
        data.semantic_map.to(dtype=torch.long),
        target_size=target_size,
        small_ids=small_ids,
        large_ids=large_ids,
    ).to(dtype=height.dtype)
    return torch.stack((height, semantic * float(semantic_scale)), dim=1)
```

Update `Go2Pvcnn/extension/mdp/__init__.py` imports and `__all__`:

```python
from .observations import downsample_height_map, downsample_semantic_priority_map, downsampled_height_scan, downsampled_semantic_height_scan
```

Include:

```python
"downsample_semantic_priority_map",
"downsampled_semantic_height_scan",
```

- [ ] **Step 7: Run observation helper tests**

Run:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_semantic_priority_pool_prefers_large_over_small \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_downsampled_semantic_height_scan_returns_two_channel_map -q
```

Expected: both pass.

## Task 2: Add Swing/Leg Collision Reward Helper

**Files:**
- Modify: `Go2Pvcnn/extension/mdp/rewards_reference.py`
- Modify: `Go2Pvcnn/extension/mdp/__init__.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Add failing reward helper import**

In `Go2Pvcnn/tests/test_batch_mpc_backend.py`, extend reward imports:

```python
from extension.mdp.rewards_reference import swing_leg_collision_reward
```

Run:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_swing_leg_collision_reward_uses_body_positions -q
```

Expected: fail because test/helper is missing.

- [ ] **Step 2: Add failing body-position reward test**

Add:

```python
class _FakeRobotData:
    def __init__(self) -> None:
        self.body_pos_w = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.30],
                    [0.0, 0.0, 0.12],
                    [0.1, 0.0, 0.03],
                    [0.2, 0.0, 0.02],
                ]
            ],
            dtype=torch.float32,
        )


class _FakeRobot:
    def __init__(self) -> None:
        self.data = _FakeRobotData()

    def find_bodies(self, pattern: str):
        if pattern == ".*_(thigh|calf|foot)" or "thigh" in pattern:
            return [1, 2, 3], ["FL_thigh", "FL_calf", "FL_foot"]
        if pattern == ".*_foot":
            return [3], ["FL_foot"]
        return [], []


class _FakeCollisionScannerData:
    def __init__(self) -> None:
        self.elevation_map = torch.zeros((1, 4, 4), dtype=torch.float32)
        self.semantic_map = torch.zeros((1, 4, 4), dtype=torch.long)
        self.semantic_map[:, 2, 2] = 2


class _FakeCollisionScanner:
    def __init__(self) -> None:
        self.data = _FakeCollisionScannerData()
        self.cfg = type("Cfg", (), {"pattern_cfg": type("Pattern", (), {"size": [1.5, 1.5]})()})()


class _FakeSceneForCollision:
    def __init__(self) -> None:
        self.sensors = {"semantic_height_scanner": _FakeCollisionScanner()}
        self._robot = _FakeRobot()

    def __getitem__(self, name: str):
        if name == "robot":
            return self._robot
        raise KeyError(name)


class _FakeEnvForCollision:
    num_envs = 1
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.scene = _FakeSceneForCollision()


def test_swing_leg_collision_reward_uses_body_positions() -> None:
    env = _FakeEnvForCollision()
    reward = swing_leg_collision_reward(
        env,
        asset_cfg=SceneEntityCfg("robot", body_names=".*_(thigh|calf|foot)"),
        sensor_cfg=SceneEntityCfg("semantic_height_scanner"),
        height_margin_m=0.05,
        small_weight=1.0,
        large_weight=5.0,
    )

    assert reward.shape == (1,)
    assert reward.item() < 0.0
```

Run:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_swing_leg_collision_reward_uses_body_positions -q
```

Expected: fail because `swing_leg_collision_reward` is missing.

- [ ] **Step 3: Implement helper functions and reward**

Add to `Go2Pvcnn/extension/mdp/rewards_reference.py` before `zero_reference_reward`:

```python
def _scene_get(container, name: str):
    try:
        return container[name]
    except Exception:  # noqa: BLE001
        return getattr(container, name)


def _scanner_world_ranges(scanner) -> tuple[tuple[float, float], tuple[float, float]]:
    pattern_cfg = getattr(getattr(scanner, "cfg", None), "pattern_cfg", None)
    size = getattr(pattern_cfg, "size", None)
    if size is None:
        return (-0.75, 0.75), (-0.75, 0.75)
    return (-0.5 * float(size[0]), 0.5 * float(size[0])), (-0.5 * float(size[1]), 0.5 * float(size[1]))


def _nearest_grid_sample(grid: torch.Tensor, xy: torch.Tensor, scanner) -> torch.Tensor:
    if grid.ndim == 2:
        grid = grid.unsqueeze(0)
    batch, rows, cols = int(grid.shape[0]), int(grid.shape[1]), int(grid.shape[2])
    if xy.shape[0] != batch:
        if batch == 1:
            grid = grid.expand(xy.shape[0], -1, -1)
            batch = int(xy.shape[0])
        else:
            raise ValueError(f"grid batch {batch} does not match points batch {xy.shape[0]}")
    x_range, y_range = _scanner_world_ranges(scanner)
    x = ((xy[..., 0] - float(x_range[0])) / max(float(x_range[1]) - float(x_range[0]), 1.0e-6) * (cols - 1)).round()
    y = ((xy[..., 1] - float(y_range[0])) / max(float(y_range[1]) - float(y_range[0]), 1.0e-6) * (rows - 1)).round()
    xi = x.clamp(0, cols - 1).to(dtype=torch.long)
    yi = y.clamp(0, rows - 1).to(dtype=torch.long)
    env_ids = torch.arange(batch, device=xy.device).view(batch, *([1] * (xy.ndim - 2))).expand_as(xi)
    return grid.to(device=xy.device)[env_ids, yi, xi]


def swing_leg_collision_reward(
    env,
    asset_cfg=None,
    sensor_cfg=None,
    *,
    height_margin_m: float = 0.04,
    small_weight: float = 1.0,
    large_weight: float = 5.0,
    semantic_weight: float = 1.0,
    penetration_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize current simulated leg/body samples that collide with high-res semantic terrain."""
    from isaaclab.assets import Articulation
    from isaaclab.managers import SceneEntityCfg

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", body_names=".*_(thigh|calf|foot)")
    if sensor_cfg is None:
        sensor_cfg = SceneEntityCfg("semantic_height_scanner")
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = getattr(asset_cfg, "body_ids", None)
    if body_ids is None or len(body_ids) == 0:
        body_ids, _ = asset.find_bodies(getattr(asset_cfg, "body_names", ".*_(thigh|calf|foot)"))
    if len(body_ids) == 0:
        raise RuntimeError("swing_leg_collision_reward found no thigh/calf/foot bodies")
    points = asset.data.body_pos_w[:, body_ids, :]
    scanner = _scene_get(env.scene.sensors, sensor_cfg.name)
    data = scanner.data
    if getattr(data, "elevation_map", None) is None or getattr(data, "semantic_map", None) is None:
        raise RuntimeError(f"Semantic scanner {sensor_cfg.name!r} has no elevation_map/semantic_map")
    height = _nearest_grid_sample(data.elevation_map.to(dtype=points.dtype, device=points.device), points[..., :2], scanner)
    semantic = _nearest_grid_sample(data.semantic_map.to(dtype=torch.long, device=points.device), points[..., :2], scanner)
    penetration = torch.relu(height + float(height_margin_m) - points[..., 2]).square()
    semantic_penalty = torch.zeros_like(penetration)
    semantic_penalty = torch.where(semantic == 1, torch.full_like(semantic_penalty, float(small_weight)), semantic_penalty)
    semantic_penalty = torch.where(semantic >= 2, torch.full_like(semantic_penalty, float(large_weight)), semantic_penalty)
    penalty = float(penetration_weight) * penetration.mean(dim=1) + float(semantic_weight) * semantic_penalty.mean(dim=1)
    return -penalty
```

Update `__all__` with:

```python
"swing_leg_collision_reward",
```

Update `Go2Pvcnn/extension/mdp/__init__.py`:

```python
from .rewards_reference import exponential_tracking_reward, swing_leg_collision_reward, zero_reference_reward
```

and add `"swing_leg_collision_reward"` to `__all__`.

- [ ] **Step 4: Run reward helper test**

Run:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_swing_leg_collision_reward_uses_body_positions -q
```

Expected: pass.

## Task 3: Add Independent MPC Semantic Env Config

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Add failing config import test**

Add to `Go2Pvcnn/tests/test_batch_mpc_backend.py`:

```python
def test_mpc_semantic_trajectory_cfg_defaults_to_mpc_and_semantic_scanner() -> None:
    from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
        TeacherElevationTrajectoryMpcSemanticEnvCfg,
    )

    cfg = TeacherElevationTrajectoryMpcSemanticEnvCfg()

    assert cfg.planner_backend == "mpc"
    assert cfg.reference_height_scanner_name == "semantic_height_scanner"
    assert cfg.scene.height_scanner is None
    assert cfg.scene.semantic_height_scanner is not None
    assert cfg.observations.policy_elevation_semantic_map.elevation_semantic.params["target_size"] == 16
    assert cfg.rewards.reference_foot_pos.weight > 0.0
    assert not hasattr(cfg.rewards, "reference_root_pose")
    assert not hasattr(cfg.rewards, "reference_joint_pos")
    assert not hasattr(cfg.rewards, "reference_contact")
    assert not hasattr(cfg.rewards, "reference_touchdown")
```

Run:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_semantic_trajectory_cfg_defaults_to_mpc_and_semantic_scanner -q
```

Expected: import failure because the new module does not exist.

- [ ] **Step 2: Create the new config file**

Create `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`:

```python
"""MPC semantic trajectory RL task config."""

from __future__ import annotations

from dataclasses import field

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from extension.batch_mpc_planner.config import MpcPlannerCfg
from extension.mdp.observations import downsampled_semantic_height_scan
from extension.mdp.rewards_reference import reference_foot_pos_reward, swing_leg_collision_reward
from extension.semantic_course import SEMANTIC_COURSE_LARGE_ROOT, SEMANTIC_COURSE_SMALL_ROOT
from go2_pvcnn.sensor.semantic_raycaster import SemanticGridRayCasterCfg
from go2_pvcnn.tasks.teacher_elevation_trajectory_env_cfg import (
    SEMANTIC_TERRAIN_CFG,
    TeacherElevationTrajectoryEnvCfg,
    TeacherElevationTrajectoryEnvCfg_PLAY,
    TeacherElevationTrajectorySceneCfg,
)
from go2_pvcnn.tasks.teacher_without_semantic_env_cfg import RewardsCfg as BaseRewardsCfg


@configclass
class TeacherElevationTrajectoryMpcSemanticSceneCfg(TeacherElevationTrajectorySceneCfg):
    height_scanner = None
    semantic_height_scanner = SemanticGridRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=SemanticGridRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[1.5, 1.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground", SEMANTIC_COURSE_SMALL_ROOT, SEMANTIC_COURSE_LARGE_ROOT],
        mesh_semantic_ids={
            "/World/ground": 0,
            SEMANTIC_COURSE_SMALL_ROOT: 1,
            SEMANTIC_COURSE_LARGE_ROOT: 2,
        },
        height_scan_offset=0.5,
    )


@configclass
class TeacherElevationTrajectoryMpcSemanticObservationsCfg:
    @configclass
    class PolicyElevationSemanticMapCfg(ObsGroup):
        elevation_semantic = ObsTerm(
            func=downsampled_semantic_height_scan,
            params={"sensor_cfg": SceneEntityCfg("semantic_height_scanner"), "target_size": 16},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 2.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PolicyStateCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        velocity_commands = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "base_velocity"})
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticElevationSemanticMapCfg(ObsGroup):
        elevation_semantic = ObsTerm(
            func=downsampled_semantic_height_scan,
            params={"sensor_cfg": SceneEntityCfg("semantic_height_scanner"), "target_size": 16},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 2.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticStateCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        velocity_commands = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "base_velocity"})
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy_elevation_semantic_map: PolicyElevationSemanticMapCfg = PolicyElevationSemanticMapCfg()
    policy_state: PolicyStateCfg = PolicyStateCfg()
    critic_elevation_semantic_map: CriticElevationSemanticMapCfg = CriticElevationSemanticMapCfg()
    critic_state: CriticStateCfg = CriticStateCfg()


@configclass
class TeacherElevationTrajectoryMpcSemanticRewardsCfg(BaseRewardsCfg):
    reference_foot_pos = RewTerm(
        func=reference_foot_pos_reward,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
    )
    swing_leg_collision = RewTerm(
        func=swing_leg_collision_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_(thigh|calf|foot)"),
            "sensor_cfg": SceneEntityCfg("semantic_height_scanner"),
        },
    )


@configclass
class TeacherElevationTrajectoryMpcSemanticEnvCfg(TeacherElevationTrajectoryEnvCfg):
    scene: TeacherElevationTrajectoryMpcSemanticSceneCfg = TeacherElevationTrajectoryMpcSemanticSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
    )
    observations: TeacherElevationTrajectoryMpcSemanticObservationsCfg = TeacherElevationTrajectoryMpcSemanticObservationsCfg()
    rewards: TeacherElevationTrajectoryMpcSemanticRewardsCfg = TeacherElevationTrajectoryMpcSemanticRewardsCfg()

    planner_backend: str = "mpc"
    reference_height_scanner_name: str = "semantic_height_scanner"
    reference_trajectory_horizon: int = 50
    reference_replan_interval_steps: int = 50
    mpc_max_dirty_envs_per_step: int = 256
    mpc_max_stale_steps: int = 100
    mpc_optimize_steps: int = 24
    mpc_diagnostics_emit_runtime_counters: bool = False
    mpc_planner_cfg: MpcPlannerCfg = field(default_factory=MpcPlannerCfg)

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner = None
        if self.scene.semantic_height_scanner is not None:
            self.scene.semantic_height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY(TeacherElevationTrajectoryEnvCfg_PLAY):
    scene: TeacherElevationTrajectoryMpcSemanticSceneCfg = TeacherElevationTrajectoryMpcSemanticSceneCfg(
        num_envs=32,
        env_spacing=2.5,
    )
    observations: TeacherElevationTrajectoryMpcSemanticObservationsCfg = TeacherElevationTrajectoryMpcSemanticObservationsCfg()
    rewards: TeacherElevationTrajectoryMpcSemanticRewardsCfg = TeacherElevationTrajectoryMpcSemanticRewardsCfg()

    planner_backend: str = "mpc"
    reference_height_scanner_name: str = "semantic_height_scanner"
    reference_trajectory_horizon: int = 50
    reference_replan_interval_steps: int = 50
    mpc_max_dirty_envs_per_step: int = 256
    mpc_max_stale_steps: int = 100
    mpc_optimize_steps: int = 24
    mpc_diagnostics_emit_runtime_counters: bool = False
    mpc_planner_cfg: MpcPlannerCfg = field(default_factory=MpcPlannerCfg)

    def __post_init__(self):
        super().__post_init__()
        tg = self.scene.terrain.terrain_generator
        if tg is not None:
            tg.num_rows = SEMANTIC_TERRAIN_CFG.num_rows
            tg.num_cols = SEMANTIC_TERRAIN_CFG.num_cols
            tg.curriculum = SEMANTIC_TERRAIN_CFG.curriculum
        self.scene.height_scanner = None
        self.observations.policy_elevation_semantic_map.enable_corruption = False
        self.observations.policy_state.enable_corruption = False
        self.observations.critic_elevation_semantic_map.enable_corruption = False
        self.observations.critic_state.enable_corruption = False
        if self.scene.semantic_height_scanner is not None:
            self.scene.semantic_height_scanner.update_period = self.decimation * self.sim.dt
```

- [ ] **Step 3: Run config default test**

Run:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_semantic_trajectory_cfg_defaults_to_mpc_and_semantic_scanner -q
```

Expected: pass.

## Task 4: Register Gym Ids And CLI Experiments

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/register_envs.py`
- Modify: `Go2Pvcnn/scripts/train.py`
- Modify: `Go2Pvcnn/scripts/play.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Add failing train/play parser test**

Add:

```python
def test_train_and_play_parsers_include_mpc_semantic_experiment() -> None:
    from scripts import play, train

    train_parser = train.build_arg_parser()
    play_parser = play.build_arg_parser()
    train_experiment = next(action for action in train_parser._actions if action.dest == "experiment")
    play_experiment = next(action for action in play_parser._actions if action.dest == "experiment")
    play_backend = next(action for action in play_parser._actions if action.dest == "planner_backend")

    assert "teacher_elevation_trajectory_mpc_semantic" in train_experiment.choices
    assert "teacher_elevation_trajectory_mpc_semantic" in play_experiment.choices
    assert "mpc" in play_backend.choices
```

Run:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_train_and_play_parsers_include_mpc_semantic_experiment -q
```

Expected: fail because parser choices are missing.

- [ ] **Step 2: Modify `train.py` parser and mapping**

In `Go2Pvcnn/scripts/train.py`, add `"teacher_elevation_trajectory_mpc_semantic"` to `--experiment` choices and help.

Import:

```python
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
    TeacherElevationTrajectoryMpcSemanticEnvCfg,
)
```

Add to `EXPERIMENT_ENV_MAP`:

```python
"teacher_elevation_trajectory_mpc_semantic": (
    TeacherElevationTrajectoryMpcSemanticEnvCfg,
    "Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0",
),
```

Change trajectory planner CLI override condition to include both experiments:

```python
if args_cli.experiment in {"teacher_elevation_trajectory", "teacher_elevation_trajectory_mpc_semantic"}:
```

- [ ] **Step 3: Modify `play.py` parser and mapping**

In `Go2Pvcnn/scripts/play.py`, add `"teacher_elevation_trajectory_mpc_semantic"` to `--experiment` choices and help.

Change `--planner-backend` choices to:

```python
choices=["together", "legacy", "mpc"]
```

Import:

```python
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
    TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY,
)
```

Add to experiment mapping:

```python
"teacher_elevation_trajectory_mpc_semantic": (
    TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY,
    "Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-Play-v0",
),
```

Change trajectory reference configuration condition to include both trajectory experiments:

```python
if experiment_name in {"teacher_elevation_trajectory", "teacher_elevation_trajectory_mpc_semantic"}:
```

- [ ] **Step 4: Register new Gym ids**

In `Go2Pvcnn/go2_pvcnn/tasks/register_envs.py`, import:

```python
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
    TeacherElevationTrajectoryMpcSemanticEnvCfg,
    TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY,
)
```

Add:

```python
gym.register(
    id="Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TeacherElevationTrajectoryMpcSemanticEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": None,
    },
)
```

- [ ] **Step 5: Run parser test and compile**

Run:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_train_and_play_parsers_include_mpc_semantic_experiment -q

/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/scripts/train.py \
  Go2Pvcnn/scripts/play.py \
  Go2Pvcnn/go2_pvcnn/tasks/register_envs.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
```

Expected: parser test passes and `py_compile` exits `0`.

## Task 5: Protect Dirty-Subset Replanning And Current Isaac State Contract

**Files:**
- Modify: `Go2Pvcnn/tests/test_batch_mpc_backend.py`
- Production changes only if test exposes a regression.

- [ ] **Step 1: Add focused dirty-subset manager test**

Add:

```python
def test_mpc_manager_selects_dirty_subset_under_budget() -> None:
    cfg = _task_cfg(
        planner_backend="mpc",
        mpc_diagnostics_emit_runtime_counters=True,
        mpc_max_dirty_envs_per_step=2,
        reference_replan_interval_steps=1000,
        mpc_max_stale_steps=1000,
    )
    manager = MpcTrajectoryManager(cfg, device="cpu")
    manager._ensure_state(5)
    assert manager._pending_command_mask is not None
    manager._pending_command_mask[:] = torch.tensor([True, True, True, False, False])
    score = torch.zeros(5, dtype=torch.float32)
    score = torch.where(manager._pending_command_mask.cpu(), torch.full_like(score, 3.0), score)

    _ids, selected = manager._select_dirty_rows(score, budget=2)

    assert int(torch.count_nonzero(selected).item()) == 2
```

Run:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_manager_selects_dirty_subset_under_budget -q
```

Expected: pass with current implementation.

- [ ] **Step 2: Add static current-state source guard**

Add:

```python
def test_mpc_manager_replan_state_reads_isaac_buffers_not_reference_cache() -> None:
    source = (GO2PVCNN_ROOT / "extension" / "batch_mpc_planner" / "manager.py").read_text(encoding="utf-8")

    assert "data.root_pos_w" in source
    assert "data.root_quat_w" in source
    assert "data.joint_pos" in source
    assert "data.body_pos_w" in source
    state_block = source[source.index("def _state_from_env"):source.index("    @staticmethod", source.index("def _state_from_env"))]
    assert "_cache" not in state_block
    assert "current_reference" not in state_block
```

Run:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_manager_replan_state_reads_isaac_buffers_not_reference_cache -q
```

Expected: pass.

## Task 6: Add Real 4096 Collect-Data Timing Test

**Files:**
- Modify: `Go2Pvcnn/tests/test_mpc_runtime_headless.py`

- [ ] **Step 1: Add opt-in runtime test**

Add near existing 4096 MPC runtime tests:

```python
@pytest.fixture(scope="module")
def real_semantic_mpc_rl_runtime_4096():
    if os.environ.get("MPC_SEMANTIC_RL_4096_TIMING") != "1":
        pytest.skip("Set MPC_SEMANTIC_RL_4096_TIMING=1 to run 4096-env MPC semantic RL timing.")
    from tests.fixtures import viewer_runtime_diagnostics as viewer_diag

    runtime = viewer_diag.make_real_runtime_fixture(
        num_envs=4096,
        planner_backend="mpc",
        device=os.environ.get("MPC_TEST_DEVICE", "cuda:0"),
        warmup_steps=2,
    )
    try:
        yield runtime
    finally:
        runtime.close()


@pytest.mark.skipif(os.environ.get("MPC_SEMANTIC_RL_4096_TIMING") != "1", reason="opt-in 4096 MPC semantic RL timing test")
def test_mpc_semantic_rl_4096_collect_data_under_10s(real_semantic_mpc_rl_runtime_4096) -> None:
    runtime = real_semantic_mpc_rl_runtime_4096
    manager = runtime.base_env._trajectory_manager
    runtime.mpc_planner_cfg.diagnostics.emit_runtime_counters = True
    runtime.mpc_planner_cfg.diagnostics.profile_cuda_sync = False
    runtime.mpc_planner_cfg.runtime.max_dirty_envs_per_step = 256

    steps = int(os.environ.get("MPC_SEMANTIC_RL_TIMING_STEPS", "24"))
    start = time.perf_counter()
    for _ in range(steps):
        runtime.env.step(runtime.zero_actions)
    elapsed = time.perf_counter() - start
    counters = manager.runtime_counters()

    assert elapsed < 10.0, f"collect-data timing too slow: {elapsed:.3f}s for {steps} steps"
    assert counters["selected_dirty_count"] <= 256
    assert counters["dirty_backlog"] == counters["dirty_count"] - counters["selected_dirty_count"]
    assert counters["planner_ms"] >= 0.0
    assert counters["cache_ms"] >= 0.0
```

- [ ] **Step 2: Run collect-only check**

Run:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_semantic_rl_4096_collect_data_under_10s --collect-only -q
```

Expected: test is collected.

- [ ] **Step 3: Run opt-in 4096 timing test**

Run:

```bash
TMPDIR=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/tmp \
MPC_SEMANTIC_RL_4096_TIMING=1 \
MPC_TEST_DEVICE=cuda:0 \
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_semantic_rl_4096_collect_data_under_10s -q -s
```

Expected: pass under `10s`. If the timing assertion fails, keep the failure and optimize tensor paths, reward sampling, scanner handling, terrain subset building, or dirty-subset scheduling without reducing MPC collision/semantic loss quality.

## Task 7: Run T302 Non-Regression Verification

**Files:**
- No production edits unless regression appears.
- Logs: create `notes/log/YYYY-MM-DD-HHMM-t302g-non-regression.md`

- [ ] **Step 1: Run backend suite**

Run:

```bash
TMPDIR=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/tmp \
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Expected: pass.

- [ ] **Step 2: Run T302 strict metric path**

Run the same strict single-process JSONL probes described in:

```text
notes/log/2026-05-17-0804-t302-strict-collision-metric-tuning.md
```

Acceptance:

- `17/17` rows pass.
- root-bottom/swing-foot/knee/shank collision ratios remain `0.0`.
- stance semantic count remains `0`.
- low-small crossing remains accepted.
- high-small/large risk deweighting remains accepted.

- [ ] **Step 3: Run py_compile and diff check**

Run:

```bash
TMPDIR=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/tmp \
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/extension/mdp/observations.py \
  Go2Pvcnn/extension/mdp/rewards_reference.py \
  Go2Pvcnn/extension/mdp/__init__.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py \
  Go2Pvcnn/go2_pvcnn/tasks/register_envs.py \
  Go2Pvcnn/scripts/train.py \
  Go2Pvcnn/scripts/play.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  Go2Pvcnn/tests/test_mpc_runtime_headless.py

git diff --check
```

Expected: both exit `0`.

## Task 8: Update Notes And Logs

**Files:**
- Modify: `notes/todo.md`
- Modify: `notes/todo/T302-mpc-body-leg-height-field-collision-safety.md`
- Create/modify: `notes/todo/T302g-mpc-semantic-rl-training-config.md`
- Modify: `notes/log/index.md`
- Create logs under `notes/log/`

- [ ] **Step 1: Update T302g branch page**

Ensure `notes/todo/T302g-mpc-semantic-rl-training-config.md` contains:

```markdown
# T302g MPC Semantic RL Training Config

## Current State

- Independent MPC semantic RL train/play config is the active child of T302.
- Existing `teacher_elevation_trajectory` / together path remains unchanged.
- Acceptance requires 4096 collect-data timing under 10s and no T302 strict metric regression.

## Open Children

| Child | Status | Priority | Purpose | Primary Files |
| --- | --- | --- | --- | --- |
| T302g.1 | verify | P0 | Implement independent config, semantic CNN observation, foot-only imitation, swing collision reward, and 4096 timing gate | `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`, `Go2Pvcnn/extension/mdp/` |

## Closed Children Archive

- None yet.

## Related Logs

- Add implementation and verification logs as they are created.

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: `pending`
- Current Work Ref: `working tree on top of 946811f`
- Key Files:
  - [../../docs/superpowers/specs/2026-05-18-mpc-semantic-rl-training-config-design.md](../../docs/superpowers/specs/2026-05-18-mpc-semantic-rl-training-config-design.md)
  - [../../docs/superpowers/plans/2026-05-18-mpc-semantic-rl-training-config.md](../../docs/superpowers/plans/2026-05-18-mpc-semantic-rl-training-config.md)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)

## Next Step

- Execute the implementation plan task-by-task.

## Node Details

### T302g.1 Independent MPC semantic RL rollout config

- why-created: promote T302 MPC safety into a trainable RL task without changing together defaults.
- hypothesis: high-res semantic scanner plus foot-only MPC imitation and swing collision reward can train safely if dirty-subset MPC scheduling keeps 4096 collect-data under 10s.
- evidence: pending implementation and timing logs.
```

- [ ] **Step 2: Update dashboard and parent**

Add T302g to:

- `notes/todo.md` Active Fronts or Open Leaves.
- `notes/todo/T302-mpc-body-leg-height-field-collision-safety.md` Open Children.

- [ ] **Step 3: Update log index**

Add rows for:

- design/spec log
- implementation plan log
- implementation verification log
- 4096 timing log
- T302 non-regression log

All links must be repository-relative.

## Final Verification Commands

Run before claiming completion:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py -q

/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_semantic_rl_4096_collect_data_under_10s --collect-only -q

TMPDIR=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/tmp \
MPC_SEMANTIC_RL_4096_TIMING=1 \
MPC_TEST_DEVICE=cuda:0 \
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_semantic_rl_4096_collect_data_under_10s -q -s

/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/extension/mdp/observations.py \
  Go2Pvcnn/extension/mdp/rewards_reference.py \
  Go2Pvcnn/extension/mdp/__init__.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py \
  Go2Pvcnn/go2_pvcnn/tasks/register_envs.py \
  Go2Pvcnn/scripts/train.py \
  Go2Pvcnn/scripts/play.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  Go2Pvcnn/tests/test_mpc_runtime_headless.py

git diff --check
```

T302 strict metric probes must also be rerun or explicitly recorded as blocked with reason.
