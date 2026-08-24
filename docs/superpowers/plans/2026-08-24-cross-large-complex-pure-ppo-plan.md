# Cross Large Complex Pure PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a clean `cross_large_complex_ppo` experiment that trains `PPO + ActorCriticCNN` from scratch on the current mixed terrain without a teacher, imitation loss, Parallelism reference manager, or trajectory replanning.

**Architecture:** Build a new config directly on the ordinary `TeacherElevationTrajectoryMpcSemanticEnvCfg` base so the policy keeps the existing robot, scanner, action, event, terrain, command, and locomotion infrastructure without inheriting `ParallelismTrackingEnv`. Reuse only the mixed-terrain and semantic-layout factories from the cross-large task. Replace the manager-dependent collision reward with a new live-policy FK/high-map reward that reuses the static Parallelism geometry kernel but never imports the reference manager.

**Tech Stack:** Python 3.10, PyTorch, IsaacLab `ManagerBasedRLEnv`, Gymnasium, RSL-RL PPO, pytest, Bash, TensorBoard.

## Global Constraints

- New experiment name: `cross_large_complex_ppo`.
- Train from scratch; do not load teacher, student, critic, optimizer, or old checkpoints.
- Actor observations match the current distillation student and exclude `base_lin_vel`.
- Critic observations include `base_lin_vel` but exclude every Parallelism reference term.
- Use ordinary `isaaclab.envs:ManagerBasedRLEnv`, never `tracking.env:ParallelismTrackingEnv`.
- Do not modify the existing `parallelism_geometry_collision_penalty` or `active_swing_foot_on_small_obstacle_reward` implementations.
- Keep the reward-term name `parallelism_geometry_collision`, but bind it to the new `policy_geometry_collision_penalty` with weight `-10.0`.
- Do not enable `active_swing_foot_on_small_obstacle` in the new experiment.
- Keep only `time_out`, `base_contact` at force threshold `1.0`, and `bad_orientation` at `0.8 rad`.
- Set command resampling to exactly `10.0 s` for cross-large teacher, distillation, and pure PPO; do not change other tasks.
- Keep initial linear command ranges at `[-0.1, 0.1]`, final x at `[-1.0, 1.0]`, final y at `[-0.5, 0.5]`, and yaw at `[-1.0, 1.0]` from iteration zero.
- Preserve the existing net-displacement terrain curriculum formula and exclude `flat_dense_small_obstacles` from terrain levels.
- Use pure PPO settings aligned to distillation: 40 steps/env, 5 epochs, 4 mini-batches, `3e-4` fixed learning rate, entropy `0.01`, and initial std `1.0`.
- Preserve all unrelated dirty worktree changes. Every commit must stage only files named by its task.
- Design source: `docs/superpowers/specs/2026-08-24-cross-large-complex-pure-ppo-design-zh.html`.

---

## File Structure

**Create**

- `Go2Pvcnn/tracking/mdp/policy_geometry_rewards.py`: scanner-to-terrain conversion, live-policy FK collision computation, and reward entry point.
- `Go2Pvcnn/tracking/cross_large_complex_ppo_env_cfg.py`: pure PPO observations, rewards, curriculum, training EnvCfg, and PLAY EnvCfg.
- `Go2Pvcnn/tests/tracking/test_policy_geometry_rewards.py`: lightweight tensor and fake-environment tests for the new reward.
- `Go2Pvcnn/tests/tracking/test_cross_large_complex_ppo_static.py`: config, registration, training, play, and no-planner contract tests.
- `Go2Pvcnn/scripts/train_cross_large_complex_ppo_headless.sh`: from-scratch launcher.

**Modify**

- `Go2Pvcnn/tracking/parallelism_cross_large_complex_env_cfg.py`: expose one reusable semantic-obstacle curriculum factory without changing current behavior.
- `Go2Pvcnn/tracking/mdp/__init__.py`: export the new reward.
- `Go2Pvcnn/tracking/register_envs.py`: register the new normal IsaacLab environment.
- `Go2Pvcnn/tracking/parallelism_cross_large_complex_distillation_env_cfg.py`: change only the command resampling period from 100 s to 10 s.
- `Go2Pvcnn/agent/train_cfg.py`: add the standard PPO runner config.
- `Go2Pvcnn/scripts/train.py`: add parser and environment mappings.
- `Go2Pvcnn/scripts/play.py`: add parser and PLAY mappings without adding the experiment to Parallelism-specific branches.
- `Go2Pvcnn/tests/tracking/test_parallelism_distillation_env_cfg.py`: update the resampling contract from 100 s to 10 s.

---

### Task 1: Live-Policy Geometry Collision Reward

**Files:**
- Create: `Go2Pvcnn/tests/tracking/test_policy_geometry_rewards.py`
- Create: `Go2Pvcnn/tracking/mdp/policy_geometry_rewards.py`
- Modify: `Go2Pvcnn/tracking/mdp/__init__.py`

**Interfaces:**
- Produces: `parallelism_terrain_from_scan(ray_hits_w, semantic_map, valid_mask, resolution) -> ParallelismTerrain`.
- Produces: `live_policy_geometry_collision_event(root_pos_w, root_quat_w, joint_pos, joint_names, terrain, cfg=ParallelismCfg()) -> torch.Tensor` with shape `[num_envs]` and dtype `float32`.
- Produces: `policy_geometry_collision_penalty(env, asset_cfg, scanner_cfg) -> torch.Tensor` with shape `[num_envs]` and non-negative values.
- Depends only on `extension.convention`, `extension.parallelism.config`, `extension.parallelism.kinematics`, `extension.parallelism.collision`, and `extension.parallelism.types`; it must not import `tracking.managers`.

- [ ] **Step 1: Write failing scan and collision tests**

Create tests that build a 5x5 batched scanner grid and verify terrain metadata, joint reordering, collision-bit aggregation, and manager independence:

```python
from types import SimpleNamespace

import torch

import tracking.mdp.policy_geometry_rewards as rewards


def _scan(batch: int = 2, side: int = 5, resolution: float = 0.1):
    axis = torch.arange(side, dtype=torch.float32) * resolution
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    xyz = torch.stack((xx, yy, torch.zeros_like(xx)), dim=-1)
    return xyz.reshape(1, side * side, 3).expand(batch, -1, -1).clone()


def test_parallelism_terrain_from_scan_preserves_grid_pose():
    hits = _scan()
    semantic = torch.zeros(2, 5, 5, dtype=torch.long)
    terrain = rewards.parallelism_terrain_from_scan(hits, semantic, None, resolution=0.1)
    assert terrain.height_w.shape == (2, 5, 5)
    assert terrain.semantic_id.shape == (2, 5, 5)
    assert terrain.valid_mask.all()
    assert torch.allclose(terrain.origin_w[:, :2], hits[:, 0, :2])
    assert torch.allclose(terrain.yaw_w, torch.zeros(2))
    assert terrain.resolution == 0.1


def test_live_policy_collision_aggregates_all_legs(monkeypatch):
    bits = torch.zeros(2, 4, 1, 6, dtype=torch.bool)
    bits[1, 2, 0, 1] = True
    monkeypatch.setattr(rewards, "official_collision_mask", lambda terrain, geometry, cfg: (~bits.any(-1), bits))
    event = rewards.live_policy_geometry_collision_event(
        root_pos_w=torch.tensor([[0.0, 0.0, 0.3], [0.0, 0.0, 0.3]]),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        joint_pos=torch.zeros(2, 12),
        joint_names=(
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ),
        terrain=rewards.parallelism_terrain_from_scan(_scan(), torch.zeros(2, 5, 5), None, resolution=0.1),
    )
    assert event.tolist() == [0.0, 1.0]


def test_policy_reward_does_not_require_reference_manager(monkeypatch):
    monkeypatch.setattr(rewards, "live_policy_geometry_collision_event", lambda **kwargs: torch.tensor([0.0, 1.0]))
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.zeros(2, 3),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1),
            joint_pos=torch.zeros(2, 12),
        ),
        joint_names=[f"joint_{index}" for index in range(12)],
    )
    scanner = SimpleNamespace(
        cfg=SimpleNamespace(pattern_cfg=SimpleNamespace(resolution=0.1)),
        data=SimpleNamespace(ray_hits_w=_scan(), semantic_map=torch.zeros(2, 5, 5), valid_mask=None),
    )
    env = SimpleNamespace(scene={"robot": robot, "semantic_height_scanner": scanner})
    result = rewards.policy_geometry_collision_penalty(
        env,
        asset_cfg=SimpleNamespace(name="robot"),
        scanner_cfg=SimpleNamespace(name="semantic_height_scanner"),
    )
    assert result.tolist() == [0.0, 1.0]
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/tracking/test_policy_geometry_rewards.py -q
```

Expected: FAIL because `tracking.mdp.policy_geometry_rewards` does not exist.

- [ ] **Step 3: Implement the minimal independent reward module**

Implement these exact stages:

```python
_PLANNER_JOINT_ORDER = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
_COLLISION_CFG = ParallelismCfg()


def parallelism_terrain_from_scan(ray_hits_w, semantic_map, valid_mask, *, resolution):
    hits = torch.as_tensor(ray_hits_w, dtype=torch.float32)
    batch, ray_count, _ = hits.shape
    side = int(round(ray_count**0.5))
    if side * side != ray_count:
        raise ValueError(f"scanner ray count {ray_count} is not a square grid")
    grid = hits.reshape(batch, side, side, 3)
    semantic = torch.as_tensor(semantic_map, dtype=torch.long, device=hits.device).reshape(batch, side, side)
    valid = torch.isfinite(grid).all(-1) if valid_mask is None else torch.as_tensor(valid_mask, dtype=torch.bool, device=hits.device).reshape(batch, side, side)
    step_xy = grid[:, 0, 1, :2] - grid[:, 0, 0, :2]
    origin = torch.zeros(batch, 3, dtype=hits.dtype, device=hits.device)
    origin[:, :2] = grid[:, 0, 0, :2]
    return ParallelismTerrain(
        height_w=torch.nan_to_num(grid[..., 2]),
        semantic_id=semantic,
        valid_mask=valid,
        origin_w=origin,
        yaw_w=torch.atan2(step_xy[:, 1], step_xy[:, 0]),
        resolution=float(resolution),
    )


def live_policy_geometry_collision_event(root_pos_w, root_quat_w, joint_pos, joint_names, terrain, cfg=_COLLISION_CFG):
    roll, pitch = extract_roll_pitch_batch(root_quat_w)
    yaw = extract_yaw_batch(root_quat_w)
    ordered_joint_pos = _reorder_joint_to_planner(joint_pos, joint_names)
    geometry = fk_go2(root_pos_w, torch.stack((roll, pitch, yaw), dim=-1), ordered_joint_pos, capsule_samples=cfg.capsule_samples)
    expanded_geometry = _expand_live_geometry_for_collision(geometry)
    _, collision_bits = official_collision_mask(terrain, expanded_geometry, cfg)
    return collision_bits.any(dim=(1, 2, 3)).to(dtype=torch.float32)


def policy_geometry_collision_penalty(env, asset_cfg=SceneEntityCfg("robot"), scanner_cfg=SceneEntityCfg("semantic_height_scanner")):
    robot = env.scene[asset_cfg.name]
    scanner = env.scene[scanner_cfg.name]
    resolution = float(scanner.cfg.pattern_cfg.resolution)
    terrain = parallelism_terrain_from_scan(
        scanner.data.ray_hits_w,
        scanner.data.semantic_map,
        getattr(scanner.data, "valid_mask", None),
        resolution=resolution,
    )
    return live_policy_geometry_collision_event(
        root_pos_w=robot.data.root_pos_w,
        root_quat_w=robot.data.root_quat_w,
        joint_pos=robot.data.joint_pos,
        joint_names=tuple(robot.joint_names),
        terrain=terrain,
    )
```

Copy only the small joint-name normalization/reordering and live-geometry candidate-dimension expansion needed by this module. Do not import private helpers from `parallelism_reference_manager.py`.

- [ ] **Step 4: Export the reward and enforce the forbidden-import contract**

Add `policy_geometry_collision_penalty` to `tracking/mdp/__init__.py` and append a static assertion to the test:

```python
from pathlib import Path


def test_policy_reward_source_has_no_reference_manager_dependency():
    source = Path("Go2Pvcnn/tracking/mdp/policy_geometry_rewards.py").read_text()
    assert "get_parallelism_reference_manager" not in source
    assert "tracking.managers" not in source
```

- [ ] **Step 5: Run focused reward tests**

Run the Task 1 pytest command again.

Expected: all tests PASS.

- [ ] **Step 6: Commit only the reward files**

```bash
git add Go2Pvcnn/tracking/mdp/policy_geometry_rewards.py \
  Go2Pvcnn/tracking/mdp/__init__.py \
  Go2Pvcnn/tests/tracking/test_policy_geometry_rewards.py
git commit -m "feat: add planner-free policy geometry reward"
```

---

### Task 2: Pure PPO Environment Configuration

**Files:**
- Create: `Go2Pvcnn/tests/tracking/test_cross_large_complex_ppo_static.py`
- Create: `Go2Pvcnn/tracking/cross_large_complex_ppo_env_cfg.py`
- Modify: `Go2Pvcnn/tracking/parallelism_cross_large_complex_env_cfg.py`

**Interfaces:**
- Consumes: `tracking_mdp.policy_geometry_collision_penalty` from Task 1.
- Produces: `cross_large_complex_semantic_obstacle_curriculum_cfg() -> SemanticObstacleCurriculumCfg` shared by old and new configs.
- Produces: `CrossLargeComplexPpoEnvCfg` and `CrossLargeComplexPpoEnvCfg_PLAY`.

- [ ] **Step 1: Write failing static config tests**

Create tests that assert the source-level inheritance and runtime-independent contract:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_pure_ppo_config_does_not_inherit_parallelism():
    source = (ROOT / "tracking/cross_large_complex_ppo_env_cfg.py").read_text()
    assert "TeacherElevationTrajectoryMpcSemanticEnvCfg" in source
    assert "class CrossLargeComplexPpoEnvCfg(TeacherElevationTrajectoryMpcSemanticEnvCfg)" in source
    assert "ParallelismTrackingCrossLargeComplexEnvCfg" not in source
    assert "ParallelismTrackingPlaySceneCfg" not in source
    assert "planner_owned_reference_cache: bool = False" in source
    assert "use_batched_reference_trajectory: bool = False" in source


def test_pure_ppo_observation_contract():
    source = (ROOT / "tracking/cross_large_complex_ppo_env_cfg.py").read_text()
    assert "TeacherElevationTrajectoryMpcSemanticObservationsCfg" in source
    assert "parallelism_ref_" not in source
    assert "parallelism_plan_valid" not in source
    assert "base_lin_vel = None" not in source  # plain policy group never declares it


def test_pure_ppo_reward_and_termination_contract():
    source = (ROOT / "tracking/cross_large_complex_ppo_env_cfg.py").read_text()
    assert "func=tracking_mdp.policy_geometry_collision_penalty" in source
    assert "weight=-10.0" in source
    assert "active_swing_foot_on_small_obstacle = None" in source
    assert "reference_foot_pos = None" in source
    assert "undesired_contacts = None" in source
    assert "semantic_contact_collision = None" in source
    assert "TeacherElevationTrajectoryMpcSemanticTerminationsCfg" in source
    assert "parallelism_consecutive_standstill" not in source
    assert "parallelism_ref_" not in source


def test_pure_ppo_reuses_mixed_terrain_and_obstacle_counts():
    source = (ROOT / "tracking/cross_large_complex_ppo_env_cfg.py").read_text()
    assert "_cross_large_complex_terrain_cfg" in source
    assert "cross_large_complex_semantic_obstacle_curriculum_cfg" in source
    assert "excluded_terrain_names" in source
    assert '"flat_dense_small_obstacles"' in source
```

- [ ] **Step 2: Run the new static tests and verify failure**

Run:

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/tracking/test_cross_large_complex_ppo_static.py -q
```

Expected: FAIL because the new EnvCfg file does not exist.

- [ ] **Step 3: Extract the shared semantic curriculum factory**

In `parallelism_cross_large_complex_env_cfg.py`, move the existing semantic obstacle curriculum constructor into:

```python
def cross_large_complex_semantic_obstacle_curriculum_cfg() -> SemanticObstacleCurriculumCfg:
    return SemanticObstacleCurriculumCfg(
        enabled=True,
        plane_terrain_names=("flat_dense_small_obstacles", "flat"),
        plane_counts=(SemanticObstacleCount(small=0, large=2),),
        non_plane_counts=(SemanticObstacleCount(small=5, large=2),),
        terrain_obstacle_count_overrides={
            "flat": SemanticObstacleCount(small=0, large=2),
            "flat_dense_small_obstacles": SemanticObstacleCount(small=40, large=0),
        },
        center_safety_half_extent_m=(0.25,),
        min_spacing_clearance_m=(0.08,),
        tile_margin_m=(0.50,),
        collision_force_threshold=1.0,
    )
```

Change the old field to `field(default_factory=cross_large_complex_semantic_obstacle_curriculum_cfg)`. Run the existing cross-large static tests to prove behavior remains unchanged.

- [ ] **Step 4: Implement the pure PPO config classes**

Create these config units:

```python
@configclass
class CrossLargeComplexPpoRewardsCfg(TeacherElevationTrajectoryMpcSemanticRewardsCfg):
    reference_foot_pos = None
    undesired_contacts = None
    semantic_contact_collision = None
    active_swing_foot_on_small_obstacle = None
    parallelism_geometry_collision = RewTerm(
        func=tracking_mdp.policy_geometry_collision_penalty,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "scanner_cfg": SceneEntityCfg("semantic_height_scanner"),
        },
    )


@configclass
class CrossLargeComplexPpoCurriculumCfg(TeacherElevationTrajectoryMpcSemanticCurriculumCfg):
    terrain_levels = CurrTerm(
        func=go2_mdp.terrain_levels_vel_semantic_plane_gate,
        params={
            "cfg_name": "semantic_obstacle_curriculum",
            "excluded_terrain_names": ("flat_dense_small_obstacles",),
        },
    )
    lin_vel_cmd_levels = CurrTerm(go2_mdp.lin_vel_cmd_levels)


@configclass
class CrossLargeComplexPpoEnvCfg(TeacherElevationTrajectoryMpcSemanticEnvCfg):
    experiment_name: str = "cross_large_complex_ppo"
    scene: TeacherElevationTrajectoryMpcSemanticSceneCfg = TeacherElevationTrajectoryMpcSemanticSceneCfg(
        num_envs=1024, env_spacing=2.5, replicate_physics=True
    )
    observations: TeacherElevationTrajectoryMpcSemanticObservationsCfg = TeacherElevationTrajectoryMpcSemanticObservationsCfg()
    rewards: CrossLargeComplexPpoRewardsCfg = CrossLargeComplexPpoRewardsCfg()
    terminations: TeacherElevationTrajectoryMpcSemanticTerminationsCfg = TeacherElevationTrajectoryMpcSemanticTerminationsCfg()
    curriculum: CrossLargeComplexPpoCurriculumCfg = CrossLargeComplexPpoCurriculumCfg()
    planner_owned_reference_cache: bool = False
    use_batched_reference_trajectory: bool = False
    semantic_obstacle_curriculum: SemanticObstacleCurriculumCfg = field(
        default_factory=cross_large_complex_semantic_obstacle_curriculum_cfg
    )

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "cross_large_complex_ppo"
        self.planner_owned_reference_cache = False
        self.use_batched_reference_trajectory = False
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.ranges.lin_vel_x = (-0.1, 0.1)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-1.0, 1.0)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.3)
        self.events.push_robot = None
        self.scene.terrain.terrain_generator = _cross_large_complex_terrain_cfg()
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.class_type = SemanticCourseTerrainImporter
        self.scene.terrain.semantic_obstacle_curriculum = self.semantic_obstacle_curriculum
        self.scene.terrain.semantic_course_layout_cfg = SemanticCourseLayoutCfg(
            tile_margin_m=0.50,
            center_safety_half_extent_m=0.25,
            center_safety_radius_m=0.30,
            min_spacing_clearance_m=0.08,
        )
```

The inherited plain actor state already excludes `base_lin_vel`; the inherited plain critic state includes it. Do not redefine either group with Parallelism terms.

- [ ] **Step 5: Add the no-reference PLAY config**

```python
@configclass
class CrossLargeComplexPpoEnvCfg_PLAY(CrossLargeComplexPpoEnvCfg):
    scene: TeacherElevationTrajectoryMpcSemanticSceneCfg = TeacherElevationTrajectoryMpcSemanticSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )

    def __post_init__(self):
        super().__post_init__()
        self.terminations.time_out = None
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.observations.policy_elevation_semantic_map.enable_corruption = False
        self.observations.policy_state.enable_corruption = False
        self.observations.critic_elevation_semantic_map.enable_corruption = False
        self.observations.critic_state.enable_corruption = False
```

- [ ] **Step 6: Run focused config and old cross-large regression tests**

Run:

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/tracking/test_cross_large_complex_ppo_static.py \
  Go2Pvcnn/tests/tracking/test_parallelism_cross_large_complex_env_cfg.py -q
```

Expected: all tests PASS.

- [ ] **Step 7: Commit only the environment task files**

```bash
git add Go2Pvcnn/tracking/cross_large_complex_ppo_env_cfg.py \
  Go2Pvcnn/tracking/parallelism_cross_large_complex_env_cfg.py \
  Go2Pvcnn/tests/tracking/test_cross_large_complex_ppo_static.py
git commit -m "feat: add planner-free cross-large PPO environment"
```

---

### Task 3: Registration, PPO Runner Config, and 10-Second Commands

**Files:**
- Modify: `Go2Pvcnn/tracking/register_envs.py`
- Modify: `Go2Pvcnn/agent/train_cfg.py`
- Modify: `Go2Pvcnn/scripts/train.py`
- Modify: `Go2Pvcnn/tracking/parallelism_cross_large_complex_distillation_env_cfg.py`
- Modify: `Go2Pvcnn/tests/tracking/test_cross_large_complex_ppo_static.py`
- Modify: `Go2Pvcnn/tests/tracking/test_parallelism_distillation_env_cfg.py`

**Interfaces:**
- Consumes: `CrossLargeComplexPpoEnvCfg`.
- Produces: Gym ID `Isaac-Go2-Cross-Large-Complex-PPO-v0`.
- Produces: `get_train_cfg("cross_large_complex_ppo")` standard PPO dictionary.

- [ ] **Step 1: Extend failing registration and runner tests**

Add assertions:

```python
def test_pure_ppo_experiment_is_registered_as_normal_manager_env():
    register = (ROOT / "tracking/register_envs.py").read_text()
    train = (ROOT / "scripts/train.py").read_text()
    assert 'id="Isaac-Go2-Cross-Large-Complex-PPO-v0"' in register
    assert 'entry_point="isaaclab.envs:ManagerBasedRLEnv"' in register
    assert '"cross_large_complex_ppo"' in train


def test_pure_ppo_runner_has_no_distillation_fields():
    from agent.train_cfg import get_train_cfg
    cfg = get_train_cfg("cross_large_complex_ppo")
    assert cfg["algorithm"]["class_name"] == "PPO"
    assert cfg["policy"]["class_name"] == "ActorCriticCNN"
    assert cfg["algorithm"]["learning_rate"] == 3e-4
    assert cfg["algorithm"]["schedule"] == "fixed"
    assert cfg["algorithm"]["entropy_coef"] == 0.01
    assert cfg["policy"]["init_noise_std"] == 1.0
    assert cfg["obs_groups"] == {
        "policy": ["policy_elevation_semantic_map", "policy_state"],
        "critic": ["critic_elevation_semantic_map", "critic_state"],
    }
    serialized = repr(cfg)
    assert "teacher_coef" not in serialized
    assert "teacher_ratio" not in serialized
    assert "HybridDistillationPPO" not in serialized
```

Update the existing distillation static contract to require:

```python
assert "resampling_time_range = (10.0, 10.0)" in source
assert "resampling_time_range = (100.0, 100.0)" not in source
```

- [ ] **Step 2: Run focused tests and verify failures**

Run the two static test files. Expected failures: missing experiment registration/config and old 100-second assertion.

- [ ] **Step 3: Register the ordinary environment**

Add to `tracking/register_envs.py`:

```python
gym.register(
    id="Isaac-Go2-Cross-Large-Complex-PPO-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": CrossLargeComplexPpoEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
    disable_env_checker=True,
)
```

- [ ] **Step 4: Add the pure PPO train dictionary**

Add `_cross_large_complex_ppo_train_cfg()` by copying the current distillation PPO-side values, then changing only:

```python
"algorithm": {
    "class_name": "PPO",
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
    "learning_rate": 3e-4,
    "clip_param": 0.2,
    "gamma": 0.99,
    "lam": 0.95,
    "value_loss_coef": 1.0,
    "entropy_coef": 0.01,
    "max_grad_norm": 1.0,
    "use_clipped_value_loss": True,
    "schedule": "fixed",
    "desired_kl": 0.01,
},
"policy": {
    "class_name": "ActorCriticCNN",
    "init_noise_std": 1.0,
    "cost_map_channels": 2,
    "cost_map_size": 16,
    "actor_cnn_cfg": {
        "output_channels": [32, 64],
        "kernel_size": [3, 3],
        "stride": [1, 1],
        "padding": "zeros",
        "max_pool": [True, True],
        "activation": "elu",
        "flatten": True,
    },
    "critic_cnn_cfg": {
        "output_channels": [32, 64],
        "kernel_size": [3, 3],
        "stride": [1, 1],
        "padding": "zeros",
        "max_pool": [True, True],
        "activation": "elu",
        "flatten": True,
    },
    "actor_hidden_dims": [256, 128],
    "critic_hidden_dims": [256, 128],
    "activation": "elu",
},
"obs_groups": {
    "policy": ["policy_elevation_semantic_map", "policy_state"],
    "critic": ["critic_elevation_semantic_map", "critic_state"],
},
```

Keep `num_steps_per_env=40`, `save_interval=100`, `empirical_normalization=False`, `cost_map_channels=2`, and `cost_map_size=16`.

- [ ] **Step 5: Add the training parser/map entry without planner special cases**

Import `CrossLargeComplexPpoEnvCfg`, add `cross_large_complex_ppo` to the parser choices, and add:

```python
"cross_large_complex_ppo": (
    CrossLargeComplexPpoEnvCfg,
    "Isaac-Go2-Cross-Large-Complex-PPO-v0",
),
```

Do not add this experiment to any distillation CLI override, teacher checkpoint, resume, or planner branch.

- [ ] **Step 6: Change distillation command resampling to 10 seconds**

Change exactly:

```python
self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
```

The teacher cross-large config already has 10 seconds; leave it unchanged.

- [ ] **Step 7: Run focused tests**

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/tracking/test_cross_large_complex_ppo_static.py \
  Go2Pvcnn/tests/tracking/test_parallelism_distillation_env_cfg.py -q
```

Expected: all tests PASS.

- [ ] **Step 8: Commit only registration/config files**

```bash
git add Go2Pvcnn/tracking/register_envs.py \
  Go2Pvcnn/agent/train_cfg.py \
  Go2Pvcnn/scripts/train.py \
  Go2Pvcnn/tracking/parallelism_cross_large_complex_distillation_env_cfg.py \
  Go2Pvcnn/tests/tracking/test_cross_large_complex_ppo_static.py \
  Go2Pvcnn/tests/tracking/test_parallelism_distillation_env_cfg.py
git commit -m "feat: register pure PPO cross-large training"
```

---

### Task 4: Launcher and Pure PPO Play Path

**Files:**
- Create: `Go2Pvcnn/scripts/train_cross_large_complex_ppo_headless.sh`
- Modify: `Go2Pvcnn/scripts/play.py`
- Modify: `Go2Pvcnn/tests/tracking/test_cross_large_complex_ppo_static.py`

**Interfaces:**
- Consumes: pure PPO experiment mapping and `CrossLargeComplexPpoEnvCfg_PLAY`.
- Produces: a headless training launcher and ordinary inference path.

- [ ] **Step 1: Add failing launcher/play static assertions**

```python
def test_pure_ppo_launcher_has_no_teacher_or_resume_arguments():
    source = (ROOT / "scripts/train_cross_large_complex_ppo_headless.sh").read_text()
    assert "--experiment cross_large_complex_ppo" in source
    assert '--num_envs "${NUM_ENVS}"' in source
    assert '--max_iterations "${MAX_ITERATIONS}"' in source
    assert "teacher_checkpoint" not in source
    assert "teacher-coef" not in source
    assert "resume" not in source


def test_play_maps_pure_ppo_without_parallelism_visualization():
    source = (ROOT / "scripts/play.py").read_text()
    assert '"cross_large_complex_ppo"' in source
    assert "CrossLargeComplexPpoEnvCfg_PLAY" in source
    parallelism_tuple = source[source.index("is_parallelism_play ="):source.index("parallelism_panel_state = None")]
    assert '"cross_large_complex_ppo"' not in parallelism_tuple
```

- [ ] **Step 2: Run the static test and verify failure**

Expected: FAIL because launcher and play mapping do not exist.

- [ ] **Step 3: Create the from-scratch launcher**

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn

export DISPLAY="${DISPLAY:-:1}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"

ISAAC_ENV="/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim"
export LD_LIBRARY_PATH="${ISAAC_ENV}/lib/python3.10/site-packages/torch/lib:${ISAAC_ENV}/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${ISAAC_ENV}/lib/python3.10/site-packages/nvidia/cudnn/lib:${ISAAC_ENV}/lib:${ISAAC_ENV}/lib/python3.10/site-packages/nvidia/cuda/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

"${ISAAC_ENV}/bin/python" \
  Go2Pvcnn/scripts/train.py \
  --experiment cross_large_complex_ppo \
  --num_envs "${NUM_ENVS}" \
  --headless \
  --max_iterations "${MAX_ITERATIONS}" \
  --device cuda:0
```

Mark the script executable.

- [ ] **Step 4: Add ordinary play support**

Add the new choice/import/map entry:

```python
"cross_large_complex_ppo": (
    CrossLargeComplexPpoEnvCfg_PLAY,
    "Isaac-Go2-Cross-Large-Complex-PPO-v0",
),
```

Do not add the experiment to the tuple that forces `planner_backend="parallelism"` or the `is_parallelism_play` tuple. The generic wrapper and `runner.load()` path must handle it as a normal `ActorCriticCNN` checkpoint.

- [ ] **Step 5: Run static tests and shell syntax check**

```bash
bash -n Go2Pvcnn/scripts/train_cross_large_complex_ppo_headless.sh
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/tracking/test_cross_large_complex_ppo_static.py -q
```

Expected: syntax check and tests PASS.

- [ ] **Step 6: Commit only launcher/play files**

```bash
git add Go2Pvcnn/scripts/train_cross_large_complex_ppo_headless.sh \
  Go2Pvcnn/scripts/play.py \
  Go2Pvcnn/tests/tracking/test_cross_large_complex_ppo_static.py
git commit -m "feat: add pure PPO cross-large launch and play paths"
```

---

### Task 5: Regression Suite and Real 1024-Environment Training

**Files:**
- Modify only if a test exposes a defect: files created or modified in Tasks 1-4.
- Do not modify unrelated dirty scripts/tests to make the suite green.

**Interfaces:**
- Verifies the complete user-facing experiment.

- [ ] **Step 1: Run formatting and static checks**

```bash
git diff --check
bash -n Go2Pvcnn/scripts/train_cross_large_complex_ppo_headless.sh
```

Expected: no whitespace errors and valid Bash.

- [ ] **Step 2: Run the complete focused tracking suite**

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/tracking/test_policy_geometry_rewards.py \
  Go2Pvcnn/tests/tracking/test_cross_large_complex_ppo_static.py \
  Go2Pvcnn/tests/tracking/test_parallelism_cross_large_complex_env_cfg.py \
  Go2Pvcnn/tests/tracking/test_parallelism_distillation_env_cfg.py \
  Go2Pvcnn/tests/tracking/test_parallelism_distillation_reward_contract.py -q
```

Expected: all tests PASS. If a pre-existing dirty test fails because it describes uncommitted user behavior outside this feature, report it separately and do not revert it.

- [ ] **Step 3: Run a real 1024-environment, 4-iteration smoke train**

```bash
NUM_ENVS=1024 MAX_ITERATIONS=4 ./Go2Pvcnn/scripts/train_cross_large_complex_ppo_headless.sh
```

Expected:

- IsaacSim creates 1024 environments.
- Runner reports `PPO`, 5 learning epochs, learning rate `0.0003`.
- Four rollout/update iterations complete.
- No output line contains `[Planner] Attached`, `Teacher checkpoint`, or `Distillation/`.
- Reward/episode/curriculum logging remains finite.

- [ ] **Step 4: Inspect the generated TensorBoard event file**

Use `EventAccumulator` to assert required and forbidden tags:

```python
required = {
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Curriculum/lin_vel_cmd_levels",
    "Curriculum/terrain_levels",
}
forbidden_prefixes = ("Distillation/", "Episode_Tracking/")
```

Also confirm an episode reward tag for `parallelism_geometry_collision` exists once at least one episode has reset. If random episode initialization produces no reset in four iterations, verify the term from `env_cfg.yaml` instead and record that limitation.

- [ ] **Step 5: Compare collection time**

Report the four pure PPO collection times and compare their mean against the latest distillation run only when both runs use 1024 environments and 40 steps/env. Do not claim a speedup from unmatched environment counts or rollout horizons.

- [ ] **Step 6: Commit any smoke-test fixes, then verify commit scope**

If no fixes were required, do not create an empty commit. If fixes were required, stage only feature files and commit:

```bash
git commit -m "fix: complete pure PPO cross-large smoke training"
```

Finally run:

```bash
git status --short
git log -5 --oneline
```

Expected: unrelated pre-existing dirty files remain untouched; all feature commits are visible.

---

## Final Acceptance Checklist

- [ ] `cross_large_complex_ppo` trains from scratch with `PPO + ActorCriticCNN`.
- [ ] Actor has no `base_lin_vel`; critic has it.
- [ ] No actor/critic observation contains Parallelism reference data.
- [ ] No teacher network, teacher action, imitation loss, or distillation context is instantiated.
- [ ] No Parallelism manager is attached and no trajectory planning/replanning occurs.
- [ ] New live-policy collision reward uses `fk_go2 + official_collision_mask` and a scanner-built terrain.
- [ ] Existing Parallelism reward functions remain unchanged.
- [ ] `active_swing_foot_on_small_obstacle` remains disabled.
- [ ] Locomotion rewards and pure PPO hyperparameters match the approved design.
- [ ] Cross-large teacher, distillation, and pure PPO command resampling are all 10 seconds.
- [ ] Static/unit tests pass.
- [ ] Real 1024-environment training completes four iterations.
- [ ] TensorBoard contains PPO/reward/curriculum metrics and no distillation/reference metrics.
