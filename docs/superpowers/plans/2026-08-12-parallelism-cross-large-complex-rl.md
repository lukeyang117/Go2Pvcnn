# Parallelism Cross Large Complex Terrain RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a registered RL experiment that inherits the small-obstacle Parallelism tracking task, trains on one mixed terrain set, crosses dense small obstacles, avoids large obstacles, and preserves `parallelism_consecutive_standstill` termination.

**Architecture:** Create a new configuration derived from `ParallelismTrackingSmallObstaclesEnvCfg`. It uses the existing semantic terrain importer and planner, with a mixed terrain generator containing one dense flat subterrain and the existing rough/slope/boxes/stairs terrain types. Dense flat receives 40 small obstacles and no large obstacles at about 1/16 proportion; every other terrain receives five small and two large obstacles. Dense flat is excluded only from terrain curriculum. The inherited termination set, including continuous-two-standstill termination, remains active.

**Tech Stack:** Python, IsaacLab config classes, Gymnasium registration, PyTorch/TorchScript-compatible masks, pytest, Isaac Sim headless smoke test.

## Global Constraints

- New experiment name is `parallelism_tracking_cross_large_complex`.
- New RL config is under `Go2Pvcnn/tracking/parallelism_cross_large_complex_env_cfg.py`.
- Existing `parallelism_tracking_flat`, `parallelism_tracking_small_obstacles`, and `parallelism_tracking_ladder` behavior must not be changed.
- Dense flat uses `small=40, large=0`; all other terrain types use `small=5, large=2`.
- Obstacles are created once during terrain/importer initialization and are not recreated on reset.
- Dense flat does not move up or down through terrain curriculum; all other terrain types continue using the existing terrain curriculum.
- `parallelism_consecutive_standstill` remains enabled with `threshold=2`; only two consecutive standstill planning outcomes terminate an environment.
- `parallelism_geometry_collision` remains the collision penalty for semantic IDs 1 and 2.
- All batch decisions use tensor masks where runtime logic is involved.

## File Map

- Create: `Go2Pvcnn/tracking/parallelism_cross_large_complex_env_cfg.py` — mixed terrain and obstacle-count configuration, training and play classes.
- Modify: `Go2Pvcnn/tracking/register_envs.py` — register training Gym ID.
- Modify: `Go2Pvcnn/Go2Pvcnn/scripts/train.py` — add training experiment mapping and imports.
- Modify: `Go2Pvcnn/Go2Pvcnn/scripts/play.py` — add play experiment mapping and parallelism-play handling.
- Modify: `Go2Pvcnn/Go2Pvcnn/agent/train_cfg.py` — accept the new experiment.
- Modify: `Go2Pvcnn/Go2Pvcnn/go2_pvcnn/mdp/curriculums.py` — add an optional tensor-mask exclusion for named terrain types if the existing curriculum cannot exclude dense flat.
- Modify: `Go2Pvcnn/Go2Pvcnn/extension/semantic_course.py` — pass terrain names into anchor/count selection only if required by the mixed-terrain count implementation.
- Create: `Go2Pvcnn/Go2Pvcnn/tests/tracking/test_parallelism_cross_large_complex_env_cfg.py` — static and lightweight config behavior tests.
- Modify: `Go2Pvcnn/Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py` — registration assertions.
- Create: `Go2Pvcnn/Go2Pvcnn/tests/tracking/parallelism_cross_large_complex_training_smoke_probe.py` — bounded 1024-environment startup probe.

### Task 1: Add failing tests for the new experiment contract

**Files:**
- Create: `Go2Pvcnn/Go2Pvcnn/tests/tracking/test_parallelism_cross_large_complex_env_cfg.py`
- Modify: `Go2Pvcnn/Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py`

**Interfaces:**
- Tests will import `ParallelismTrackingCrossLargeComplexEnvCfg` and inspect its instantiated config.
- Tests will require `parallelism_tracking_cross_large_complex` in train/play registration sources.

- [ ] **Step 1: Write failing config tests**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_cross_large_complex_config_declares_mixed_terrain_and_counts() -> None:
    source = (ROOT / "tracking/parallelism_cross_large_complex_env_cfg.py").read_text()
    assert "ParallelismTrackingSmallObstaclesEnvCfg" in source
    assert "parallelism_tracking_cross_large_complex" in source
    assert "flat_dense_small_obstacles" in source
    assert "SemanticObstacleCount(small=40, large=0)" in source
    assert "SemanticObstacleCount(small=5, large=2)" in source
    assert "proportion=0.0625" in source


def test_cross_large_complex_config_keeps_standstill_termination() -> None:
    source = (ROOT / "tracking/parallelism_cross_large_complex_env_cfg.py").read_text()
    assert "parallelism_consecutive_standstill" in source
    assert '"threshold": 2' in source


def test_cross_large_complex_config_keeps_geometry_collision_reward() -> None:
    source = (ROOT / "tracking/parallelism_cross_large_complex_env_cfg.py").read_text()
    assert "parallelism_geometry_collision" in source
    assert "obstacle_semantic_ids" in source or "semantic_ids" in source
```

- [ ] **Step 2: Add failing registration assertions**

```python
def test_registration_contains_cross_large_complex_experiment() -> None:
    train_source = (ROOT / "scripts/train.py").read_text()
    play_source = (ROOT / "scripts/play.py").read_text()
    registration_source = (ROOT / "tracking/register_envs.py").read_text()
    train_cfg_source = (ROOT / "agent/train_cfg.py").read_text()
    experiment = "parallelism_tracking_cross_large_complex"
    assert experiment in train_source
    assert experiment in play_source
    assert experiment in registration_source
    assert experiment in train_cfg_source
```

- [ ] **Step 3: Run the tests and verify the expected RED state**

Run:

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn/Go2Pvcnn
pytest -q tests/tracking/test_parallelism_cross_large_complex_env_cfg.py \
  tests/tracking/test_parallelism_tracking_registration_static.py
```

Expected: FAIL because the new config and experiment registration do not exist yet.

### Task 2: Implement the mixed terrain configuration

**Files:**
- Create: `Go2Pvcnn/tracking/parallelism_cross_large_complex_env_cfg.py`

**Interfaces:**
- Produces `ParallelismTrackingCrossLargeComplexEnvCfg`.
- Produces `ParallelismTrackingCrossLargeComplexEnvCfg_PLAY`.
- Reuses `ParallelismSmallObstaclesRewardsCfg`, `ParallelismTrackingPlaySceneCfg`, `SemanticCourseTerrainImporter`, `SemanticCourseLayoutCfg`, and existing Parallelism tracking observations/terminations/curriculum.

- [ ] **Step 1: Define the terrain generator**

Use the existing teacher terrain parameters:

```python
def _cross_large_complex_terrain_cfg():
    return terrain_gen.TerrainGeneratorCfg(
        size=SEMANTIC_TERRAIN_CFG.size,
        border_width=SEMANTIC_TERRAIN_CFG.border_width,
        num_rows=SEMANTIC_TERRAIN_CFG.num_rows,
        num_cols=SEMANTIC_TERRAIN_CFG.num_cols,
        horizontal_scale=SEMANTIC_TERRAIN_CFG.horizontal_scale,
        vertical_scale=SEMANTIC_TERRAIN_CFG.vertical_scale,
        slope_threshold=SEMANTIC_TERRAIN_CFG.slope_threshold,
        difficulty_range=SEMANTIC_TERRAIN_CFG.difficulty_range,
        curriculum=SEMANTIC_TERRAIN_CFG.curriculum,
        sub_terrains={
            "flat_dense_small_obstacles": terrain_gen.MeshPlaneTerrainCfg(proportion=0.0625),
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.0375),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.10, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
            ),
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.10, slope_range=(0.0, 0.4), platform_width=1.0, border_width=0.25
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.10, slope_range=(0.0, 0.4), platform_width=1.0, border_width=0.25
            ),
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.20, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
            ),
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.20, step_height_range=(0.05, 0.23), step_width=0.3,
                platform_width=1.0, border_width=1.0, holes=False
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.20, step_height_range=(0.05, 0.23), step_width=0.3,
                platform_width=1.0, border_width=1.0, holes=False
            ),
        },
    )
```

- [ ] **Step 2: Define obstacle counts by terrain name**

Configure `SemanticObstacleCurriculumCfg` with named terrain groups:

```python
semantic_obstacle_curriculum = SemanticObstacleCurriculumCfg(
    enabled=True,
    plane_terrain_names=("flat_dense_small_obstacles", "flat"),
    plane_counts=(SemanticObstacleCount(small=5, large=2),),
    non_plane_counts=(SemanticObstacleCount(small=5, large=2),),
    terrain_obstacle_count_overrides={
        "flat_dense_small_obstacles": SemanticObstacleCount(small=40, large=0),
    },
    center_safety_half_extent_m=(0.25,),
    min_spacing_clearance_m=(0.08,),
    tile_margin_m=(0.50,),
    collision_force_threshold=1.0,
)
```

If the current dataclass does not support `terrain_obstacle_count_overrides`, add that field and make `count_for_row` select the named override before plane/non-plane counts. Keep the selection pure and deterministic.

- [ ] **Step 3: Define the inherited RL config**

```python
@configclass
class ParallelismTrackingCrossLargeComplexEnvCfg(ParallelismTrackingSmallObstaclesEnvCfg):
    experiment_name: str = "parallelism_tracking_cross_large_complex"
    dense_small_obstacle_count: int = 40
    normal_small_obstacle_count: int = 5
    normal_large_obstacle_count: int = 2

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parallelism_tracking_cross_large_complex"
        self.scene.terrain.terrain_generator = _cross_large_complex_terrain_cfg()
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.class_type = SemanticCourseTerrainImporter
        self.scene.terrain.semantic_obstacle_curriculum = self.semantic_obstacle_curriculum
        self.scene.terrain.semantic_course_layout_cfg = SemanticCourseLayoutCfg(
            tile_margin_m=0.50,
            center_safety_half_extent_m=self.reset_clear_radius_m,
            center_safety_radius_m=self.obstacle_center_exclusion_radius_m,
            min_spacing_clearance_m=self.inner_obstacle_min_spacing_m,
        )
        assert self.terminations.parallelism_consecutive_standstill is not None
        assert self.terminations.parallelism_consecutive_standstill.params["threshold"] == 2
```

Do not set `self.terminations.parallelism_consecutive_standstill = None`. Do not disable `parallelism_geometry_collision`.

- [ ] **Step 4: Define play config**

Subclass `ParallelismTrackingCrossLargeComplexEnvCfg` with `ParallelismTrackingPlaySceneCfg(num_envs=1, ...)`, disable timeout and velocity curriculum only, and preserve every other inherited termination including `parallelism_consecutive_standstill`.

- [ ] **Step 5: Run config tests**

Run:

```bash
pytest -q tests/tracking/test_parallelism_cross_large_complex_env_cfg.py
```

Expected: PASS.

### Task 3: Implement named obstacle count overrides and dense-flat curriculum exclusion

**Files:**
- Modify: `Go2Pvcnn/extension/semantic_curriculum.py`
- Modify: `Go2Pvcnn/Go2Pvcnn/go2_pvcnn/mdp/curriculums.py`
- Modify: `Go2Pvcnn/extension/semantic_course.py` only if its caller needs to pass terrain names explicitly.

**Interfaces:**
- `count_for_row(cfg, row, terrain_name)` returns the explicit terrain-name override first.
- `terrain_levels_vel_semantic_plane_gate(..., excluded_terrain_names=...)` leaves excluded environments' terrain level and origin unchanged while updating normal environments with the existing tensor masks.

- [ ] **Step 1: Add failing unit tests for named count override**

```python
def test_named_terrain_count_override_wins_over_plane_count() -> None:
    cfg = SemanticObstacleCurriculumCfg(
        plane_terrain_names=("flat",),
        plane_counts=(SemanticObstacleCount(small=5, large=2),),
        non_plane_counts=(SemanticObstacleCount(small=5, large=2),),
        terrain_obstacle_count_overrides={
            "flat_dense_small_obstacles": SemanticObstacleCount(small=40, large=0),
        },
    )
    assert count_for_row(cfg, row=0, terrain_name="flat_dense_small_obstacles") == SemanticObstacleCount(
        small=40, large=0
    )
    assert count_for_row(cfg, row=0, terrain_name="boxes") == SemanticObstacleCount(small=5, large=2)
```

- [ ] **Step 2: Run the unit test and verify RED**

Run:

```bash
pytest -q tests/test_semantic_obstacle_curriculum.py -k named_terrain_count_override
```

Expected: FAIL because the override field does not exist.

- [ ] **Step 3: Implement the override**

Add:

```python
terrain_obstacle_count_overrides: dict[str, SemanticObstacleCount] = field(default_factory=dict)
```

Validate keys and values, then update:

```python
def count_for_row(cfg, *, row, terrain_name):
    if terrain_name is not None and terrain_name in cfg.terrain_obstacle_count_overrides:
        return cfg.terrain_obstacle_count_overrides[terrain_name]
    ...
```

- [ ] **Step 4: Add excluded terrain names to the curriculum term**

The new config should call the existing curriculum term with:

```python
params={
    "cfg_name": "semantic_obstacle_curriculum",
    "excluded_terrain_names": ("flat_dense_small_obstacles",),
}
```

Inside `terrain_levels_vel_semantic_plane_gate`, compute:

```python
excluded_all = terrain_name_mask(
    torch.as_tensor(terrain.terrain_types, dtype=torch.long, device=device),
    terrain_names,
    tuple(excluded_terrain_names),
)
active = ~excluded_all[env_ids_t]
```

Run the existing `terrain.update_env_origins` only for `env_ids_t[active]`; return the same metric and leave excluded terrain IDs untouched. Avoid Python loops over environments.

- [ ] **Step 5: Run curriculum tests**

Run:

```bash
pytest -q tests/test_semantic_obstacle_curriculum.py tests/test_semantic_obstacle_curriculum_term.py
```

Expected: PASS.

### Task 4: Register training and play entry points

**Files:**
- Modify: `Go2Pvcnn/tracking/register_envs.py`
- Modify: `Go2Pvcnn/Go2Pvcnn/scripts/train.py`
- Modify: `Go2Pvcnn/Go2Pvcnn/scripts/play.py`
- Modify: `Go2Pvcnn/Go2Pvcnn/agent/train_cfg.py`
- Modify: `Go2Pvcnn/Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py`

**Interfaces:**
- Gym ID: `Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-v0`.
- Experiment string: `parallelism_tracking_cross_large_complex`.

- [ ] **Step 1: Add the new config imports and maps**

Add the new config and play config to the existing maps in `train.py` and `play.py`, and add the experiment string to `agent/train_cfg.py`.

- [ ] **Step 2: Register the Gym task**

Add a `gym.register` entry using `ParallelismTrackingCrossLargeComplexEnvCfg`.

- [ ] **Step 3: Preserve parallelism play behavior**

Include the new experiment in the same `is_parallelism_play` tuple and any branch that attaches the reference manager.

- [ ] **Step 4: Run static registration tests**

Run:

```bash
pytest -q tests/tracking/test_parallelism_tracking_registration_static.py \
  tests/test_train_script_static.py
```

Expected: PASS.

### Task 5: Add a bounded 1024-environment smoke probe

**Files:**
- Create: `Go2Pvcnn/Go2Pvcnn/tests/tracking/parallelism_cross_large_complex_training_smoke_probe.py`

**Interfaces:**
- Command-line options: `--num-envs`, `--iterations`, `--device`.
- Uses `scripts/train.py` with `--experiment parallelism_tracking_cross_large_complex`, `--headless`, and a bounded iteration count.

- [ ] **Step 1: Implement the probe**

The probe should launch:

```bash
python scripts/train.py \
  --experiment parallelism_tracking_cross_large_complex \
  --num_envs 1024 \
  --max_iterations 4 \
  --headless \
  --device cuda:0
```

Capture stdout/stderr and fail on non-zero exit, traceback, or missing `Learning iteration`.

- [ ] **Step 2: Run config-only tests before Isaac Sim**

Run:

```bash
pytest -q tests/tracking/test_parallelism_cross_large_complex_env_cfg.py \
  tests/test_semantic_obstacle_curriculum.py \
  tests/test_semantic_obstacle_curriculum_term.py
```

Expected: PASS.

- [ ] **Step 3: Run the 1024-environment smoke test**

Run:

```bash
python tests/tracking/parallelism_cross_large_complex_training_smoke_probe.py \
  --num-envs 1024 --iterations 4 --device cuda:0
```

Expected: process exits 0 and prints at least four learning iterations without reset/planner/reward exceptions.

### Task 6: Verify, document, and commit

**Files:**
- Modify: `docs/superpowers/specs/2026-08-12-parallelism-cross-large-complex-rl-design.html` — already updated to document `parallelism_consecutive_standstill`.

- [ ] **Step 1: Run the focused test suite**

```bash
pytest -q \
  tests/tracking/test_parallelism_cross_large_complex_env_cfg.py \
  tests/tracking/test_parallelism_tracking_registration_static.py \
  tests/tracking/test_parallelism_small_obstacles_env_cfg_static.py \
  tests/test_semantic_obstacle_curriculum.py \
  tests/test_semantic_obstacle_curriculum_term.py
```

- [ ] **Step 2: Review the diff**

```bash
git diff --check
git diff --stat
git status --short
```

Confirm no existing experiment was modified unintentionally and confirm the new config still contains:

```python
self.terminations.parallelism_consecutive_standstill.params["threshold"] == 2
```

- [ ] **Step 3: Commit**

```bash
git add \
  Go2Pvcnn/tracking/parallelism_cross_large_complex_env_cfg.py \
  Go2Pvcnn/tracking/register_envs.py \
  Go2Pvcnn/scripts/train.py \
  Go2Pvcnn/scripts/play.py \
  Go2Pvcnn/agent/train_cfg.py \
  Go2Pvcnn/extension/semantic_curriculum.py \
  Go2Pvcnn/go2_pvcnn/mdp/curriculums.py \
  Go2Pvcnn/extension/semantic_course.py \
  Go2Pvcnn/tests \
  docs/superpowers/specs/2026-08-12-parallelism-cross-large-complex-rl-design.html \
  docs/superpowers/plans/2026-08-12-parallelism-cross-large-complex-rl.md
git commit -m "feat: add cross large complex terrain parallelism rl"
```

- [ ] **Step 4: Verify commit**

```bash
git status --short --branch
git log -1 --oneline
```

Expected: clean worktree, current branch `parallelism-large-obstacles-rl`, and the new feature commit at `HEAD`.
