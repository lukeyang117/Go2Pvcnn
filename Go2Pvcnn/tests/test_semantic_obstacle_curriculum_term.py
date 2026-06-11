from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))


def _install_fake_isaaclab(monkeypatch) -> None:
    isaaclab_module = types.ModuleType("isaaclab")
    managers_module = types.ModuleType("isaaclab.managers")

    class SceneEntityCfg:
        def __init__(self, name: str, **kwargs):
            self.name = name
            for key, value in kwargs.items():
                setattr(self, key, value)

    managers_module.SceneEntityCfg = SceneEntityCfg
    isaaclab_module.managers = managers_module
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers_module)


class _Data:
    def __init__(self, force_matrix_w: torch.Tensor):
        self.force_matrix_w = force_matrix_w


class _Sensor:
    def __init__(self, force_matrix_w: torch.Tensor):
        self.data = _Data(force_matrix_w)


class _Sensors(dict):
    pass


class _TerrainGenerator:
    sub_terrains = {"flat": object(), "boxes": object()}
    size = (8.0, 8.0)


class _TerrainCfg:
    terrain_generator = _TerrainGenerator()


class _Terrain:
    def __init__(self, terrain_types: torch.Tensor):
        self.terrain_types = terrain_types
        self.terrain_levels = torch.zeros_like(terrain_types)
        self.max_terrain_level = 10
        self.terrain_origins = torch.zeros(10, 2, 3)
        for row in range(10):
            for col in range(2):
                self.terrain_origins[row, col] = torch.tensor([float(row) * 8.0, float(col) * 8.0, 0.0])
        self.env_origins = self.terrain_origins[self.terrain_levels, self.terrain_types].clone()
        self.cfg = _TerrainCfg()
        self.update_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def update_env_origins(self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long)
        move_up = torch.as_tensor(move_up, dtype=torch.bool)
        move_down = torch.as_tensor(move_down, dtype=torch.bool)
        self.update_calls.append((env_ids.clone(), move_up.clone(), move_down.clone()))
        self.terrain_levels[env_ids] += 1 * move_up.to(torch.long) - 1 * move_down.to(torch.long)
        self.terrain_levels[env_ids] = torch.clamp(self.terrain_levels[env_ids], 0, self.max_terrain_level - 1)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]


class _RobotData:
    def __init__(self, root_pos_w: torch.Tensor):
        self.root_pos_w = root_pos_w


class _Robot:
    def __init__(self, root_pos_w: torch.Tensor):
        self.data = _RobotData(root_pos_w)


class _Scene:
    def __init__(self, terrain_types: torch.Tensor, small: torch.Tensor, large: torch.Tensor, root_pos_w: torch.Tensor):
        self.terrain = _Terrain(terrain_types)
        self.env_origins = self.terrain.env_origins
        self.robot = _Robot(root_pos_w)
        self.sensors = _Sensors(
            semantic_contact_small=_Sensor(small),
            semantic_contact_large=_Sensor(large),
        )

    def __getitem__(self, name: str):
        return getattr(self, name)


class _Env:
    def __init__(
        self,
        *,
        terrain_types: torch.Tensor,
        small: torch.Tensor,
        large: torch.Tensor,
        cfg,
        root_pos_w: torch.Tensor | None = None,
        command: torch.Tensor | None = None,
        time_out: torch.Tensor | None = None,
        base_contact: torch.Tensor | None = None,
        bad_orientation: torch.Tensor | None = None,
    ):
        self.device = "cpu"
        self.num_envs = int(terrain_types.numel())
        if root_pos_w is None:
            root_pos_w = torch.zeros(self.num_envs, 3)
            root_pos_w[:, 0] = 5.0
        if command is None:
            command = torch.ones(self.num_envs, 3)
        self.scene = _Scene(terrain_types, small, large, root_pos_w)
        self.cfg = cfg
        self.unwrapped = self
        self.max_episode_length_s = 20.0
        self.command_manager = types.SimpleNamespace(get_command=lambda _name: command)
        self.time_out_buf = torch.ones(self.num_envs, dtype=torch.bool) if time_out is None else time_out
        self.base_contact_buf = (
            torch.zeros(self.num_envs, dtype=torch.bool) if base_contact is None else base_contact
        )
        self.bad_orientation_buf = (
            torch.zeros(self.num_envs, dtype=torch.bool) if bad_orientation is None else bad_orientation
        )


class _Cfg:
    def __init__(self, semantic_obstacle_curriculum):
        self.semantic_obstacle_curriculum = semantic_obstacle_curriculum


def _force(num_envs: int) -> torch.Tensor:
    return torch.zeros(num_envs, 1, 1, 3)


def _load_curriculums_module(monkeypatch):
    _install_fake_isaaclab(monkeypatch)
    module_path = GO2PVCNN_ROOT / "go2_pvcnn/mdp/curriculums.py"
    spec = importlib.util.spec_from_file_location("_test_go2_curriculums", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_test_go2_curriculums"] = module
    spec.loader.exec_module(module)
    return module


def test_semantic_collision_mask_from_force_matrices(monkeypatch) -> None:
    curriculums = _load_curriculums_module(monkeypatch)

    small = _force(3)
    large = _force(3)
    small[1, 0, 0, 0] = 2.0
    large[2, 0, 0, 1] = 3.0

    mask = curriculums.semantic_collision_mask_from_force_matrices(small, large, threshold=1.0)

    assert mask.tolist() == [False, True, True]


def test_plane_env_mask_from_terrain(monkeypatch) -> None:
    curriculums = _load_curriculums_module(monkeypatch)

    mask = curriculums.plane_env_mask_from_terrain(
        torch.tensor([0, 1, 0, 1]),
        ("flat", "boxes"),
        ("flat",),
    )

    assert mask.tolist() == [True, False, True, False]


def test_plane_env_mask_treats_single_flat_subterrain_columns_as_flat(monkeypatch) -> None:
    curriculums = _load_curriculums_module(monkeypatch)

    terrain_types = torch.arange(20, dtype=torch.long)

    mask = curriculums.plane_env_mask_from_terrain(
        terrain_types,
        ("flat",),
        ("flat",),
    )

    assert mask.tolist() == [True] * 20


def test_env_level_curriculum_single_successful_flat_episode_moves_up(monkeypatch) -> None:
    from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg

    curriculums = _load_curriculums_module(monkeypatch)

    cfg = SemanticObstacleCurriculumCfg(
        plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        non_plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        center_safety_half_extent_m=(0.8, 0.4),
        min_spacing_clearance_m=(0.2, 0.1),
        tile_margin_m=(0.5, 0.4),
    )
    small = _force(4)
    large = _force(4)
    env = _Env(
        terrain_types=torch.tensor([0, 1, 0, 1]),
        small=small,
        large=large,
        cfg=_Cfg(cfg),
    )

    out = curriculums.terrain_levels_vel_semantic_plane_gate(env, [0])

    assert set(out) == {"mean_terrain_level"}
    assert env.scene.terrain.terrain_levels.tolist() == [1, 0, 0, 0]
    assert not hasattr(env, "_semantic_obstacle_curriculum_level")
    assert not hasattr(env.scene.terrain.cfg, "semantic_obstacle_curriculum_level")


def test_env_level_curriculum_small_collision_blocks_flat_move_up_only(monkeypatch) -> None:
    from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg

    curriculums = _load_curriculums_module(monkeypatch)

    cfg = SemanticObstacleCurriculumCfg(
        plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        non_plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        center_safety_half_extent_m=(0.8, 0.4),
        min_spacing_clearance_m=(0.2, 0.1),
        tile_margin_m=(0.5, 0.4),
    )
    small = _force(4)
    small[0, 0, 0, 0] = 4.0
    env = _Env(
        terrain_types=torch.tensor([0, 1, 0, 1]),
        small=small,
        large=_force(4),
        cfg=_Cfg(cfg),
    )

    out = curriculums.terrain_levels_vel_semantic_plane_gate(env, [0, 1, 2, 3])

    assert set(out) == {"mean_terrain_level"}
    assert env.scene.terrain.terrain_levels.tolist() == [0, 1, 1, 1]
    _, move_up, move_down = env.scene.terrain.update_calls[-1]
    assert move_up.tolist() == [False, True, True, True]
    assert move_down.tolist() == [False, False, False, False]


def test_terrain_gate_accepts_slice_env_ids(monkeypatch) -> None:
    from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg

    curriculums = _load_curriculums_module(monkeypatch)

    cfg = SemanticObstacleCurriculumCfg(
        plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        non_plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        center_safety_half_extent_m=(0.8, 0.4),
        min_spacing_clearance_m=(0.2, 0.1),
        tile_margin_m=(0.5, 0.4),
    )
    env = _Env(
        terrain_types=torch.tensor([0, 1, 0, 1]),
        small=_force(4),
        large=_force(4),
        cfg=_Cfg(cfg),
    )

    out = curriculums.terrain_levels_vel_semantic_plane_gate(env, slice(None))

    assert set(out) == {"mean_terrain_level"}
    assert env.scene.terrain.terrain_levels.tolist() == [1, 1, 1, 1]


def test_env_level_curriculum_base_contact_and_bad_orientation_force_flat_move_down(monkeypatch) -> None:
    from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg

    curriculums = _load_curriculums_module(monkeypatch)

    cfg = SemanticObstacleCurriculumCfg(
        plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        non_plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        center_safety_half_extent_m=(0.8, 0.4),
        min_spacing_clearance_m=(0.2, 0.1),
        tile_margin_m=(0.5, 0.4),
    )
    root_pos_w = torch.zeros(4, 3)
    root_pos_w[:, 0] = 1.0
    env = _Env(
        terrain_types=torch.tensor([0, 0, 1, 1]),
        small=_force(4),
        large=_force(4),
        cfg=_Cfg(cfg),
        root_pos_w=root_pos_w,
        base_contact=torch.tensor([True, False, False, False]),
        bad_orientation=torch.tensor([False, True, False, False]),
    )
    env.scene.terrain.terrain_levels[:] = 2

    out = curriculums.terrain_levels_vel_semantic_plane_gate(env, [0, 1, 2, 3])

    assert set(out) == {"mean_terrain_level"}
    _, move_up, move_down = env.scene.terrain.update_calls[-1]
    assert move_up.tolist() == [False, False, True, True]
    assert move_down.tolist() == [True, True, False, False]


def test_env_level_curriculum_requires_timeout_for_flat_move_up(monkeypatch) -> None:
    from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg

    curriculums = _load_curriculums_module(monkeypatch)

    cfg = SemanticObstacleCurriculumCfg(
        plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        non_plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        center_safety_half_extent_m=(0.8, 0.4),
        min_spacing_clearance_m=(0.2, 0.1),
        tile_margin_m=(0.5, 0.4),
    )
    env = _Env(
        terrain_types=torch.tensor([0, 0, 1, 1]),
        small=_force(4),
        large=_force(4),
        cfg=_Cfg(cfg),
        time_out=torch.tensor([True, False, True, True]),
        base_contact=torch.tensor([False, False, False, False]),
        bad_orientation=torch.tensor([False, True, False, False]),
    )

    out = curriculums.terrain_levels_vel_semantic_plane_gate(env, [0, 1, 2, 3])

    assert set(out) == {"mean_terrain_level"}
    assert env.scene.terrain.terrain_levels.tolist() == [1, 0, 1, 1]


def test_terrain_gate_counts_only_passed_env_ids_as_completed_episodes(monkeypatch) -> None:
    from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg

    curriculums = _load_curriculums_module(monkeypatch)

    cfg = SemanticObstacleCurriculumCfg(
        plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        non_plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        center_safety_half_extent_m=(0.8, 0.4),
        min_spacing_clearance_m=(0.2, 0.1),
        tile_margin_m=(0.5, 0.4),
    )
    env = _Env(
        terrain_types=torch.tensor([0, 0, 0, 1]),
        small=_force(4),
        large=_force(4),
        cfg=_Cfg(cfg),
        time_out=torch.tensor([True, True, True, True]),
    )

    out = curriculums.terrain_levels_vel_semantic_plane_gate(env, [0, 2])

    assert set(out) == {"mean_terrain_level"}
    assert env.scene.terrain.terrain_levels.tolist() == [1, 0, 1, 0]
