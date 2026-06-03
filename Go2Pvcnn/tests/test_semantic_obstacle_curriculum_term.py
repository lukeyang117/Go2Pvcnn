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


def test_terrain_gate_blocks_flat_move_up_but_not_non_flat_when_gate_closed(monkeypatch) -> None:
    from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg

    curriculums = _load_curriculums_module(monkeypatch)

    cfg = SemanticObstacleCurriculumCfg(
        plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        non_plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        center_safety_half_extent_m=(0.8, 0.4),
        min_spacing_clearance_m=(0.2, 0.1),
        tile_margin_m=(0.5, 0.4),
        plane_collision_rate_threshold=0.25,
        consecutive_success_required=2,
    )
    small = _force(4)
    large = _force(4)
    small[1, 0, 0, 0] = 4.0  # non-flat collision must not count
    env = _Env(
        terrain_types=torch.tensor([0, 1, 0, 1]),
        small=small,
        large=large,
        cfg=_Cfg(cfg),
    )

    out = curriculums.terrain_levels_vel_semantic_plane_gate(env, [0, 1, 2, 3])

    assert out["plane_env_count"].item() == 2.0
    assert out["plane_collision_rate"].item() == 0.0
    assert out["consecutive_success_count"].item() == 1.0
    assert out["semantic_gate_pass"].item() == 0.0
    assert env.scene.terrain.terrain_levels.tolist() == [0, 1, 0, 1]
    assert not hasattr(env, "_semantic_obstacle_curriculum_level")
    assert not hasattr(env.scene.terrain.cfg, "semantic_obstacle_curriculum_level")


def test_terrain_gate_allows_flat_move_up_after_consecutive_success(monkeypatch) -> None:
    from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg

    curriculums = _load_curriculums_module(monkeypatch)

    cfg = SemanticObstacleCurriculumCfg(
        plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        non_plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        center_safety_half_extent_m=(0.8, 0.4),
        min_spacing_clearance_m=(0.2, 0.1),
        tile_margin_m=(0.5, 0.4),
        plane_collision_rate_threshold=0.25,
        consecutive_success_required=2,
    )
    env = _Env(
        terrain_types=torch.tensor([0, 1, 0, 1]),
        small=_force(4),
        large=_force(4),
        cfg=_Cfg(cfg),
    )

    curriculums.terrain_levels_vel_semantic_plane_gate(env, [0, 1, 2, 3])
    out = curriculums.terrain_levels_vel_semantic_plane_gate(env, [0, 1, 2, 3])

    assert out["semantic_gate_pass"].item() == 1.0
    assert env.scene.terrain.terrain_levels.tolist() == [1, 2, 1, 2]
    assert out["flat_move_up_count"].item() == 2.0
    assert out["non_flat_move_up_count"].item() == 2.0


def test_terrain_gate_accepts_slice_env_ids(monkeypatch) -> None:
    from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg

    curriculums = _load_curriculums_module(monkeypatch)

    cfg = SemanticObstacleCurriculumCfg(
        plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        non_plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        center_safety_half_extent_m=(0.8, 0.4),
        min_spacing_clearance_m=(0.2, 0.1),
        tile_margin_m=(0.5, 0.4),
        plane_collision_rate_threshold=0.25,
        consecutive_success_required=1,
    )
    env = _Env(
        terrain_types=torch.tensor([0, 1, 0, 1]),
        small=_force(4),
        large=_force(4),
        cfg=_Cfg(cfg),
    )

    out = curriculums.terrain_levels_vel_semantic_plane_gate(env, slice(None))

    assert out["semantic_gate_pass"].item() == 1.0
    assert env.scene.terrain.terrain_levels.tolist() == [1, 1, 1, 1]


def test_terrain_gate_flat_collision_resets_success_and_blocks_flat(monkeypatch) -> None:
    from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg

    curriculums = _load_curriculums_module(monkeypatch)

    cfg = SemanticObstacleCurriculumCfg(
        plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        non_plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        center_safety_half_extent_m=(0.8, 0.4),
        min_spacing_clearance_m=(0.2, 0.1),
        tile_margin_m=(0.5, 0.4),
        plane_collision_rate_threshold=0.25,
        consecutive_success_required=2,
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

    assert out["plane_collision_rate"].item() == 0.5
    assert out["consecutive_success_count"].item() == 0.0
    assert out["semantic_gate_pass"].item() == 0.0
    assert env.scene.terrain.terrain_levels.tolist() == [0, 1, 0, 1]
