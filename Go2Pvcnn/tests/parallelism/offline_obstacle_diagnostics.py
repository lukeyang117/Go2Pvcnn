"""Pure-Torch, YAML-driven diagnostics for Parallelism obstacle failures."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml

from extension.parallelism import ParallelismCfg, ParallelismState, ParallelismTerrain
from extension.parallelism.kinematics import fk_go2, rpy_to_rotation_matrix
from extension.parallelism.planner import plan_trajectory
from extension.parallelism.terrain import query_height_semantic_valid


@dataclass(frozen=True)
class _Obstacle:
    shape: str
    center_w: tuple[float, float]
    radius_m: float
    height_m: float
    semantic_id: int


@dataclass(frozen=True)
class _Terrain:
    resolution_m: float
    size: int
    origin_w: tuple[float, float, float]
    yaw_w: float


@dataclass(frozen=True)
class _Root:
    position_w: tuple[float, float, float]
    rpy_w: tuple[float, float, float]


@dataclass(frozen=True)
class OfflineObstacleScene:
    name: str
    terrain: _Terrain
    obstacle: _Obstacle
    root: _Root
    joint_pos: tuple[float, ...]
    commands: tuple[tuple[float, float, float], ...]
    planner: dict[str, Any]


def _tuple(values: Any, length: int, field: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{field} must contain {length} values")
    return tuple(float(value) for value in values)


def load_scene(path: Path, scene_name: str) -> OfflineObstacleScene:
    payload = yaml.safe_load(Path(path).read_text())
    scenes = payload.get("scenes") if isinstance(payload, dict) else None
    if not isinstance(scenes, dict) or scene_name not in scenes:
        raise ValueError(f"scene {scene_name!r} is missing from {path}")
    data = scenes[scene_name]
    terrain_data = data["terrain"]
    obstacle_data = data["obstacle"]
    root_data = data["root"]
    terrain = _Terrain(
        resolution_m=float(terrain_data["resolution_m"]),
        size=int(terrain_data["size"]),
        origin_w=_tuple(terrain_data["origin_w"], 3, "terrain.origin_w"),
        yaw_w=float(terrain_data.get("yaw_w", 0.0)),
    )
    if terrain.resolution_m <= 0.0 or terrain.size < 3:
        raise ValueError("terrain resolution_m must be positive and size must be at least 3")
    obstacle = _Obstacle(
        shape=str(obstacle_data["shape"]),
        center_w=_tuple(obstacle_data["center_w"], 2, "obstacle.center_w"),
        radius_m=float(obstacle_data["radius_m"]),
        height_m=float(obstacle_data["height_m"]),
        semantic_id=int(obstacle_data["semantic_id"]),
    )
    if obstacle.shape not in {"circle", "cylinder", "cuboid"}:
        raise ValueError(f"unsupported obstacle.shape: {obstacle.shape}")
    if obstacle.radius_m <= 0.0 or obstacle.height_m < 0.0:
        raise ValueError("obstacle radius_m must be positive and height_m must be non-negative")
    root = _Root(
        position_w=_tuple(root_data["position_w"], 3, "root.position_w"),
        rpy_w=_tuple(root_data["rpy_w"], 3, "root.rpy_w"),
    )
    joints = _tuple(data["joint_pos"], 12, "joint_pos")
    commands = tuple(_tuple(command, 3, "commands[]") for command in data["commands"])
    if not commands:
        raise ValueError("commands must contain at least one command")
    return OfflineObstacleScene(
        name=scene_name,
        terrain=terrain,
        obstacle=obstacle,
        root=root,
        joint_pos=joints,
        commands=commands,
        planner=dict(data.get("planner", {})),
    )


def build_terrain(scene: OfflineObstacleScene, device: torch.device) -> ParallelismTerrain:
    terrain = scene.terrain
    obstacle = scene.obstacle
    dtype = torch.float32
    origin = torch.tensor([terrain.origin_w], dtype=dtype, device=device)
    axis = torch.arange(terrain.size, dtype=dtype, device=device) * terrain.resolution_m
    grid_y, grid_x = torch.meshgrid(
        axis + origin[0, 0],
        axis + origin[0, 1],
        indexing="ij",
    )
    delta_x = grid_x - obstacle.center_w[0]
    delta_y = grid_y - obstacle.center_w[1]
    if obstacle.shape in {"circle", "cylinder"}:
        occupied = delta_x.square() + delta_y.square() <= obstacle.radius_m**2
    else:
        occupied = delta_x.abs() <= obstacle.radius_m
        occupied = occupied & (delta_y.abs() <= obstacle.radius_m)
    height = torch.where(
        occupied,
        torch.full_like(grid_x, obstacle.height_m),
        torch.zeros_like(grid_x),
    ).unsqueeze(0)
    semantic = torch.where(
        occupied,
        torch.full_like(grid_x, obstacle.semantic_id, dtype=torch.long),
        torch.zeros_like(grid_x, dtype=torch.long),
    ).unsqueeze(0)
    return ParallelismTerrain(
        height_w=height,
        semantic_id=semantic,
        valid_mask=torch.ones_like(semantic, dtype=torch.bool),
        origin_w=origin,
        yaw_w=torch.tensor([terrain.yaw_w], dtype=dtype, device=device),
        resolution=terrain.resolution_m,
    )


def build_state(scene: OfflineObstacleScene, device: torch.device) -> ParallelismState:
    root_pos = torch.tensor([scene.root.position_w], dtype=torch.float32, device=device)
    root_rpy = torch.tensor([scene.root.rpy_w], dtype=torch.float32, device=device)
    joint_pos = torch.tensor([scene.joint_pos], dtype=torch.float32, device=device)
    foot_pos = fk_go2(root_pos, root_rpy, joint_pos).foot_pos_w
    return ParallelismState(
        root_pos_w=root_pos,
        root_rpy_w=root_rpy,
        joint_pos=joint_pos,
        foot_pos_w=foot_pos,
    )


def obstacle_geometry(scene: OfflineObstacleScene, device: torch.device) -> dict[str, list[float]]:
    center_w = torch.tensor(
        [scene.obstacle.center_w[0], scene.obstacle.center_w[1], 0.5 * scene.obstacle.height_m],
        dtype=torch.float32,
        device=device,
    )
    root_pos = torch.tensor([scene.root.position_w], dtype=torch.float32, device=device)
    root_rpy = torch.tensor([scene.root.rpy_w], dtype=torch.float32, device=device)
    rotation = rpy_to_rotation_matrix(root_rpy)
    center_root = torch.matmul(rotation.transpose(-1, -2), (center_w - root_pos).unsqueeze(-1)).squeeze(-1)
    radius = scene.obstacle.radius_m
    min_w = [scene.obstacle.center_w[0] - radius, scene.obstacle.center_w[1] - radius, 0.0]
    max_w = [scene.obstacle.center_w[0] + radius, scene.obstacle.center_w[1] + radius, scene.obstacle.height_m]
    return {
        "center_w": center_w.cpu().tolist(),
        "min_w": min_w,
        "max_w": max_w,
        "center_root": center_root[0].cpu().tolist(),
    }


def _planner_cfg(scene: OfflineObstacleScene) -> ParallelismCfg:
    allowed = {
        "candidate_radius_m",
        "candidates_per_leg",
        "swing_clearance_m",
        "semantic_touchdown_margin_m",
        "foothold_step_gain",
    }
    values = {key: value for key, value in scene.planner.items() if key in allowed}
    return replace(ParallelismCfg(), **values)


def _write_failure_snapshot(
    scene: OfflineObstacleScene,
    report: dict[str, Any],
    *,
    snapshot_dir: Path,
) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    command = report["command"]
    obstacle = scene.obstacle
    root_yaw = scene.root.rpy_w[2]
    tags = (
        f"vx{float(command[0]):+.2f}",
        f"vy{float(command[1]):+.2f}",
        f"yaw{float(command[2]):+.2f}",
        f"ox{obstacle.center_w[0]:+.2f}",
        f"oy{obstacle.center_w[1]:+.2f}",
        f"or{obstacle.radius_m:.2f}",
        f"oh{obstacle.height_m:.2f}",
        f"ry{root_yaw:+.2f}",
    )
    filename = "__".join(
        (scene.name, *tags)
    ).replace("+", "p").replace("-", "m").replace(".", "d") + ".yaml"
    output_path = snapshot_dir / filename
    payload = {
        "captured_from": scene.name,
        "failure": {
            "standstill": report["standstill"],
            "valid": report["valid"],
            "per_leg_valid": report["per_leg_valid"],
            "valid_count": report["valid_count"],
            "reject_counts": report["reject_counts"],
            "collision_counts": report["collision_counts"],
        },
        "terrain": {
            "resolution_m": scene.terrain.resolution_m,
            "size": scene.terrain.size,
            "origin_w": list(scene.terrain.origin_w),
            "yaw_w": scene.terrain.yaw_w,
        },
        "obstacle": {
            "shape": scene.obstacle.shape,
            "center_w": list(scene.obstacle.center_w),
            "radius_m": scene.obstacle.radius_m,
            "height_m": scene.obstacle.height_m,
            "semantic_id": scene.obstacle.semantic_id,
            "geometry": report["obstacle"],
        },
        "root": {
            "position_w": list(scene.root.position_w),
            "rpy_w": list(scene.root.rpy_w),
        },
        "joint_pos": list(scene.joint_pos),
        "commands": [list(command)],
        "planner": dict(scene.planner),
    }
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return output_path


def run_scene(
    scene: OfflineObstacleScene,
    command_index: int = 0,
    device: torch.device | None = None,
    snapshot_dir: Path | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    terrain = build_terrain(scene, device)
    state = build_state(scene, device)
    command = torch.tensor([scene.commands[command_index]], dtype=torch.float32, device=device)
    trajectory = plan_trajectory(state, command, terrain, _planner_cfg(scene))
    diagnostics = trajectory.diagnostics
    valid = diagnostics.candidate_valid[0]
    reject = diagnostics.candidate_reject_bits[0]
    collision = diagnostics.candidate_collision_bits[0]
    root_static = torch.allclose(trajectory.root_pos_w, trajectory.root_pos_w[:, :1], atol=1e-6, rtol=1e-6)
    reject_names = ("valid_map", "joint", "landing", "collision", "candidate_semantic", "fk_touchdown_semantic")
    reject_counts = {name: int(value) for name, value in zip(reject_names, reject.sum(dim=(0, 1)).tolist())}
    collision_counts = {
        name: int(value)
        for name, value in zip(diagnostics.collision_shape_names, collision.sum(dim=(0, 1)).tolist())
    }
    report = {
        "scene": scene.name,
        "command": scene.commands[command_index],
        "device": str(device),
        "obstacle": obstacle_geometry(scene, device),
        "root_pos_w": scene.root.position_w,
        "root_rpy_w": scene.root.rpy_w,
        "joint_pos": scene.joint_pos,
        "current_foot_pos_w": state.foot_pos_w[0].detach().cpu().tolist(),
        "valid": bool(trajectory.valid[0].item()),
        "standstill": bool(root_static),
        "per_leg_valid": [int(value) for value in valid.sum(dim=-1).tolist()],
        "valid_count": int(valid.sum().item()),
        "reject_counts": reject_counts,
        "collision_counts": collision_counts,
        "selected_index": diagnostics.selected_index[0].detach().cpu().tolist(),
    }
    if snapshot_dir is not None and (report["standstill"] or min(report["per_leg_valid"]) == 0):
        report["snapshot_path"] = str(_write_failure_snapshot(scene, report, snapshot_dir=snapshot_dir))
    return report


def sweep_scene(scene: OfflineObstacleScene, overrides: dict[str, list[Any]], device: torch.device | None = None) -> list[dict[str, Any]]:
    keys = tuple(overrides)
    reports = []
    for values in itertools.product(*(overrides[key] for key in keys)):
        current = scene
        for key, value in zip(keys, values):
            if key == "obstacle.center_w":
                current = replace(current, obstacle=replace(current.obstacle, center_w=_tuple(value, 2, key)))
            elif key == "obstacle.height_m":
                current = replace(current, obstacle=replace(current.obstacle, height_m=float(value)))
            elif key == "root.rpy_w":
                current = replace(current, root=replace(current.root, rpy_w=_tuple(value, 3, key)))
            elif key == "planner.swing_clearance_m":
                current = replace(current, planner={**current.planner, "swing_clearance_m": float(value)})
            else:
                raise ValueError(f"unsupported sweep key: {key}")
        reports.append(run_scene(current, device=device))
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-file", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path(__file__).with_name("situations"),
        help="Directory for YAML snapshots of standstill or zero-valid-leg cases.",
    )
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else None
    scene = load_scene(args.scene_file, args.scene)
    reports = [
        run_scene(scene, index, device=device, snapshot_dir=args.snapshot_dir)
        for index in range(len(scene.commands))
    ]
    payload = json.dumps(reports, indent=2, ensure_ascii=True)
    if args.output is not None:
        args.output.write_text(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
