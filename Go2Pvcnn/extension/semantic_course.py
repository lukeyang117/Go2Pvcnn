"""Static semantic-course helpers for the trajectory viewer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any


SEMANTIC_COURSE_ROOT = "/World/semantic_course"
SEMANTIC_COURSE_SMALL_ROOT = f"{SEMANTIC_COURSE_ROOT}/small"
SEMANTIC_COURSE_LARGE_ROOT = f"{SEMANTIC_COURSE_ROOT}/large"
SEMANTIC_COURSE_ROOTS = (SEMANTIC_COURSE_SMALL_ROOT, SEMANTIC_COURSE_LARGE_ROOT)

SMALL_OBSTACLE_SIZE = (0.12, 0.12, 0.22)
LARGE_OBSTACLE_SIZE = (0.45, 0.45, 0.55)


class SemanticCourseStage(str, Enum):
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"


DEFAULT_VIEWER_REPRESENTATIVE_STAGE = SemanticCourseStage.S4


@dataclass(frozen=True)
class CourseAnchor:
    row: int
    col: int
    stage: SemanticCourseStage
    semantic_class: str
    slot_index: int
    local_xy: tuple[float, float]
    size: tuple[float, float, float]
    world_xy: tuple[float, float]
    prim_path: str


@dataclass(frozen=True)
class GroundedCourseObstacle:
    row: int
    col: int
    stage: SemanticCourseStage
    semantic_class: str
    slot_index: int
    local_xy: tuple[float, float]
    size: tuple[float, float, float]
    world_center: tuple[float, float, float]
    prim_path: str


_STAGE_LAYOUTS: dict[SemanticCourseStage, dict[str, tuple[tuple[float, float], ...]]] = {
    SemanticCourseStage.S1: {"small": (), "large": ()},
    SemanticCourseStage.S2: {
        "small": ((0.35, 0.35), (0.35, -0.35), (0.65, 0.20), (0.65, -0.20)),
        "large": (),
    },
    SemanticCourseStage.S3: {
        "small": ((0.25, 0.45), (0.25, -0.45), (0.70, 0.45), (0.70, -0.45)),
        "large": ((0.55, 0.00),),
    },
    SemanticCourseStage.S4: {
        "small": ((0.20, 0.50), (0.20, -0.50), (0.45, 0.28), (0.45, -0.28), (0.70, 0.50), (0.70, -0.50)),
        "large": ((0.55, 0.00),),
    },
}


def stage_row_bands(num_rows: int) -> dict[SemanticCourseStage, tuple[int, int]]:
    """Return inclusive-exclusive row bands for S1..S4 using quarter splits."""
    if num_rows <= 0:
        raise ValueError(f"num_rows must be positive, got {num_rows}.")
    b1 = math.ceil(num_rows * 1 / 4)
    b2 = math.ceil(num_rows * 2 / 4)
    b3 = math.ceil(num_rows * 3 / 4)
    return {
        SemanticCourseStage.S1: (0, b1),
        SemanticCourseStage.S2: (b1, b2),
        SemanticCourseStage.S3: (b2, b3),
        SemanticCourseStage.S4: (b3, num_rows),
    }


def stage_for_row(row: int, num_rows: int) -> SemanticCourseStage:
    """Map a terrain row to its semantic-course stage."""
    if row < 0 or row >= num_rows:
        raise ValueError(f"row must be in [0, {num_rows}), got {row}.")
    for stage, (start, stop) in stage_row_bands(num_rows).items():
        if start <= row < stop:
            return stage
    raise RuntimeError(f"Failed to assign stage for row={row}, num_rows={num_rows}.")


def representative_rows(num_rows: int) -> dict[SemanticCourseStage, int]:
    """Choose one stable representative row per semantic stage."""
    rows: dict[SemanticCourseStage, int] = {}
    for stage, (start, stop) in stage_row_bands(num_rows).items():
        if stop <= start:
            raise ValueError(
                f"Cannot choose representative row for {stage.value}: empty band [{start}, {stop}) with num_rows={num_rows}."
            )
        rows[stage] = (start + stop - 1) // 2
    return rows


def stage_layout(stage: SemanticCourseStage) -> dict[str, tuple[tuple[float, float], ...]]:
    return _STAGE_LAYOUTS[stage]


def course_anchor_counts(stage: SemanticCourseStage) -> dict[str, int]:
    layout = stage_layout(stage)
    return {semantic_class: len(layout[semantic_class]) for semantic_class in ("small", "large")}


def build_course_anchors(terrain_origins: Any) -> list[CourseAnchor]:
    """Build deterministic per-tile obstacle anchors before terrain grounding."""
    num_rows = len(terrain_origins)
    num_cols = len(terrain_origins[0]) if num_rows > 0 else 0
    anchors: list[CourseAnchor] = []
    for row in range(num_rows):
        stage = stage_for_row(row, num_rows)
        layout = stage_layout(stage)
        for col in range(num_cols):
            origin = terrain_origins[row][col]
            origin_x = float(origin[0])
            origin_y = float(origin[1])
            for semantic_class in ("small", "large"):
                size = SMALL_OBSTACLE_SIZE if semantic_class == "small" else LARGE_OBSTACLE_SIZE
                root = SEMANTIC_COURSE_SMALL_ROOT if semantic_class == "small" else SEMANTIC_COURSE_LARGE_ROOT
                for slot_index, local_xy in enumerate(layout[semantic_class]):
                    local_x, local_y = local_xy
                    anchors.append(
                        CourseAnchor(
                            row=row,
                            col=col,
                            stage=stage,
                            semantic_class=semantic_class,
                            slot_index=slot_index,
                            local_xy=local_xy,
                            size=size,
                            world_xy=(origin_x + local_x, origin_y + local_y),
                            prim_path=f"{root}/row_{row:02d}/col_{col:02d}/slot_{slot_index:02d}",
                        )
                    )
    return anchors


def ground_course_anchors(
    anchors: list[CourseAnchor],
    *,
    terrain_height_at_xy,
) -> list[GroundedCourseObstacle]:
    """Place cuboid centers on terrain height plus half obstacle height."""
    obstacles: list[GroundedCourseObstacle] = []
    for anchor in anchors:
        world_x, world_y = anchor.world_xy
        terrain_z = float(terrain_height_at_xy(world_x, world_y))
        center_z = terrain_z + 0.5 * float(anchor.size[2])
        obstacles.append(
            GroundedCourseObstacle(
                row=anchor.row,
                col=anchor.col,
                stage=anchor.stage,
                semantic_class=anchor.semantic_class,
                slot_index=anchor.slot_index,
                local_xy=anchor.local_xy,
                size=anchor.size,
                world_center=(world_x, world_y, center_z),
                prim_path=anchor.prim_path,
            )
        )
    return obstacles


def set_scene_env_to_representative_stage(scene, *, env_id: int, stage: SemanticCourseStage | str) -> int:
    """Force one environment onto the representative row for a given stage."""
    stage = SemanticCourseStage(stage)
    terrain = scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        raise RuntimeError("Representative-row override requires generated terrain origins.")
    rep_rows = representative_rows(len(terrain.terrain_origins))
    row = rep_rows[stage]
    if not hasattr(terrain, "terrain_types") or not hasattr(terrain, "terrain_levels"):
        raise RuntimeError("Terrain importer does not expose curriculum row/type buffers.")
    terrain_col_value = terrain.terrain_types[env_id]
    terrain_col = int(terrain_col_value.item()) if hasattr(terrain_col_value, "item") else int(terrain_col_value)
    terrain.terrain_levels[env_id] = row
    terrain.env_origins[env_id] = terrain.terrain_origins[row, terrain_col]
    return row


def spawn_semantic_course_prestartup(
    env,
    _env_ids,
    *,
    default_stage: str = DEFAULT_VIEWER_REPRESENTATIVE_STAGE.value,
) -> None:
    """Prestartup event: create semantic-course geometry before sensor initialization."""
    scene = env.scene
    terrain = scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        raise RuntimeError("Semantic course generation requires terrain origins from a generated terrain.")

    ensure_semantic_course_roots()
    clear_semantic_course_children()

    anchors = build_course_anchors(terrain.terrain_origins)
    obstacles = _ground_with_runtime_terrain_sampler(anchors, device=getattr(env, "device", "cpu"))
    for obstacle in obstacles:
        _spawn_grounded_cuboid(obstacle)

    if scene.num_envs > 0:
        set_scene_env_to_representative_stage(scene, env_id=0, stage=default_stage)


def ensure_semantic_course_roots() -> None:
    """Create the stable semantic-course container Xforms."""
    import isaacsim.core.utils.prims as prim_utils

    if not prim_utils.is_prim_path_valid(SEMANTIC_COURSE_ROOT):
        prim_utils.create_prim(SEMANTIC_COURSE_ROOT, "Xform")
    for prim_path in SEMANTIC_COURSE_ROOTS:
        if not prim_utils.is_prim_path_valid(prim_path):
            prim_utils.create_prim(prim_path, "Xform")


def clear_semantic_course_children() -> None:
    """Delete generated descendants while preserving the stable container roots."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for root_path in SEMANTIC_COURSE_ROOTS:
        prim = stage.GetPrimAtPath(root_path)
        if not prim.IsValid():
            continue
        for child in list(prim.GetChildren()):
            stage.RemovePrim(child.GetPath().pathString)


def _spawn_grounded_cuboid(obstacle: GroundedCourseObstacle) -> None:
    import isaacsim.core.utils.prims as prim_utils
    import isaaclab.sim as sim_utils

    row_path = obstacle.prim_path.rsplit("/", 2)[0]
    col_path = obstacle.prim_path.rsplit("/", 1)[0]
    if not prim_utils.is_prim_path_valid(row_path):
        prim_utils.create_prim(row_path, "Xform")
    if not prim_utils.is_prim_path_valid(col_path):
        prim_utils.create_prim(col_path, "Xform")

    cuboid_cfg = sim_utils.CuboidCfg(
        size=obstacle.size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    cuboid_cfg.func(obstacle.prim_path, cuboid_cfg, translation=obstacle.world_center)


def _ground_with_runtime_terrain_sampler(anchors: list[CourseAnchor], *, device: str) -> list[GroundedCourseObstacle]:
    if not anchors:
        return []
    xy_points = [anchor.world_xy for anchor in anchors]
    heights = _sample_terrain_heights_world(xy_points, device=device)
    return [
        GroundedCourseObstacle(
            row=anchor.row,
            col=anchor.col,
            stage=anchor.stage,
            semantic_class=anchor.semantic_class,
            slot_index=anchor.slot_index,
            local_xy=anchor.local_xy,
            size=anchor.size,
            world_center=(anchor.world_xy[0], anchor.world_xy[1], heights[index] + 0.5 * float(anchor.size[2])),
            prim_path=anchor.prim_path,
        )
        for index, anchor in enumerate(anchors)
    ]


def _sample_terrain_heights_world(xy_points: list[tuple[float, float]], *, device: str) -> list[float]:
    import numpy as np
    import torch
    from pxr import UsdGeom

    import omni
    import isaaclab.sim as sim_utils
    from isaaclab.terrains.trimesh.utils import make_plane
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

    def world_transform_T(usd_geom) -> np.ndarray:
        return np.array(omni.usd.get_world_transform_matrix(usd_geom)).T

    def apply_world_transform(points_local: np.ndarray, transform_T: np.ndarray) -> np.ndarray:
        r = transform_T[:3, :3].astype(np.float64)
        t = transform_T[:3, 3].astype(np.float64)
        return (points_local @ r.T + t).astype(np.float32)

    def mesh_to_world_trimesh(geom_prim):
        mesh = UsdGeom.Mesh(geom_prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        transform_T = world_transform_T(mesh)
        points = points @ transform_T[:3, :3].T + transform_T[:3, 3]
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
        return points.astype(np.float32), faces.astype(np.int32)

    def plane_to_world_trimesh(geom_prim):
        mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
        transform_T = world_transform_T(UsdGeom.Plane(geom_prim))
        return apply_world_transform(mesh.vertices.astype(np.float64), transform_T), mesh.faces.astype(np.int32)

    def collect_geometry(root_path: str):
        root = sim_utils.find_first_matching_prim(root_path)
        if root is None or not root.IsValid():
            raise RuntimeError(f"Missing terrain root for semantic course grounding: {root_path!r}")
        geometries: list[tuple[np.ndarray, np.ndarray]] = []
        stack = [root]
        while stack:
            prim = stack.pop()
            prim_type = prim.GetTypeName()
            if prim_type == "Mesh":
                geometries.append(mesh_to_world_trimesh(prim))
            elif prim_type == "Plane":
                geometries.append(plane_to_world_trimesh(prim))
            else:
                stack.extend(reversed(list(prim.GetChildren())))
        if not geometries:
            raise RuntimeError(f"No supported terrain geometry found under {root_path!r}")
        return geometries

    geometries = collect_geometry("/World/ground")
    vert_blocks: list[np.ndarray] = []
    face_blocks: list[np.ndarray] = []
    vertex_offset = 0
    for points, faces in geometries:
        vert_blocks.append(points)
        face_blocks.append(faces + vertex_offset)
        vertex_offset += points.shape[0]
    points = np.concatenate(vert_blocks, axis=0)
    faces = np.concatenate(face_blocks, axis=0)
    wp_mesh = convert_to_warp_mesh(points, faces, device)

    bbox_max_z = float(points[:, 2].max()) if len(points) > 0 else 0.0
    ray_start_z = bbox_max_z + 5.0
    starts = torch.tensor([[x, y, ray_start_z] for x, y in xy_points], dtype=torch.float32, device=device)
    dirs = torch.zeros_like(starts)
    dirs[:, 2] = -1.0
    ray_hits = raycast_mesh(starts, dirs, wp_mesh)[0]
    if torch.any(~torch.isfinite(ray_hits[:, 2])):
        raise RuntimeError("Terrain grounding raycast missed at least one semantic-course anchor.")
    return [float(z) for z in ray_hits[:, 2].tolist()]
