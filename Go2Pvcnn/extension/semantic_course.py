"""Static semantic-course helpers for the trajectory viewer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import statistics
from typing import Any, NamedTuple, Literal


SEMANTIC_COURSE_ROOT = "/World/semantic_course"
SEMANTIC_COURSE_SMALL_ROOT = f"{SEMANTIC_COURSE_ROOT}/small"
SEMANTIC_COURSE_LARGE_ROOT = f"{SEMANTIC_COURSE_ROOT}/large"
SEMANTIC_COURSE_ROOTS = (SEMANTIC_COURSE_SMALL_ROOT, SEMANTIC_COURSE_LARGE_ROOT)

ShapeKind = Literal["sphere", "cuboid", "cylinder", "capsule", "cone"]

SHARED_NATIVE_SHAPE_POOL: tuple[ShapeKind, ...] = ("sphere", "cuboid", "cylinder", "capsule", "cone")
SHAPE_AXIS_Z = "Z"
SHAPE_EPSILON = 1.0e-6

SMALL_OBSTACLE_DIAMETER = 0.12
SMALL_OBSTACLE_HEIGHT = 0.16
LARGE_OBSTACLE_DIAMETER = 0.45
LARGE_OBSTACLE_HEIGHT = 0.55

# Kept for compatibility with existing scale-based tests and callers.
SMALL_OBSTACLE_SIZE = (SMALL_OBSTACLE_DIAMETER, SMALL_OBSTACLE_DIAMETER, SMALL_OBSTACLE_HEIGHT)
LARGE_OBSTACLE_SIZE = (LARGE_OBSTACLE_DIAMETER, LARGE_OBSTACLE_DIAMETER, LARGE_OBSTACLE_HEIGHT)

DEFAULT_SEMANTIC_COURSE_SEED = 20260430
DEFAULT_SEMANTIC_COURSE_TILE_SIZE = (8.0, 8.0)
DEFAULT_TILE_MARGIN_M = 0.50
DEFAULT_CENTER_SAFETY_HALF_EXTENT_M = 0.85
DEFAULT_MIN_SPACING_CLEARANCE_M = 0.15
DEFAULT_MAX_LAYOUT_ATTEMPTS = 64
DEFAULT_GROUNDING_HEIGHT_QUANTILE = 1.0
DEFAULT_GROUNDING_EMBED_DEPTH_M = 0.015


class SemanticCourseStage(str, Enum):
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"


DEFAULT_VIEWER_REPRESENTATIVE_STAGE = SemanticCourseStage.S4


@dataclass(frozen=True)
class SemanticCourseLayoutCfg:
    tile_margin_m: float = DEFAULT_TILE_MARGIN_M
    center_safety_half_extent_m: float = DEFAULT_CENTER_SAFETY_HALF_EXTENT_M
    min_spacing_clearance_m: float = DEFAULT_MIN_SPACING_CLEARANCE_M
    max_layout_attempts: int = DEFAULT_MAX_LAYOUT_ATTEMPTS


@dataclass(frozen=True)
class SemanticCourseGroundingCfg:
    height_quantile: float = DEFAULT_GROUNDING_HEIGHT_QUANTILE
    embed_depth_m: float = DEFAULT_GROUNDING_EMBED_DEPTH_M


DEFAULT_SEMANTIC_COURSE_LAYOUT_CFG = SemanticCourseLayoutCfg()
DEFAULT_SEMANTIC_COURSE_GROUNDING_CFG = SemanticCourseGroundingCfg()


@dataclass(frozen=True)
class CourseAnchor:
    row: int
    col: int
    stage: SemanticCourseStage
    semantic_class: str
    slot_index: int
    shape_kind: ShapeKind
    shape_params: dict[str, float | str | tuple[float, float, float]]
    target_diameter: float
    target_height: float
    ground_offset: float
    local_xy: tuple[float, float]
    world_xy: tuple[float, float]
    prim_path: str
    layout_fallback_used: bool = False


@dataclass(frozen=True)
class GroundedCourseObstacle:
    row: int
    col: int
    stage: SemanticCourseStage
    semantic_class: str
    slot_index: int
    shape_kind: ShapeKind
    shape_params: dict[str, float | str | tuple[float, float, float]]
    target_diameter: float
    target_height: float
    ground_offset: float
    local_xy: tuple[float, float]
    world_center: tuple[float, float, float]
    prim_path: str


_STAGE_LAYOUTS: dict[SemanticCourseStage, dict[str, int]] = {
    SemanticCourseStage.S1: {"small": 0, "large": 0},
    SemanticCourseStage.S2: {"small": 4, "large": 0},
    SemanticCourseStage.S3: {"small": 4, "large": 1},
    SemanticCourseStage.S4: {"small": 6, "large": 1},
}


class _LayoutSlot(NamedTuple):
    semantic_class: str
    slot_index: int
    local_xy: tuple[float, float]
    layout_fallback_used: bool


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


def stage_layout(stage: SemanticCourseStage) -> dict[str, int]:
    return _STAGE_LAYOUTS[stage]


def course_anchor_counts(stage: SemanticCourseStage) -> dict[str, int]:
    layout = stage_layout(stage)
    return {semantic_class: int(layout[semantic_class]) for semantic_class in ("small", "large")}


def semantic_scale_profile(semantic_class: str) -> tuple[float, float]:
    if semantic_class == "small":
        return SMALL_OBSTACLE_DIAMETER, SMALL_OBSTACLE_HEIGHT
    if semantic_class == "large":
        return LARGE_OBSTACLE_DIAMETER, LARGE_OBSTACLE_HEIGHT
    raise ValueError(f"Unsupported semantic class {semantic_class!r}.")


def deterministic_shape_key(
    *,
    stage: SemanticCourseStage | str,
    row: int,
    col: int,
    slot_index: int,
    semantic_class: str,
) -> int:
    stage = SemanticCourseStage(stage)
    stage_index = int(stage.value[1:])
    semantic_index = {"small": 0, "large": 1}.get(semantic_class)
    if semantic_index is None:
        raise ValueError(f"Unsupported semantic class {semantic_class!r}.")
    return (
        stage_index * 1_000_003
        + row * 10_007
        + col * 1_009
        + slot_index * 97
        + semantic_index * 17
    )


def select_shape_kind(
    *,
    stage: SemanticCourseStage | str,
    row: int,
    col: int,
    slot_index: int,
    semantic_class: str,
    shape_pool: tuple[ShapeKind, ...] = SHARED_NATIVE_SHAPE_POOL,
) -> ShapeKind:
    if not shape_pool:
        raise ValueError("shape_pool must be non-empty.")
    key = deterministic_shape_key(
        stage=stage,
        row=row,
        col=col,
        slot_index=slot_index,
        semantic_class=semantic_class,
    )
    return shape_pool[key % len(shape_pool)]


def shape_params_for_profile(
    shape_kind: ShapeKind,
    *,
    target_diameter: float,
    target_height: float,
) -> dict[str, float | str | tuple[float, float, float]]:
    radius = 0.5 * float(target_diameter)
    height = float(target_height)
    if shape_kind == "sphere":
        return {"radius": radius}
    if shape_kind == "cuboid":
        return {"size": (float(target_diameter), float(target_diameter), height)}
    if shape_kind == "cylinder":
        return {"radius": radius, "height": height, "axis": SHAPE_AXIS_Z}
    if shape_kind == "capsule":
        return {"radius": radius, "height": max(height - float(target_diameter), SHAPE_EPSILON), "axis": SHAPE_AXIS_Z}
    if shape_kind == "cone":
        return {"radius": radius, "height": height, "axis": SHAPE_AXIS_Z}
    raise ValueError(f"Unsupported shape kind {shape_kind!r}.")


def bottom_to_center_offset(
    shape_kind: ShapeKind,
    shape_params: dict[str, float | str | tuple[float, float, float]],
) -> float:
    if shape_kind == "cuboid":
        size = shape_params["size"]
        return 0.5 * float(size[2])  # type: ignore[index]
    if shape_kind == "sphere":
        return float(shape_params["radius"])
    if shape_kind in ("cylinder", "cone"):
        return 0.5 * float(shape_params["height"])
    if shape_kind == "capsule":
        return float(shape_params["radius"]) + 0.5 * float(shape_params["height"])
    raise ValueError(f"Unsupported shape kind {shape_kind!r}.")


def footprint_sample_offsets(
    shape_kind: ShapeKind,
    shape_params: dict[str, float | str | tuple[float, float, float]],
) -> tuple[tuple[float, float], ...]:
    """Return center plus eight support offsets for shape-aware terrain grounding."""
    if shape_kind == "cuboid":
        size = shape_params["size"]
        half_x = 0.5 * float(size[0])  # type: ignore[index]
        half_y = 0.5 * float(size[1])  # type: ignore[index]
        diagonal_x = half_x
        diagonal_y = half_y
    elif shape_kind in ("sphere", "cylinder", "capsule", "cone"):
        radius = float(shape_params["radius"])
        half_x = radius
        half_y = radius
        diagonal_x = radius / math.sqrt(2.0)
        diagonal_y = radius / math.sqrt(2.0)
    else:
        raise ValueError(f"Unsupported shape kind {shape_kind!r}.")
    if half_x <= 0.0 or half_y <= 0.0:
        raise ValueError(f"Footprint dimensions must be positive for {shape_kind!r}, got {shape_params!r}.")
    return (
        (0.0, 0.0),
        (half_x, 0.0),
        (-half_x, 0.0),
        (0.0, half_y),
        (0.0, -half_y),
        (diagonal_x, diagonal_y),
        (diagonal_x, -diagonal_y),
        (-diagonal_x, diagonal_y),
        (-diagonal_x, -diagonal_y),
    )


def _positive_pair(values: Any) -> tuple[float, float] | None:
    try:
        x = float(values[0])
        y = float(values[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if math.isfinite(x) and math.isfinite(y) and x > 0.0 and y > 0.0:
        return x, y
    return None


def _validated_tile_size(values: Any) -> tuple[float, float]:
    tile_size = _positive_pair(values)
    if tile_size is None:
        raise ValueError(f"tile_size must contain two positive finite values, got {values!r}.")
    return tile_size


def _origin_component(origin: Any, index: int) -> float:
    value = origin[index]
    return float(value.item()) if hasattr(value, "item") else float(value)


def _infer_axis_spacing(terrain_origins: Any, *, axis: int) -> float | None:
    num_rows = len(terrain_origins)
    num_cols = len(terrain_origins[0]) if num_rows > 0 else 0
    diffs: list[float] = []
    if axis == 0:
        for col in range(num_cols):
            for row in range(num_rows - 1):
                try:
                    diff = abs(
                        _origin_component(terrain_origins[row + 1][col], 0)
                        - _origin_component(terrain_origins[row][col], 0)
                    )
                except (TypeError, ValueError, IndexError, KeyError):
                    continue
                if math.isfinite(diff) and diff > 0.0:
                    diffs.append(diff)
    elif axis == 1:
        for row in range(num_rows):
            for col in range(num_cols - 1):
                try:
                    diff = abs(
                        _origin_component(terrain_origins[row][col + 1], 1)
                        - _origin_component(terrain_origins[row][col], 1)
                    )
                except (TypeError, ValueError, IndexError, KeyError):
                    continue
                if math.isfinite(diff) and diff > 0.0:
                    diffs.append(diff)
    else:
        raise ValueError(f"Unsupported axis {axis}.")
    return float(statistics.median(diffs)) if diffs else None


def resolve_tile_size(
    terrain_origins,
    *,
    terrain_generator=None,
    fallback_tile_size: tuple[float, float] = DEFAULT_SEMANTIC_COURSE_TILE_SIZE,
) -> tuple[float, float]:
    """Resolve sub-terrain tile size from generator config, origins, then fallback."""
    fallback = _validated_tile_size(fallback_tile_size)
    if terrain_generator is not None and hasattr(terrain_generator, "size"):
        generator_size = _positive_pair(getattr(terrain_generator, "size"))
        if generator_size is not None:
            return generator_size
    inferred_x = _infer_axis_spacing(terrain_origins, axis=0)
    inferred_y = _infer_axis_spacing(terrain_origins, axis=1)
    return (
        inferred_x if inferred_x is not None else fallback[0],
        inferred_y if inferred_y is not None else fallback[1],
    )


def _stable_unit_interval(*parts: object) -> float:
    digest = hashlib.blake2b("|".join(str(part) for part in parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)


def _candidate_local_xy(
    *,
    seed: int,
    stage: SemanticCourseStage,
    row: int,
    col: int,
    slot_index: int,
    semantic_class: str,
    attempt_index: int,
    tile_size: tuple[float, float],
    layout_cfg: SemanticCourseLayoutCfg,
    radius: float,
) -> tuple[float, float]:
    min_x = -tile_size[0] / 2.0 + layout_cfg.tile_margin_m + radius
    max_x = tile_size[0] / 2.0 - layout_cfg.tile_margin_m - radius
    min_y = -tile_size[1] / 2.0 + layout_cfg.tile_margin_m + radius
    max_y = tile_size[1] / 2.0 - layout_cfg.tile_margin_m - radius
    if min_x > max_x or min_y > max_y:
        raise ValueError(
            f"Tile size {tile_size!r} is too small for margin {layout_cfg.tile_margin_m} and radius {radius}."
        )
    x_unit = _stable_unit_interval(seed, stage.value, row, col, slot_index, semantic_class, attempt_index, "x")
    y_unit = _stable_unit_interval(seed, stage.value, row, col, slot_index, semantic_class, attempt_index, "y")
    return (min_x + (max_x - min_x) * x_unit, min_y + (max_y - min_y) * y_unit)


def _outside_center_safety(local_xy: tuple[float, float], layout_cfg: SemanticCourseLayoutCfg) -> bool:
    return (
        abs(local_xy[0]) > layout_cfg.center_safety_half_extent_m
        or abs(local_xy[1]) > layout_cfg.center_safety_half_extent_m
    )


def _has_spacing(
    candidate: tuple[float, float],
    *,
    radius: float,
    placed: list[tuple[tuple[float, float], float]],
    layout_cfg: SemanticCourseLayoutCfg,
) -> bool:
    for local_xy, placed_radius in placed:
        required = radius + placed_radius + layout_cfg.min_spacing_clearance_m
        if math.hypot(candidate[0] - local_xy[0], candidate[1] - local_xy[1]) < required:
            return False
    return True


def _fallback_candidates(
    *,
    tile_size: tuple[float, float],
    layout_cfg: SemanticCourseLayoutCfg,
    radius: float,
) -> list[tuple[float, float]]:
    min_x = -tile_size[0] / 2.0 + layout_cfg.tile_margin_m + radius
    max_x = tile_size[0] / 2.0 - layout_cfg.tile_margin_m - radius
    min_y = -tile_size[1] / 2.0 + layout_cfg.tile_margin_m + radius
    max_y = tile_size[1] / 2.0 - layout_cfg.tile_margin_m - radius
    xs = (min_x, max_x, 0.5 * (min_x + max_x))
    ys = (min_y, max_y, 0.5 * (min_y + max_y))
    candidates = [(x, y) for x in xs for y in ys if _outside_center_safety((x, y), layout_cfg)]
    if not candidates:
        raise ValueError(f"Tile size {tile_size!r} leaves no fallback point outside the center safety box.")
    return candidates


def _stage_slots(
    *,
    stage: SemanticCourseStage,
    row: int,
    col: int,
    tile_size: tuple[float, float],
    semantic_course_seed: int,
    layout_cfg: SemanticCourseLayoutCfg,
) -> list[_LayoutSlot]:
    stage_counts = stage_layout(stage)
    ordered_classes = ("large", "small")
    placed: list[tuple[tuple[float, float], float]] = []
    slots: list[_LayoutSlot] = []
    for semantic_class in ordered_classes:
        target_diameter, _target_height = semantic_scale_profile(semantic_class)
        radius = target_diameter / 2.0
        for slot_index in range(stage_counts[semantic_class]):
            selected_xy: tuple[float, float] | None = None
            for attempt_index in range(layout_cfg.max_layout_attempts):
                candidate = _candidate_local_xy(
                    seed=semantic_course_seed,
                    stage=stage,
                    row=row,
                    col=col,
                    slot_index=slot_index,
                    semantic_class=semantic_class,
                    attempt_index=attempt_index,
                    tile_size=tile_size,
                    layout_cfg=layout_cfg,
                    radius=radius,
                )
                if _outside_center_safety(candidate, layout_cfg) and _has_spacing(
                    candidate,
                    radius=radius,
                    placed=placed,
                    layout_cfg=layout_cfg,
                ):
                    selected_xy = candidate
                    break
            fallback_used = selected_xy is None
            if fallback_used:
                candidates = _fallback_candidates(tile_size=tile_size, layout_cfg=layout_cfg, radius=radius)
                selected_xy = candidates[(len(placed) + slot_index) % len(candidates)]
            placed.append((selected_xy, radius))
            slots.append(_LayoutSlot(semantic_class, slot_index, selected_xy, fallback_used))
    return sorted(slots, key=lambda slot: (0 if slot.semantic_class == "small" else 1, slot.slot_index))


def build_course_anchors(
    terrain_origins: Any,
    *,
    terrain_generator=None,
    tile_size: tuple[float, float] | None = None,
    semantic_course_seed: int = DEFAULT_SEMANTIC_COURSE_SEED,
    layout_cfg: SemanticCourseLayoutCfg = DEFAULT_SEMANTIC_COURSE_LAYOUT_CFG,
) -> list[CourseAnchor]:
    """Build deterministic per-tile obstacle anchors before terrain grounding."""
    num_rows = len(terrain_origins)
    num_cols = len(terrain_origins[0]) if num_rows > 0 else 0
    resolved_tile_size = _validated_tile_size(tile_size) if tile_size is not None else resolve_tile_size(
        terrain_origins,
        terrain_generator=terrain_generator,
    )
    anchors: list[CourseAnchor] = []
    for row in range(num_rows):
        stage = stage_for_row(row, num_rows)
        for col in range(num_cols):
            origin = terrain_origins[row][col]
            origin_x = _origin_component(origin, 0)
            origin_y = _origin_component(origin, 1)
            for slot in _stage_slots(
                stage=stage,
                row=row,
                col=col,
                tile_size=resolved_tile_size,
                semantic_course_seed=semantic_course_seed,
                layout_cfg=layout_cfg,
            ):
                semantic_class = slot.semantic_class
                target_diameter, target_height = semantic_scale_profile(semantic_class)
                root = SEMANTIC_COURSE_SMALL_ROOT if semantic_class == "small" else SEMANTIC_COURSE_LARGE_ROOT
                local_x, local_y = slot.local_xy
                shape_kind = select_shape_kind(
                    stage=stage,
                    row=row,
                    col=col,
                    slot_index=slot.slot_index,
                    semantic_class=semantic_class,
                )
                shape_params = shape_params_for_profile(
                    shape_kind,
                    target_diameter=target_diameter,
                    target_height=target_height,
                )
                anchors.append(
                    CourseAnchor(
                        row=row,
                        col=col,
                        stage=stage,
                        semantic_class=semantic_class,
                        slot_index=slot.slot_index,
                        shape_kind=shape_kind,
                        shape_params=shape_params,
                        target_diameter=target_diameter,
                        target_height=target_height,
                        ground_offset=bottom_to_center_offset(shape_kind, shape_params),
                        local_xy=slot.local_xy,
                        world_xy=(origin_x + local_x, origin_y + local_y),
                        prim_path=f"{root}/row_{row:02d}/col_{col:02d}/slot_{slot.slot_index:02d}",
                        layout_fallback_used=slot.layout_fallback_used,
                    )
                )
    return anchors


def ground_course_anchors(
    anchors: list[CourseAnchor],
    *,
    terrain_height_at_xy,
    grounding_cfg: SemanticCourseGroundingCfg = DEFAULT_SEMANTIC_COURSE_GROUNDING_CFG,
) -> list[GroundedCourseObstacle]:
    """Place obstacle centers from robust footprint terrain height plus shape offset."""
    obstacles: list[GroundedCourseObstacle] = []
    for anchor in anchors:
        world_x, world_y = anchor.world_xy
        sample_points = _footprint_world_xy(anchor)
        terrain_heights = [float(terrain_height_at_xy(x, y)) for x, y in sample_points]
        robust_ground_z = _robust_ground_height(
            terrain_heights,
            grounding_cfg=grounding_cfg,
            context=f"{anchor.prim_path} footprint",
            error_type=ValueError,
        )
        center_z = robust_ground_z - grounding_cfg.embed_depth_m + anchor.ground_offset
        obstacles.append(
            GroundedCourseObstacle(
                row=anchor.row,
                col=anchor.col,
                stage=anchor.stage,
                semantic_class=anchor.semantic_class,
                slot_index=anchor.slot_index,
                shape_kind=anchor.shape_kind,
                shape_params=anchor.shape_params,
                target_diameter=anchor.target_diameter,
                target_height=anchor.target_height,
                ground_offset=anchor.ground_offset,
                local_xy=anchor.local_xy,
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
    semantic_course_seed: int = DEFAULT_SEMANTIC_COURSE_SEED,
    tile_size: tuple[float, float] | None = None,
    layout_cfg: SemanticCourseLayoutCfg = DEFAULT_SEMANTIC_COURSE_LAYOUT_CFG,
    grounding_cfg: SemanticCourseGroundingCfg = DEFAULT_SEMANTIC_COURSE_GROUNDING_CFG,
) -> None:
    """Prestartup event: create semantic-course geometry before sensor initialization."""
    scene = env.scene
    terrain = scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        raise RuntimeError("Semantic course generation requires terrain origins from a generated terrain.")

    ensure_semantic_course_roots()
    clear_semantic_course_children()

    terrain_cfg = getattr(terrain, "cfg", None)
    terrain_generator = getattr(terrain_cfg, "terrain_generator", None) if terrain_cfg is not None else None
    anchors = build_course_anchors(
        terrain.terrain_origins,
        terrain_generator=terrain_generator,
        tile_size=tile_size,
        semantic_course_seed=semantic_course_seed,
        layout_cfg=layout_cfg,
    )
    obstacles = _ground_with_runtime_terrain_sampler(
        anchors,
        device=getattr(env, "device", "cpu"),
        grounding_cfg=grounding_cfg,
    )
    for obstacle in obstacles:
        _spawn_grounded_shape(obstacle)

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


def _shape_spawn_cfg(
    obstacle: GroundedCourseObstacle,
    *,
    sim_utils,
):
    shared_kwargs = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    if obstacle.shape_kind == "sphere":
        return sim_utils.SphereCfg(radius=float(obstacle.shape_params["radius"]), **shared_kwargs)
    if obstacle.shape_kind == "cuboid":
        return sim_utils.CuboidCfg(size=obstacle.shape_params["size"], **shared_kwargs)
    if obstacle.shape_kind == "cylinder":
        return sim_utils.CylinderCfg(
            radius=float(obstacle.shape_params["radius"]),
            height=float(obstacle.shape_params["height"]),
            axis=str(obstacle.shape_params["axis"]),
            **shared_kwargs,
        )
    if obstacle.shape_kind == "capsule":
        return sim_utils.CapsuleCfg(
            radius=float(obstacle.shape_params["radius"]),
            height=float(obstacle.shape_params["height"]),
            axis=str(obstacle.shape_params["axis"]),
            **shared_kwargs,
        )
    if obstacle.shape_kind == "cone":
        return sim_utils.ConeCfg(
            radius=float(obstacle.shape_params["radius"]),
            height=float(obstacle.shape_params["height"]),
            axis=str(obstacle.shape_params["axis"]),
            **shared_kwargs,
        )
    raise ValueError(f"Unsupported shape kind {obstacle.shape_kind!r}.")


def _spawn_grounded_shape(obstacle: GroundedCourseObstacle) -> None:
    import isaacsim.core.utils.prims as prim_utils
    import isaaclab.sim as sim_utils

    row_path = obstacle.prim_path.rsplit("/", 2)[0]
    col_path = obstacle.prim_path.rsplit("/", 1)[0]
    if not prim_utils.is_prim_path_valid(row_path):
        prim_utils.create_prim(row_path, "Xform")
    if not prim_utils.is_prim_path_valid(col_path):
        prim_utils.create_prim(col_path, "Xform")

    shape_cfg = _shape_spawn_cfg(obstacle, sim_utils=sim_utils)
    shape_cfg.func(obstacle.prim_path, shape_cfg, translation=obstacle.world_center)


def _footprint_world_xy(anchor: CourseAnchor) -> tuple[tuple[float, float], ...]:
    world_x, world_y = anchor.world_xy
    return tuple(
        (world_x + offset_x, world_y + offset_y)
        for offset_x, offset_y in footprint_sample_offsets(anchor.shape_kind, anchor.shape_params)
    )


def _robust_ground_height(
    heights: list[float],
    *,
    grounding_cfg: SemanticCourseGroundingCfg,
    context: str,
    error_type: type[Exception],
) -> float:
    finite_heights = [float(height) for height in heights if math.isfinite(float(height))]
    if len(finite_heights) != len(heights):
        raise error_type(f"Semantic-course grounding found non-finite terrain height for {context}.")
    if not finite_heights:
        raise error_type(f"Semantic-course grounding received no footprint heights for {context}.")
    quantile = float(grounding_cfg.height_quantile)
    if not math.isfinite(quantile) or quantile < 0.0 or quantile > 1.0:
        raise error_type(f"grounding height_quantile must be in [0, 1], got {grounding_cfg.height_quantile!r}.")
    embed_depth = float(grounding_cfg.embed_depth_m)
    if not math.isfinite(embed_depth) or embed_depth < 0.0:
        raise error_type(f"grounding embed_depth_m must be finite and non-negative, got {grounding_cfg.embed_depth_m!r}.")
    if quantile == 1.0 or len(finite_heights) == 1:
        return max(finite_heights)
    ordered = sorted(finite_heights)
    position = quantile * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    alpha = position - lower_index
    return ordered[lower_index] * (1.0 - alpha) + ordered[upper_index] * alpha


def _ground_with_runtime_terrain_sampler(
    anchors: list[CourseAnchor],
    *,
    device: str,
    grounding_cfg: SemanticCourseGroundingCfg = DEFAULT_SEMANTIC_COURSE_GROUNDING_CFG,
) -> list[GroundedCourseObstacle]:
    if not anchors:
        return []
    anchor_sample_counts: list[int] = []
    xy_points: list[tuple[float, float]] = []
    for anchor in anchors:
        sample_points = _footprint_world_xy(anchor)
        anchor_sample_counts.append(len(sample_points))
        xy_points.extend(sample_points)
    heights = _sample_terrain_heights_world(xy_points, device=device)
    obstacles: list[GroundedCourseObstacle] = []
    height_index = 0
    for anchor, sample_count in zip(anchors, anchor_sample_counts, strict=True):
        anchor_heights = heights[height_index : height_index + sample_count]
        height_index += sample_count
        robust_ground_z = _robust_ground_height(
            anchor_heights,
            grounding_cfg=grounding_cfg,
            context=f"{anchor.prim_path} footprint",
            error_type=RuntimeError,
        )
        obstacles.append(
            GroundedCourseObstacle(
                row=anchor.row,
                col=anchor.col,
                stage=anchor.stage,
                semantic_class=anchor.semantic_class,
                slot_index=anchor.slot_index,
                shape_kind=anchor.shape_kind,
                shape_params=anchor.shape_params,
                target_diameter=anchor.target_diameter,
                target_height=anchor.target_height,
                ground_offset=anchor.ground_offset,
                local_xy=anchor.local_xy,
                world_center=(
                    anchor.world_xy[0],
                    anchor.world_xy[1],
                    robust_ground_z - grounding_cfg.embed_depth_m + anchor.ground_offset,
                ),
                prim_path=anchor.prim_path,
            )
        )
    return obstacles


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
