"""Fixed condition manifests shared by all model checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Iterable

from .protocols import MIXED_VX, MIXED_VY, MIXED_YAW, RUNWAY_VX


SMALL_DENSITIES = ((2.5, 40), (5.0, 80), (7.5, 120), (10.0, 160))
LARGE_COUNTS = (4, 6, 8, 10)


@dataclass(frozen=True)
class Condition:
    condition_id: str
    suite: str
    layout_id: int
    seed: int
    command: tuple[float, float, float]
    goal_xy: tuple[float, float]
    terrain_row: int
    terrain_col: int
    large_obstacle_count: int
    small_obstacle_count: int
    timeout_steps: int

    def to_json(self) -> dict[str, object]:
        value = asdict(self)
        value["command"] = list(self.command)
        value["goal_xy"] = list(self.goal_xy)
        return value

    @classmethod
    def from_json(cls, value: dict[str, object]) -> "Condition":
        return cls(
            condition_id=str(value["condition_id"]),
            suite=str(value["suite"]),
            layout_id=int(value["layout_id"]),
            seed=int(value["seed"]),
            command=tuple(float(v) for v in value["command"]),  # type: ignore[arg-type]
            goal_xy=tuple(float(v) for v in value["goal_xy"]),  # type: ignore[arg-type]
            terrain_row=int(value["terrain_row"]),
            terrain_col=int(value["terrain_col"]),
            large_obstacle_count=int(value["large_obstacle_count"]),
            small_obstacle_count=int(value["small_obstacle_count"]),
            timeout_steps=int(value["timeout_steps"]),
        )


def _runway_timeout_steps(vx: float, *, distance_m: float = 10.0, dt: float = 0.02) -> int:
    if float(vx) <= 0.0:
        raise ValueError("runway vx must be positive")
    return int(math.ceil(1.5 * float(distance_m) / float(vx) / float(dt)))


def _condition_seed(seed: int, suite: str, layout_id: int, variant: int) -> int:
    # Avoid Python's process-randomized hash so manifests are stable across runs.
    suite_code = sum((index + 1) * ord(char) for index, char in enumerate(suite))
    return int(seed) + 1009 * suite_code + 7919 * int(layout_id) + int(variant)


def build_conditions(suite: str, layout_count: int, seed: int) -> list[Condition]:
    """Build deterministic conditions for one benchmark suite."""

    name = str(suite)
    count = int(layout_count)
    if count <= 0:
        raise ValueError("layout_count must be positive")
    if name == "complex_mixed":
        conditions = []
        for layout_id in range(count):
            for command_index, command in enumerate(
                (tuple(values) for values in ((vx, vy, yaw) for vx in MIXED_VX for vy in MIXED_VY for yaw in MIXED_YAW))
            ):
                conditions.append(
                    Condition(
                        condition_id=f"complex_mixed-l{layout_id:04d}-c{command_index:03d}",
                        suite=name,
                        layout_id=layout_id,
                        seed=_condition_seed(seed, name, layout_id, command_index),
                        command=command,
                        goal_xy=(0.0, 0.0),
                        terrain_row=layout_id % 10,
                        terrain_col=layout_id % 20,
                        large_obstacle_count=2,
                        small_obstacle_count=5,
                        timeout_steps=int(math.ceil(20.0 / 0.02)),
                    )
                )
        return conditions

    if name not in {"large_runway", "small_runway"}:
        raise ValueError(f"unknown benchmark suite: {name}")

    conditions: list[Condition] = []
    for layout_id in range(count):
        for density_index, density_value in enumerate(SMALL_DENSITIES):
            density, small_count = density_value
            for speed_index, vx in enumerate(RUNWAY_VX):
                large_count = LARGE_COUNTS[density_index]
                if name == "large_runway":
                    small_count_for_condition = 0
                    large_count_for_condition = large_count
                else:
                    small_count_for_condition = small_count
                    large_count_for_condition = 0
                variant = density_index * len(RUNWAY_VX) + speed_index
                conditions.append(
                    Condition(
                        condition_id=f"{name}-l{layout_id:04d}-d{density_index:02d}-v{speed_index:02d}",
                        suite=name,
                        layout_id=layout_id,
                        seed=_condition_seed(seed, name, layout_id, variant),
                        command=(float(vx), 0.0, 0.0),
                        goal_xy=(10.0, 0.0),
                        terrain_row=0,
                        terrain_col=layout_id,
                        large_obstacle_count=large_count_for_condition,
                        small_obstacle_count=small_count_for_condition,
                        timeout_steps=_runway_timeout_steps(vx),
                    )
                )
    random.Random(int(seed)).shuffle(conditions)
    return conditions


def write_manifest(path: str | Path, conditions: Iterable[Condition]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(conditions)
    ids = [row.condition_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate condition_id")
    target.write_text(
        "".join(json.dumps(row.to_json(), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_manifest(path: str | Path) -> list[Condition]:
    rows: list[Condition] = []
    seen: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            condition = Condition.from_json(json.loads(line))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid manifest line {line_number}") from exc
        if condition.condition_id in seen:
            raise ValueError(f"duplicate condition_id: {condition.condition_id}")
        seen.add(condition.condition_id)
        rows.append(condition)
    if not rows:
        raise ValueError("manifest is empty")
    return rows
