from __future__ import annotations

from dataclasses import dataclass

import torch


SHAPES = ("sphere", "cuboid", "cylinder", "capsule", "cone")
SPEEDS = (0.1, 0.2, 0.4)
PARTS = ("foot", "calf", "thigh", "base")


@dataclass(frozen=True)
class CrossingCaseMetrics:
    cross_successes: int
    cross_opportunities: int
    collision_frames: dict[str, int]
    valid_frames: dict[str, int]
    max_penetration_m: dict[str, float]
    stance_on_small_frames: int

    @property
    def cross_success_rate(self) -> float:
        return 0.0 if self.cross_opportunities == 0 else self.cross_successes / self.cross_opportunities


@dataclass(frozen=True)
class CrossingMatrixMetrics:
    cases: dict[tuple[str, float], CrossingCaseMetrics]
    invalid_count: int

    @property
    def cross_successes(self) -> int:
        return sum(case.cross_successes for case in self.cases.values())

    @property
    def cross_opportunities(self) -> int:
        return sum(case.cross_opportunities for case in self.cases.values())

    @property
    def overall_cross_success_rate(self) -> float:
        return 0.0 if self.cross_opportunities == 0 else self.cross_successes / self.cross_opportunities

    @property
    def cross_success_rate_by_case(self) -> dict[tuple[str, float], float]:
        return {key: value.cross_success_rate for key, value in self.cases.items()}

    @property
    def collision_frames(self) -> dict[str, int]:
        return {part: sum(case.collision_frames[part] for case in self.cases.values()) for part in PARTS}

    @property
    def stance_on_small_frames(self) -> int:
        return sum(case.stance_on_small_frames for case in self.cases.values())


def _shape_height(
    shape: str,
    x: torch.Tensor,
    y: torch.Tensor,
    center_x: torch.Tensor,
    center_y: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dx = x - center_x
    dy = y - float(center_y)
    radius = 0.06
    radial = torch.sqrt(dx.square() + dy.square())
    if shape == "cuboid":
        inside = torch.logical_and(dx.abs() <= radius, dy.abs() <= radius)
        height = torch.full_like(x, 0.16)
    elif shape == "capsule":
        segment_dx = (dx.abs() - 0.03).clamp_min(0.0)
        capsule_radius = torch.sqrt(segment_dx.square() + dy.square())
        inside = capsule_radius <= 0.04
        height = torch.full_like(x, 0.16)
    elif shape == "sphere":
        inside = radial <= radius
        normalized = (1.0 - (radial / radius).square()).clamp_min(0.0).sqrt()
        height = 0.10 + 0.06 * normalized
    elif shape == "cone":
        inside = radial <= radius
        height = 0.16 * (1.0 - radial / radius).clamp_min(0.0)
    else:
        inside = radial <= radius
        height = torch.full_like(x, 0.16)
    return inside, torch.where(inside, height, torch.zeros_like(height))


def _build_matrix_field(
    device: str,
    placements_per_case: int,
    *,
    case_rows: list[tuple[str, float, float]] | None = None,
    origin_w: torch.Tensor | None = None,
):
    from extension.joint_mpc_rti.terrain.field_builder import build_field_batch

    if case_rows is None:
        case_rows = []
        for shape in SHAPES:
            for speed in SPEEDS:
                for placement in range(placements_per_case):
                    center_x = 0.27 + 0.015 * placement
                    case_rows.append((shape, speed, center_x))
    batch = len(case_rows)
    origin = torch.zeros(batch, 3, device=device) if origin_w is None else origin_w.to(device=device).clone()
    origin[:, 2] = 0.0
    coordinate = (torch.arange(151, device=device, dtype=torch.float32) - 75.0) * 0.01
    grid_x = origin[:, 0, None, None] + coordinate.view(1, 151, 1)
    grid_y = origin[:, 1, None, None] + coordinate.view(1, 1, 151)
    grid_x = grid_x.expand(batch, 151, 151)
    grid_y = grid_y.expand(batch, 151, 151)
    semantic = torch.zeros(batch, 151, 151, dtype=torch.long, device=device)
    height = torch.zeros(batch, 151, 151, dtype=torch.float32, device=device)
    for shape in SHAPES:
        indices = [index for index, row in enumerate(case_rows) if row[0] == shape]
        ids = torch.tensor(indices, dtype=torch.long, device=device)
        center_x = torch.tensor([case_rows[index][2] for index in indices], device=device).view(-1, 1, 1)
        inside, shape_height = _shape_height(
            shape,
            grid_x.index_select(0, ids),
            grid_y.index_select(0, ids),
            center_x,
            0.142,
        )
        semantic_rows = torch.where(inside, torch.ones_like(inside, dtype=torch.long), torch.zeros_like(inside, dtype=torch.long))
        semantic.index_copy_(0, ids, semantic_rows)
        height.index_copy_(0, ids, shape_height)
    field = build_field_batch(
        height_w=height,
        semantic_id=semantic,
        origin_w=origin,
        yaw_w=torch.zeros(batch, device=device),
        timestamp=torch.zeros(batch, device=device),
        version=torch.zeros(batch, dtype=torch.long, device=device),
        resolution=0.01,
        small_ids=(1,),
        large_ids=(2,),
    )
    return field, case_rows


def run_crossing_matrix(
    *,
    device: str = "cuda",
    steps: int = 128,
    placements_per_case: int = 6,
) -> CrossingMatrixMetrics:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
    from extension.joint_mpc_rti.planner import step
    from extension.joint_mpc_rti.terrain.query import query_world
    from extension.joint_mpc_rti.types import JointMpcRtiState
    from .helpers import make_state

    field, case_rows = _build_matrix_field(device, placements_per_case)
    batch = len(case_rows)
    command = torch.tensor(
        [[speed, 0.0, 0.0] for _, speed, _ in case_rows],
        dtype=torch.float32,
        device=device,
    )
    measured = make_state(batch, device=device)
    cfg = JointMpcRtiCfg()
    cfg.solver.line_search_alphas = (1.0, 0.25)
    solver_state = None
    active = torch.zeros(batch, 4, dtype=torch.bool, device=device)
    pre_safe = torch.zeros_like(active)
    swing_over = torch.zeros_like(active)
    event_collision = torch.zeros_like(active)
    successes = torch.zeros(batch, dtype=torch.long, device=device)
    opportunities = torch.zeros(batch, dtype=torch.long, device=device)
    collision_frames = {part: torch.zeros(batch, dtype=torch.long, device=device) for part in PARTS}
    max_penetration = {part: torch.zeros(batch, device=device) for part in PARTS}
    valid_frames = {part: torch.zeros(batch, dtype=torch.long, device=device) for part in PARTS}
    invalid_count = torch.zeros((), dtype=torch.long, device=device)
    stance_on_small_frames = torch.zeros(batch, dtype=torch.long, device=device)
    previous_contact = None
    previous_foot_distance = None

    for _ in range(int(steps)):
        field, _ = _build_matrix_field(
            device,
            placements_per_case,
            case_rows=case_rows,
            origin_w=measured.root_pos_w,
        )
        result = step(measured, command, field, solver_state, cfg)
        trajectory = result.full_trajectory
        executed_state = trajectory.state[:, 1]
        geometry = go2_fk(executed_state[:, :3], executed_state[:, 3:6], executed_state[:, 6:])
        contact = trajectory.contact_state[:, 1]
        points = torch.cat(
            (
                geometry.foot_pos_w,
                geometry.shank_samples_w.reshape(batch, 12, 3),
                geometry.thigh_samples_w.reshape(batch, 12, 3),
                geometry.body_samples_w,
            ),
            dim=1,
        )
        queried = query_world(field, points)
        foot_distance = queried.small_distance_m[:, :4]
        calf_distance = queried.small_distance_m[:, 4:16].reshape(batch, 4, 3)
        thigh_distance = queried.small_distance_m[:, 16:28].reshape(batch, 4, 3)
        base_distance = queried.small_distance_m[:, 28:]
        top = torch.full((batch,), 0.16, device=device)

        def sphere_collision(position: torch.Tensor, distance: torch.Tensor, radius: float):
            vertical = torch.logical_and(
                position[..., 2] - float(radius) < top.view(batch, *([1] * (position.ndim - 2))),
                position[..., 2] + float(radius) > 0.0,
            )
            collision = torch.logical_and(distance < float(radius), vertical)
            penetration = torch.where(collision, float(radius) - distance, torch.zeros_like(distance))
            return collision, penetration

        foot_collision, foot_penetration = sphere_collision(geometry.foot_pos_w, foot_distance, 0.022)
        calf_collision, calf_penetration = sphere_collision(geometry.shank_samples_w, calf_distance, 0.040)
        thigh_collision, thigh_penetration = sphere_collision(geometry.thigh_samples_w, thigh_distance, 0.040)
        base_vertical = torch.logical_and(geometry.body_samples_w[..., 2] < top[:, None], geometry.body_samples_w[..., 2] > 0.0)
        base_collision = torch.logical_and(base_distance < 0.0, base_vertical)
        base_penetration = torch.where(base_collision, -base_distance, torch.zeros_like(base_distance))
        part_collision = {
            "foot": foot_collision.any(dim=1),
            "calf": calf_collision.any(dim=(1, 2)),
            "thigh": thigh_collision.any(dim=(1, 2)),
            "base": base_collision.any(dim=1),
        }
        part_penetration = {
            "foot": foot_penetration.amax(dim=1),
            "calf": calf_penetration.amax(dim=(1, 2)),
            "thigh": thigh_penetration.amax(dim=(1, 2)),
            "base": base_penetration.amax(dim=1),
        }
        for part in PARTS:
            collision_frames[part] += part_collision[part].to(torch.long)
            max_penetration[part] = torch.maximum(max_penetration[part], part_penetration[part])
            valid_frames[part] += queried.valid.all(dim=1).to(torch.long)
        stance_on_small_frames += torch.logical_and(contact, foot_distance <= 0.0).any(dim=1).to(torch.long)

        leg_collision = torch.logical_or(
            foot_collision,
            torch.logical_or(calf_collision.any(dim=2), thigh_collision.any(dim=2)),
        )
        if previous_contact is None:
            previous_contact = trajectory.contact_state[:, 0]
            previous_geometry = go2_fk(
                trajectory.state[:, 0, :3],
                trajectory.state[:, 0, 3:6],
                trajectory.state[:, 0, 6:],
            )
            previous_foot_distance = query_world(field, previous_geometry.foot_pos_w).small_distance_m
        liftoff = torch.logical_and(previous_contact, torch.logical_not(contact))
        touchdown = torch.logical_and(torch.logical_not(previous_contact), contact)
        active = torch.where(liftoff, torch.ones_like(active), active)
        pre_safe = torch.where(liftoff, previous_foot_distance > 0.0, pre_safe)
        swing_over = torch.where(liftoff, torch.zeros_like(swing_over), swing_over)
        event_collision = torch.where(liftoff, torch.zeros_like(event_collision), event_collision)
        swing_over = torch.logical_or(swing_over, torch.logical_and(active, foot_distance <= 0.022))
        event_collision = torch.logical_or(event_collision, torch.logical_and(active, leg_collision))
        opportunity = torch.logical_and(touchdown, swing_over)
        success = torch.logical_and(
            opportunity,
            torch.logical_and(pre_safe, torch.logical_and(foot_distance > 0.0, torch.logical_not(event_collision))),
        )
        opportunities += opportunity.sum(dim=1)
        successes += success.sum(dim=1)
        active = torch.where(touchdown, torch.zeros_like(active), active)
        previous_contact = contact
        previous_foot_distance = foot_distance
        invalid_count += torch.logical_not(trajectory.valid).sum()
        invalid_count += torch.logical_not(queried.valid).sum()
        first_control = trajectory.control[:, 0]
        measured = JointMpcRtiState(
            root_pos_w=executed_state[:, :3],
            root_rpy_w=executed_state[:, 3:6],
            joint_pos=executed_state[:, 6:],
            root_lin_vel_b=first_control[:, :3],
            root_ang_vel_b=first_control[:, 3:6],
            joint_vel=first_control[:, 6:],
        )
        solver_state = result.solver_state

    cases: dict[tuple[str, float], CrossingCaseMetrics] = {}
    for shape in SHAPES:
        for speed in SPEEDS:
            indices = [index for index, row in enumerate(case_rows) if row[0] == shape and row[1] == speed]
            ids = torch.tensor(indices, dtype=torch.long, device=device)
            cases[(shape, speed)] = CrossingCaseMetrics(
                cross_successes=int(successes.index_select(0, ids).sum().item()),
                cross_opportunities=int(opportunities.index_select(0, ids).sum().item()),
                collision_frames={part: int(collision_frames[part].index_select(0, ids).sum().item()) for part in PARTS},
                valid_frames={part: int(valid_frames[part].index_select(0, ids).sum().item()) for part in PARTS},
                max_penetration_m={part: float(max_penetration[part].index_select(0, ids).max().item()) for part in PARTS},
                stance_on_small_frames=int(stance_on_small_frames.index_select(0, ids).sum().item()),
            )
    return CrossingMatrixMetrics(cases=cases, invalid_count=int(invalid_count.item()))


__all__ = [
    "CrossingCaseMetrics",
    "CrossingMatrixMetrics",
    "PARTS",
    "SHAPES",
    "SPEEDS",
    "run_crossing_matrix",
]
