from __future__ import annotations

from dataclasses import dataclass

import torch

from .small_obstacle_crossing_probe import PARTS, SHAPES, _build_matrix_field


STOP_OFFSETS = (-0.24, -0.20, -0.16, -0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24)


@dataclass(frozen=True)
class StopCaseMetrics:
    hold_frames: int
    stance_frames: int
    grounded_stance_frames: int
    physical_support_frames: int
    floating_stance_frames: int
    zero_support_frames: int
    max_consecutive_zero_support_frames: int
    stance_ground_gap_max_m: float
    max_stop_root_xy_drift_m: float
    stance_on_small_frames: int
    collision_frames: dict[str, int]
    max_penetration_m: dict[str, float]

    @property
    def recovered_support(self) -> bool:
        return self.max_consecutive_zero_support_frames <= 4


@dataclass(frozen=True)
class StopMatrixMetrics:
    cases: dict[tuple[str, float], StopCaseMetrics]
    joint_metrics: dict[tuple[str, float], dict[str, float]]
    invalid_count: int

    @property
    def support_recovery_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(case.recovered_support for case in self.cases.values()) / len(self.cases)

    @property
    def max_consecutive_zero_support_frames(self) -> int:
        return max(case.max_consecutive_zero_support_frames for case in self.cases.values())

    @property
    def max_zero_support_frames(self) -> int:
        return max(case.zero_support_frames for case in self.cases.values())

    @property
    def floating_stance_frames(self) -> int:
        return sum(case.floating_stance_frames for case in self.cases.values())

    @property
    def stance_ground_gap_max_m(self) -> float:
        return max(case.stance_ground_gap_max_m for case in self.cases.values())

    @property
    def max_stop_root_xy_drift_m(self) -> float:
        return max(case.max_stop_root_xy_drift_m for case in self.cases.values())

    @property
    def stance_on_small_frames(self) -> int:
        return sum(case.stance_on_small_frames for case in self.cases.values())

    @property
    def collision_frames(self) -> dict[str, int]:
        return {part: sum(case.collision_frames[part] for case in self.cases.values()) for part in PARTS}


def run_stop_matrix(
    *,
    device: str = "cuda",
    hold_steps: int = 32,
    max_steps: int = 224,
    shapes: tuple[str, ...] = SHAPES,
    offsets: tuple[float, ...] = STOP_OFFSETS,
    cfg=None,
) -> StopMatrixMetrics:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
    from extension.joint_mpc_rti.planner import step
    from extension.joint_mpc_rti.terrain.query import query_world
    from extension.joint_mpc_rti.types import JointMpcRtiState
    from .helpers import make_state

    center_x = 0.34
    case_rows = [(shape, 0.2, center_x) for shape in shapes for _ in offsets]
    stop_x = torch.tensor(
        [center_x + offset for _ in shapes for offset in offsets],
        dtype=torch.float32,
        device=device,
    )
    batch = len(case_rows)
    measured = make_state(batch, device=device)
    cfg = JointMpcRtiCfg() if cfg is None else cfg
    solver_state = None
    stopped = torch.zeros(batch, dtype=torch.bool, device=device)
    stop_root_xy = torch.zeros(batch, 2, device=device)
    hold_frames = torch.zeros(batch, dtype=torch.long, device=device)
    stance_frames = torch.zeros(batch, dtype=torch.long, device=device)
    grounded_stance_frames = torch.zeros(batch, dtype=torch.long, device=device)
    physical_support_frames = torch.zeros(batch, dtype=torch.long, device=device)
    floating_stance_frames = torch.zeros(batch, dtype=torch.long, device=device)
    zero_support_run = torch.zeros(batch, dtype=torch.long, device=device)
    zero_support_frames = torch.zeros(batch, dtype=torch.long, device=device)
    max_zero_support_run = torch.zeros(batch, dtype=torch.long, device=device)
    max_stance_gap = torch.zeros(batch, device=device)
    max_stop_drift = torch.zeros(batch, device=device)
    stance_on_small = torch.zeros(batch, dtype=torch.long, device=device)
    collision_frames = {part: torch.zeros(batch, dtype=torch.long, device=device) for part in PARTS}
    max_penetration = {part: torch.zeros(batch, device=device) for part in PARTS}
    invalid_count = torch.zeros((), dtype=torch.long, device=device)
    trace_root: list[torch.Tensor] = []
    trace_rpy: list[torch.Tensor] = []
    trace_foot: list[torch.Tensor] = []
    trace_contact: list[torch.Tensor] = []
    trace_command: list[torch.Tensor] = []
    trace_height: list[torch.Tensor] = []
    trace_distance: list[torch.Tensor] = []
    trace_collision = {part: [] for part in PARTS}
    trace_valid: list[torch.Tensor] = []

    for _ in range(int(max_steps)):
        field, _ = _build_matrix_field(device, 1, case_rows=case_rows, origin_w=measured.root_pos_w)
        just_stopped = torch.logical_and(torch.logical_not(stopped), measured.root_pos_w[:, 0] >= stop_x)
        stop_root_xy = torch.where(just_stopped[:, None], measured.root_pos_w[:, :2], stop_root_xy)
        stopped = torch.logical_or(stopped, just_stopped)
        command = torch.zeros(batch, 3, device=device)
        command[torch.logical_not(stopped), 0] = 0.2
        result = step(measured, command, field, solver_state, cfg)
        trajectory = result.full_trajectory
        executed_state = trajectory.state[:, 1]
        geometry = go2_fk(executed_state[:, :3], executed_state[:, 3:6], executed_state[:, 6:])
        contact = trajectory.contact_state[:, 1]
        points = torch.cat(
            (
                geometry.foot_pos_w,
                geometry.knee_pos_w,
                geometry.shank_samples_w.reshape(batch, 12, 3),
                geometry.thigh_samples_w.reshape(batch, 12, 3),
                geometry.body_samples_w,
            ),
            dim=1,
        )
        queried = query_world(field, points)
        foot_distance = queried.small_distance_m[:, :4]
        knee_distance = queried.small_distance_m[:, 4:8]
        calf_distance = queried.small_distance_m[:, 8:20].reshape(batch, 4, 3)
        thigh_distance = queried.small_distance_m[:, 20:32].reshape(batch, 4, 3)
        base_distance = queried.small_distance_m[:, 32:]
        foot_height = queried.height_w[:, :4]
        knee_height = queried.height_w[:, 4:8]
        calf_height = queried.height_w[:, 8:20].reshape(batch, 4, 3)
        thigh_height = queried.height_w[:, 20:32].reshape(batch, 4, 3)
        base_height = queried.height_w[:, 32:]
        held = torch.logical_and(stopped, hold_frames < int(hold_steps))

        gap = geometry.foot_pos_w[..., 2] - foot_height - float(cfg.gait.foot_contact_offset)
        grounded = torch.logical_and(contact, torch.abs(gap) <= 0.012)
        physical_support = torch.logical_and(torch.abs(gap) <= 0.012, foot_distance > 0.0)
        floating = torch.logical_and(contact, gap > 0.03)
        stance_frames += torch.logical_and(held[:, None], contact).sum(dim=1)
        grounded_stance_frames += torch.logical_and(held[:, None], grounded).sum(dim=1)
        physical_support_frames += torch.logical_and(held[:, None], physical_support).any(dim=1).to(torch.long)
        floating_stance_frames += torch.logical_and(held[:, None], floating).sum(dim=1)
        max_stance_gap = torch.maximum(
            max_stance_gap,
            torch.where(torch.logical_and(held[:, None], contact), gap, torch.zeros_like(gap)).amax(dim=1),
        )
        no_support = torch.logical_and(held, torch.logical_not(physical_support.any(dim=1)))
        zero_support_frames += no_support.to(torch.long)
        zero_support_run = torch.where(no_support, zero_support_run + 1, torch.zeros_like(zero_support_run))
        max_zero_support_run = torch.maximum(max_zero_support_run, zero_support_run)
        max_stop_drift = torch.where(
            held,
            torch.maximum(
                max_stop_drift,
                torch.linalg.vector_norm(executed_state[:, :2] - stop_root_xy, dim=1),
            ),
            max_stop_drift,
        )
        stance_on_small += torch.logical_and(
            held[:, None], torch.logical_and(contact, foot_distance <= 0.0)
        ).sum(dim=1)

        def sphere_collision(
            position: torch.Tensor,
            distance: torch.Tensor,
            top_height: torch.Tensor,
            radius: float,
        ):
            vertical = torch.logical_and(
                position[..., 2] - float(radius) < top_height,
                position[..., 2] + float(radius) > 0.0,
            )
            collision = torch.logical_and(distance < float(radius), vertical)
            penetration = torch.where(collision, float(radius) - distance, torch.zeros_like(distance))
            return collision, penetration

        foot_collision, foot_penetration = sphere_collision(
            geometry.foot_pos_w, foot_distance, foot_height, 0.022
        )
        knee_collision, knee_penetration = sphere_collision(
            geometry.knee_pos_w, knee_distance, knee_height, 0.040
        )
        calf_collision, calf_penetration = sphere_collision(
            geometry.shank_samples_w, calf_distance, calf_height, 0.040
        )
        thigh_collision, thigh_penetration = sphere_collision(
            geometry.thigh_samples_w, thigh_distance, thigh_height, 0.040
        )
        base_vertical = torch.logical_and(
            geometry.body_samples_w[..., 2] < base_height,
            geometry.body_samples_w[..., 2] > 0.0,
        )
        base_collision = torch.logical_and(base_distance < 0.0, base_vertical)
        part_collision = {
            "foot": foot_collision.any(dim=1),
            "knee": knee_collision.any(dim=1),
            "calf": calf_collision.any(dim=(1, 2)),
            "thigh": thigh_collision.any(dim=(1, 2)),
            "base": base_collision.any(dim=1),
        }
        part_penetration = {
            "foot": foot_penetration.amax(dim=1),
            "knee": knee_penetration.amax(dim=1),
            "calf": calf_penetration.amax(dim=(1, 2)),
            "thigh": thigh_penetration.amax(dim=(1, 2)),
            "base": torch.where(base_collision, -base_distance, torch.zeros_like(base_distance)).amax(dim=1),
        }
        if not trace_root:
            initial_geometry = go2_fk(
                trajectory.state[:, 0, :3], trajectory.state[:, 0, 3:6], trajectory.state[:, 0, 6:]
            )
            initial_query = query_world(field, initial_geometry.foot_pos_w)
            trace_root.append(trajectory.state[:, 0, :3])
            trace_rpy.append(trajectory.state[:, 0, 3:6])
            trace_foot.append(initial_geometry.foot_pos_w)
            trace_contact.append(trajectory.contact_state[:, 0])
            trace_command.append(command)
            trace_height.append(initial_query.height_w)
            trace_distance.append(initial_query.small_distance_m)
            for part in PARTS:
                trace_collision[part].append(torch.zeros(batch, dtype=torch.bool, device=device))
            trace_valid.append(trajectory.valid)
        trace_root.append(executed_state[:, :3])
        trace_rpy.append(executed_state[:, 3:6])
        trace_foot.append(geometry.foot_pos_w)
        trace_contact.append(contact)
        trace_command.append(command)
        trace_height.append(foot_height)
        trace_distance.append(foot_distance)
        for part in PARTS:
            trace_collision[part].append(part_collision[part])
        trace_valid.append(trajectory.valid)
        for part in PARTS:
            collision_frames[part] += torch.logical_and(held, part_collision[part]).to(torch.long)
            max_penetration[part] = torch.where(
                held,
                torch.maximum(max_penetration[part], part_penetration[part]),
                max_penetration[part],
            )

        invalid_count += torch.logical_and(held, torch.logical_not(trajectory.valid)).sum()
        invalid_count += torch.logical_and(held[:, None], torch.logical_not(queried.valid)).sum()
        hold_frames += held.to(torch.long)
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
        if bool(torch.all(hold_frames >= int(hold_steps))):
            break

    cases: dict[tuple[str, float], StopCaseMetrics] = {}
    joint_metrics: dict[tuple[str, float], dict[str, float]] = {}
    from .joint_metrics import JointMetricTrace, accumulate_joint_metrics

    stacked = {
        "root": torch.stack(trace_root, dim=1),
        "rpy": torch.stack(trace_rpy, dim=1),
        "foot": torch.stack(trace_foot, dim=1),
        "contact": torch.stack(trace_contact, dim=1),
        "command": torch.stack(trace_command, dim=1),
        "height": torch.stack(trace_height, dim=1),
        "distance": torch.stack(trace_distance, dim=1),
        "valid": torch.stack(trace_valid, dim=1),
    }
    stacked_collision = {part: torch.stack(trace_collision[part], dim=1) for part in PARTS}
    for index, shape in enumerate(shapes):
        for offset_index, offset in enumerate(offsets):
            row = index * len(offsets) + offset_index
            cases[(shape, offset)] = StopCaseMetrics(
                hold_frames=int(hold_frames[row]),
                stance_frames=int(stance_frames[row]),
                grounded_stance_frames=int(grounded_stance_frames[row]),
                physical_support_frames=int(physical_support_frames[row]),
                floating_stance_frames=int(floating_stance_frames[row]),
                zero_support_frames=int(zero_support_frames[row]),
                max_consecutive_zero_support_frames=int(max_zero_support_run[row]),
                stance_ground_gap_max_m=float(max_stance_gap[row]),
                max_stop_root_xy_drift_m=float(max_stop_drift[row]),
                stance_on_small_frames=int(stance_on_small[row]),
                collision_frames={part: int(collision_frames[part][row]) for part in PARTS},
                max_penetration_m={part: float(max_penetration[part][row]) for part in PARTS},
            )
            row_ids = torch.tensor([row], dtype=torch.long, device=device)
            joint_metrics[(shape, offset)] = accumulate_joint_metrics(
                JointMetricTrace(
                    root_pos_w=stacked["root"].index_select(0, row_ids),
                    root_rpy_w=stacked["rpy"].index_select(0, row_ids),
                    foot_pos_w=stacked["foot"].index_select(0, row_ids),
                    contact_state=stacked["contact"].index_select(0, row_ids),
                    command_body=stacked["command"].index_select(0, row_ids),
                    foot_height_w=stacked["height"].index_select(0, row_ids),
                    foot_small_distance_m=stacked["distance"].index_select(0, row_ids),
                    part_collision={
                        part: stacked_collision[part].index_select(0, row_ids) for part in PARTS
                    },
                    valid=stacked["valid"].index_select(0, row_ids),
                    dt=float(cfg.runtime.dt),
                )
            )
    return StopMatrixMetrics(
        cases=cases,
        joint_metrics=joint_metrics,
        invalid_count=int(invalid_count),
    )


__all__ = [
    "PARTS",
    "SHAPES",
    "STOP_OFFSETS",
    "StopCaseMetrics",
    "StopMatrixMetrics",
    "run_stop_matrix",
]
