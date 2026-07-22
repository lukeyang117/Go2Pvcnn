from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import torch


DIRECTIONS = {
    "forward": (1.0, 0.0),
    "backward": (-1.0, 0.0),
    "left": (0.0, 1.0),
    "right": (0.0, -1.0),
    "forward_left": (2.0**-0.5, 2.0**-0.5),
    "forward_right": (2.0**-0.5, -(2.0**-0.5)),
    "backward_left": (-(2.0**-0.5), 2.0**-0.5),
    "backward_right": (-(2.0**-0.5), -(2.0**-0.5)),
}
SPEEDS = (0.1, 0.2, 0.4)
YAW_RATES = (-0.4, 0.0, 0.4)
SHAPES = ("sphere", "cuboid")
COLLISION_PARTS = ("foot", "knee", "calf", "thigh", "base")


def _masked_max(value: torch.Tensor, mask: torch.Tensor, *, dimensions: tuple[int, ...]) -> torch.Tensor:
    masked = torch.where(mask, value, torch.full_like(value, -float("inf")))
    maximum = masked.amax(dim=dimensions)
    return torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))


def summarize_attitude_trace(
    *,
    root_rpy_w: torch.Tensor,
    control: torch.Tensor,
    foot_pos_w: torch.Tensor,
    foot_height_w: torch.Tensor,
    contact_state: torch.Tensor,
    joint_pos: torch.Tensor,
    line_search_alpha: torch.Tensor,
    foot_contact_offset: float,
    dt: float,
    foot_small_distance_m: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    rpy = torch.as_tensor(root_rpy_w)
    controls = torch.as_tensor(control, dtype=rpy.dtype, device=rpy.device)
    foot = torch.as_tensor(foot_pos_w, dtype=rpy.dtype, device=rpy.device)
    height = torch.as_tensor(foot_height_w, dtype=rpy.dtype, device=rpy.device)
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=rpy.device)
    joint = torch.as_tensor(joint_pos, dtype=rpy.dtype, device=rpy.device)
    alpha = torch.as_tensor(line_search_alpha, dtype=rpy.dtype, device=rpy.device)
    touchdown = torch.logical_and(torch.logical_not(contact[:, :-1]), contact[:, 1:])
    surface_error = foot[..., 2] - height - float(foot_contact_offset)
    touchdown_error = surface_error[:, 1:]
    stance_mask = contact
    swing_step_mask = torch.logical_not(contact[:, 1:])
    stance_step_mask = torch.logical_and(contact[:, :-1], contact[:, 1:])
    foot_step = torch.linalg.vector_norm(torch.diff(foot, dim=1), dim=-1)
    rpy_degrees = torch.rad2deg(rpy[..., :2])
    rpy_step_degrees = torch.diff(rpy_degrees, dim=1).abs()
    angular_velocity = controls[..., 3:5]
    angular_acceleration = torch.diff(angular_velocity, dim=1) / float(dt)
    joint_lower = joint.new_tensor((-1.0472, -0.6632, -2.721) * 4)
    joint_upper = joint.new_tensor((1.0472, 2.966, -0.837) * 4)
    joint_margin = torch.minimum(joint - joint_lower, joint_upper - joint)
    result = {
        "roll_abs_max_deg": rpy_degrees[..., 0].abs().amax(dim=1),
        "pitch_abs_max_deg": rpy_degrees[..., 1].abs().amax(dim=1),
        "roll_step_max_deg": rpy_step_degrees[..., 0].amax(dim=1),
        "pitch_step_max_deg": rpy_step_degrees[..., 1].amax(dim=1),
        "root_rp_rate_max_rps": angular_velocity.abs().amax(dim=(1, 2)),
        "root_rp_accel_max_rps2": angular_acceleration.abs().amax(dim=(1, 2)),
        "touchdown_count": touchdown.sum(dim=(1, 2)),
        "airborne_touchdown_5mm_count": torch.logical_and(
            touchdown, touchdown_error > 0.005
        ).sum(dim=(1, 2)),
        "airborne_touchdown_20mm_count": torch.logical_and(
            touchdown, touchdown_error > 0.020
        ).sum(dim=(1, 2)),
        "touchdown_surface_error_max_m": _masked_max(
            touchdown_error, touchdown, dimensions=(1, 2)
        ),
        "touchdown_roll_abs_max_deg": _masked_max(
            rpy_degrees[:, 1:, 0, None].abs().expand_as(touchdown_error),
            touchdown,
            dimensions=(1, 2),
        ),
        "touchdown_pitch_abs_max_deg": _masked_max(
            rpy_degrees[:, 1:, 1, None].expand_as(touchdown_error).abs(),
            touchdown,
            dimensions=(1, 2),
        ),
        "stance_airborne_5mm_frames": torch.logical_and(
            stance_mask, surface_error > 0.005
        ).sum(dim=(1, 2)),
        "stance_surface_error_abs_max_m": _masked_max(
            surface_error.abs(), stance_mask, dimensions=(1, 2)
        ),
        "swing_foot_step_max_m": _masked_max(
            foot_step, swing_step_mask, dimensions=(1, 2)
        ),
        "stance_foot_step_max_m": _masked_max(
            foot_step, stance_step_mask, dimensions=(1, 2)
        ),
        "foot_surface_height_max_m": surface_error.amax(dim=(1, 2)),
        "joint_margin_min_rad": joint_margin.amin(dim=(1, 2)),
        "line_search_zero_count": (alpha <= 0.0).sum(dim=1),
    }
    if foot_small_distance_m is not None:
        small_distance = torch.as_tensor(
            foot_small_distance_m, dtype=rpy.dtype, device=rpy.device
        )
        result["touchdown_on_small_count"] = torch.logical_and(
            touchdown, small_distance[:, 1:] <= 0.0
        ).sum(dim=(1, 2))
    return result


@dataclass(frozen=True)
class AttitudeCase:
    shape: str
    direction: str
    speed: float
    yaw_rate: float
    obstacle_x: float
    obstacle_y: float


def _cases() -> list[AttitudeCase]:
    cases: list[AttitudeCase] = []
    for shape in SHAPES:
        for direction_name, direction in DIRECTIONS.items():
            perpendicular = (-direction[1], direction[0])
            obstacle_x = 0.27 * direction[0] + 0.142 * perpendicular[0]
            obstacle_y = 0.27 * direction[1] + 0.142 * perpendicular[1]
            for speed in SPEEDS:
                for yaw_rate in YAW_RATES:
                    cases.append(
                        AttitudeCase(
                            shape=shape,
                            direction=direction_name,
                            speed=speed,
                            yaw_rate=yaw_rate,
                            obstacle_x=obstacle_x,
                            obstacle_y=obstacle_y,
                        )
                    )
    return cases


def _build_field(
    cases: list[AttitudeCase], origin_w: torch.Tensor, *, obstacles: bool = True
):
    from extension.joint_mpc_rti.terrain.field_builder import build_field_batch

    device = origin_w.device
    batch = len(cases)
    coordinate = (torch.arange(151, device=device, dtype=torch.float32) - 75.0) * 0.01
    grid_x = origin_w[:, 0, None, None] + coordinate.view(1, 151, 1)
    grid_y = origin_w[:, 1, None, None] + coordinate.view(1, 1, 151)
    grid_x = grid_x.expand(batch, 151, 151)
    grid_y = grid_y.expand(batch, 151, 151)
    center_x = torch.tensor([case.obstacle_x for case in cases], device=device).view(-1, 1, 1)
    center_y = torch.tensor([case.obstacle_y for case in cases], device=device).view(-1, 1, 1)
    dx = grid_x - center_x
    dy = grid_y - center_y
    radial = torch.sqrt(dx.square() + dy.square())
    sphere = torch.tensor([case.shape == "sphere" for case in cases], device=device).view(-1, 1, 1)
    sphere_inside = radial <= 0.06
    sphere_height = 0.10 + 0.06 * (1.0 - (radial / 0.06).square()).clamp_min(0.0).sqrt()
    cuboid_inside = torch.logical_and(dx.abs() <= 0.06, dy.abs() <= 0.06)
    inside = torch.where(sphere, sphere_inside, cuboid_inside)
    if not obstacles:
        inside = torch.zeros_like(inside)
    height = torch.where(
        inside,
        torch.where(sphere, sphere_height, torch.full_like(grid_x, 0.16)),
        torch.zeros_like(grid_x),
    )
    semantic = inside.to(torch.long)
    return build_field_batch(
        height_w=height,
        semantic_id=semantic,
        origin_w=origin_w,
        yaw_w=torch.zeros(batch, device=device),
        timestamp=torch.zeros(batch, device=device),
        version=torch.zeros(batch, dtype=torch.long, device=device),
        resolution=0.01,
        small_ids=(1,),
        large_ids=(2,),
    )


def run_probe(
    *, device: str = "cuda", steps: int = 160, obstacles: bool = True
) -> dict[str, object]:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
    from extension.joint_mpc_rti.planner import step
    from extension.joint_mpc_rti.terrain.query import query_world
    from extension.joint_mpc_rti.types import JointMpcRtiState
    try:
        from .helpers import make_state
    except ImportError:
        from tests.joint_mpc_rti.helpers import make_state

    cases = _cases()
    batch = len(cases)
    measured = make_state(batch, device=device)
    command = torch.tensor(
        [
            (
                case.speed * DIRECTIONS[case.direction][0],
                case.speed * DIRECTIONS[case.direction][1],
                case.yaw_rate,
            )
            for case in cases
        ],
        dtype=torch.float32,
        device=device,
    )
    cfg = JointMpcRtiCfg()
    cfg.solver.emit_loss_breakdown = False
    solver_state = None
    collision_frames = {
        part: torch.zeros(batch, dtype=torch.long, device=device) for part in COLLISION_PARTS
    }
    collision_phase_frames = {
        part: {
            phase: torch.zeros(batch, dtype=torch.long, device=device)
            for phase in ("swing", "touchdown", "continuing_stance")
        }
        for part in ("foot", "knee", "calf", "thigh")
    }
    max_penetration = {part: torch.zeros(batch, device=device) for part in COLLISION_PARTS}
    traces: dict[str, list[torch.Tensor]] = {
        name: []
        for name in ("root", "rpy", "control", "foot", "height", "distance", "contact", "joint", "alpha")
    }
    for frame in range(int(steps)):
        field = _build_field(cases, measured.root_pos_w, obstacles=obstacles)
        result = step(measured, command, field, solver_state, cfg)
        trajectory = result.full_trajectory
        if frame == 0:
            state0 = trajectory.state[:, 0]
            geometry0 = go2_fk(state0[:, :3], state0[:, 3:6], state0[:, 6:])
            query0 = query_world(field, geometry0.foot_pos_w)
            traces["root"].append(state0[:, :3])
            traces["rpy"].append(state0[:, 3:6])
            traces["control"].append(torch.zeros(batch, 18, device=device))
            traces["foot"].append(geometry0.foot_pos_w)
            traces["height"].append(query0.height_w)
            traces["distance"].append(query0.small_distance_m)
            traces["contact"].append(trajectory.contact_state[:, 0])
            traces["joint"].append(state0[:, 6:])
            traces["alpha"].append(torch.ones(batch, device=device))
        state1 = trajectory.state[:, 1]
        geometry = go2_fk(state1[:, :3], state1[:, 3:6], state1[:, 6:])
        body_count = int(geometry.body_samples_w.shape[1])
        collision_points = torch.cat(
            (
                geometry.foot_pos_w,
                geometry.knee_pos_w,
                geometry.shank_samples_w.reshape(batch, 12, 3),
                geometry.thigh_samples_w.reshape(batch, 12, 3),
                geometry.body_samples_w,
            ),
            dim=1,
        )
        collision_query = query_world(field, collision_points)
        foot_distance = collision_query.small_distance_m[:, :4]
        knee_distance = collision_query.small_distance_m[:, 4:8]
        calf_distance = collision_query.small_distance_m[:, 8:20].reshape(batch, 4, 3)
        thigh_distance = collision_query.small_distance_m[:, 20:32].reshape(batch, 4, 3)
        base_distance = collision_query.small_distance_m[:, 32 : 32 + body_count]
        foot_height = collision_query.height_w[:, :4]
        knee_height = collision_query.height_w[:, 4:8]
        calf_height = collision_query.height_w[:, 8:20].reshape(batch, 4, 3)
        thigh_height = collision_query.height_w[:, 20:32].reshape(batch, 4, 3)
        base_height = collision_query.height_w[:, 32 : 32 + body_count]

        def sphere_collision(position, distance, top_height, radius: float):
            vertical = torch.logical_and(
                position[..., 2] - float(radius) < top_height,
                position[..., 2] + float(radius) > 0.0,
            )
            collision = torch.logical_and(distance < float(radius), vertical)
            penetration = torch.where(
                collision,
                float(radius) - distance,
                torch.zeros_like(distance),
            )
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
        base_collision = torch.logical_and(
            base_distance < 0.0,
            torch.logical_and(
                geometry.body_samples_w[..., 2] < base_height,
                geometry.body_samples_w[..., 2] > 0.0,
            ),
        )
        base_penetration = torch.where(
            base_collision, -base_distance, torch.zeros_like(base_distance)
        )
        leg_collision = {
            "foot": foot_collision,
            "knee": knee_collision,
            "calf": calf_collision.any(dim=2),
            "thigh": thigh_collision.any(dim=2),
        }
        part_collision = {
            part: collision.any(dim=1) for part, collision in leg_collision.items()
        }
        part_collision["base"] = base_collision.any(dim=1)
        part_penetration = {
            "foot": foot_penetration.amax(dim=1),
            "knee": knee_penetration.amax(dim=1),
            "calf": calf_penetration.amax(dim=(1, 2)),
            "thigh": thigh_penetration.amax(dim=(1, 2)),
            "base": base_penetration.amax(dim=1),
        }
        contact_x0 = trajectory.contact_state[:, 0]
        contact_x1 = trajectory.contact_state[:, 1]
        phase_masks = {
            "swing": torch.logical_not(contact_x1),
            "touchdown": torch.logical_and(torch.logical_not(contact_x0), contact_x1),
            "continuing_stance": torch.logical_and(contact_x0, contact_x1),
        }
        for part in COLLISION_PARTS:
            collision_frames[part] += part_collision[part].to(torch.long)
            max_penetration[part] = torch.maximum(max_penetration[part], part_penetration[part])
        for part, collision in leg_collision.items():
            for phase, phase_mask in phase_masks.items():
                collision_phase_frames[part][phase] += torch.logical_and(
                    collision, phase_mask
                ).any(dim=1).to(torch.long)
        traces["root"].append(state1[:, :3])
        traces["rpy"].append(state1[:, 3:6])
        traces["control"].append(trajectory.control[:, 0])
        traces["foot"].append(geometry.foot_pos_w)
        traces["height"].append(foot_height)
        traces["distance"].append(foot_distance)
        traces["contact"].append(trajectory.contact_state[:, 1])
        traces["joint"].append(state1[:, 6:])
        traces["alpha"].append(trajectory.loss_breakdown["line_search_alpha"])
        measured = JointMpcRtiState(
            root_pos_w=state1[:, :3],
            root_rpy_w=state1[:, 3:6],
            joint_pos=state1[:, 6:],
            root_lin_vel_b=trajectory.control[:, 0, :3],
            root_ang_vel_b=trajectory.control[:, 0, 3:6],
            joint_vel=trajectory.control[:, 0, 6:],
        )
        solver_state = result.solver_state

    stacked = {name: torch.stack(values, dim=1) for name, values in traces.items()}
    metrics = summarize_attitude_trace(
        root_rpy_w=stacked["rpy"],
        control=stacked["control"],
        foot_pos_w=stacked["foot"],
        foot_height_w=stacked["height"],
        contact_state=stacked["contact"],
        joint_pos=stacked["joint"],
        line_search_alpha=stacked["alpha"],
        foot_contact_offset=float(cfg.gait.foot_contact_offset),
        dt=float(cfg.runtime.dt),
        foot_small_distance_m=stacked["distance"],
    )
    for part in COLLISION_PARTS:
        metrics[f"{part}_collision_frames"] = collision_frames[part]
        metrics[f"{part}_max_penetration_m"] = max_penetration[part]
    for part, phases in collision_phase_frames.items():
        for phase, value in phases.items():
            metrics[f"{part}_{phase}_collision_frames"] = value
    root_delta = stacked["root"][:, -1, :2] - stacked["root"][:, 0, :2]
    direction_tensor = torch.tensor(
        [DIRECTIONS[case.direction] for case in cases], device=device
    )
    progress = (root_delta * direction_tensor).sum(dim=1)
    expected_progress = torch.tensor(
        [case.speed * float(steps) * float(cfg.runtime.dt) for case in cases], device=device
    )
    rows: list[dict[str, object]] = []
    cpu_metrics = {name: value.detach().cpu() for name, value in metrics.items()}
    for index, case in enumerate(cases):
        row = {
            "shape": case.shape,
            "direction": case.direction,
            "speed_mps": case.speed,
            "yaw_rate_rps": case.yaw_rate,
            "progress_ratio": float((progress[index] / expected_progress[index].clamp_min(1.0e-6)).item()),
        }
        row.update({name: float(value[index].item()) for name, value in cpu_metrics.items()})
        rows.append(row)
    metric_names = tuple(cpu_metrics)
    total_names = [
        "touchdown_count",
        "airborne_touchdown_5mm_count",
        "airborne_touchdown_20mm_count",
        "stance_airborne_5mm_frames",
        "touchdown_on_small_count",
        "line_search_zero_count",
        *(f"{part}_collision_frames" for part in COLLISION_PARTS),
        *(
            f"{part}_{phase}_collision_frames"
            for part in ("foot", "knee", "calf", "thigh")
            for phase in ("swing", "touchdown", "continuing_stance")
        ),
    ]
    aggregate = {
        "case_count": len(rows),
        "steps": int(steps),
        "obstacles": bool(obstacles),
        "worst": {
            name: max(rows, key=lambda row: float(row[name])) for name in metric_names
        },
        "totals": {
            name: sum(float(row[name]) for row in rows)
            for name in total_names
        },
        "rows": rows,
    }
    return aggregate


def compact_report(result: dict[str, object]) -> dict[str, object]:
    rows = result["rows"]
    assert isinstance(rows, list)

    def grouped(field: str) -> dict[str, object]:
        output: dict[str, object] = {}
        for value in sorted({str(row[field]) for row in rows}):
            group = [row for row in rows if str(row[field]) == value]
            output[value] = {
                "cases": len(group),
                "roll_abs_max_deg": max(float(row["roll_abs_max_deg"]) for row in group),
                "pitch_abs_max_deg": max(float(row["pitch_abs_max_deg"]) for row in group),
                "rp_step_max_deg": max(
                    max(float(row["roll_step_max_deg"]), float(row["pitch_step_max_deg"]))
                    for row in group
                ),
                "touchdowns": sum(float(row["touchdown_count"]) for row in group),
                "airborne_touchdown_5mm": sum(
                    float(row["airborne_touchdown_5mm_count"]) for row in group
                ),
                "airborne_touchdown_20mm": sum(
                    float(row["airborne_touchdown_20mm_count"]) for row in group
                ),
                "touchdown_error_max_m": max(
                    float(row["touchdown_surface_error_max_m"]) for row in group
                ),
                "stance_airborne_frames": sum(
                    float(row["stance_airborne_5mm_frames"]) for row in group
                ),
                "joint_margin_min_rad": min(float(row["joint_margin_min_rad"]) for row in group),
                "line_search_zero": sum(float(row["line_search_zero_count"]) for row in group),
                "collision_frames": {
                    part: sum(float(row[f"{part}_collision_frames"]) for row in group)
                    for part in COLLISION_PARTS
                },
            }
        return output

    selected_worst = {}
    worst = result["worst"]
    assert isinstance(worst, dict)
    for name in (
        "roll_abs_max_deg",
        "pitch_abs_max_deg",
        "roll_step_max_deg",
        "pitch_step_max_deg",
        "root_rp_rate_max_rps",
        "root_rp_accel_max_rps2",
        "touchdown_surface_error_max_m",
        "stance_surface_error_abs_max_m",
        "swing_foot_step_max_m",
        "stance_foot_step_max_m",
        *(f"{part}_collision_frames" for part in COLLISION_PARTS),
        *(f"{part}_max_penetration_m" for part in COLLISION_PARTS),
    ):
        selected_worst[name] = worst[name]
    ranked = sorted(
        rows,
        key=lambda row: (
            max(float(row["roll_abs_max_deg"]), float(row["pitch_abs_max_deg"]))
            + 100.0 * float(row["touchdown_surface_error_max_m"])
            + float(row["airborne_touchdown_20mm_count"])
        ),
        reverse=True,
    )[:12]

    def correlation(left_name: str, right_name: str) -> float:
        left = torch.tensor([float(row[left_name]) for row in rows])
        right = torch.tensor([float(row[right_name]) for row in rows])
        if left.std() <= 1.0e-9 or right.std() <= 1.0e-9:
            return 0.0
        return float(torch.corrcoef(torch.stack((left, right)))[0, 1].item())

    return {
        "case_count": result["case_count"],
        "steps": result["steps"],
        "obstacles": result["obstacles"],
        "totals": result["totals"],
        "correlations": {
            "roll_vs_airborne20": correlation(
                "roll_abs_max_deg", "airborne_touchdown_20mm_count"
            ),
            "pitch_vs_airborne20": correlation(
                "pitch_abs_max_deg", "airborne_touchdown_20mm_count"
            ),
            "line_search_zero_vs_airborne20": correlation(
                "line_search_zero_count", "airborne_touchdown_20mm_count"
            ),
            "roll_vs_stance_airborne": correlation(
                "roll_abs_max_deg", "stance_airborne_5mm_frames"
            ),
            "pitch_vs_stance_airborne": correlation(
                "pitch_abs_max_deg", "stance_airborne_5mm_frames"
            ),
        },
        "worst": selected_worst,
        "by_direction": grouped("direction"),
        "by_speed": grouped("speed_mps"),
        "by_yaw_rate": grouped("yaw_rate_rps"),
        "by_shape": grouped("shape"),
        "top_combined_cases": ranked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--obstacles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    result = run_probe(device=args.device, steps=args.steps, obstacles=bool(args.obstacles))
    print(json.dumps(result if args.full else compact_report(result), sort_keys=True))


if __name__ == "__main__":
    main()
