"""Full continuous rollout objective evaluated against the bound world field."""

from __future__ import annotations

import torch

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.losses.clearance import clearance_losses
from extension.joint_mpc_rti.losses.command import command_losses
from extension.joint_mpc_rti.losses.contact import (
    stance_losses,
    swing_losses,
    touchdown_geometry_losses,
    touchdown_losses,
)
from extension.joint_mpc_rti.losses.objective import terminal_losses, weighted_objective
from extension.joint_mpc_rti.losses.posture import posture_losses
from extension.joint_mpc_rti.losses.semantic import large_obstacle_losses, small_object_losses
from extension.joint_mpc_rti.losses.smoothness import smoothness_losses
from extension.joint_mpc_rti.model.go2_kinematics import rpy_to_rotation_matrix
from extension.joint_mpc_rti.model.rollout import JointMpcRollout
from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery, query_world, query_world_maybe_compiled
from extension.joint_mpc_rti.tensor_constants import constant_like
from extension.joint_mpc_rti.types import JointMpcTerrainField


def _packed_geometry_queries(
    field: JointMpcTerrainField,
    rollout: JointMpcRollout,
    cfg: JointMpcRtiCfg,
) -> tuple[JointMpcTerrainQuery, JointMpcTerrainQuery, JointMpcTerrainQuery, JointMpcTerrainQuery, JointMpcTerrainQuery]:
    batch, nodes = int(rollout.state.shape[0]), int(rollout.state.shape[1])
    shank = rollout.shank_samples_w.reshape(batch, nodes, 12, 3)
    root = rollout.state[..., :3].unsqueeze(2)
    packed = torch.cat(
        (rollout.foot_pos_w, rollout.knee_pos_w, shank, rollout.body_samples_w, root),
        dim=2,
    )
    points_per_node = int(packed.shape[2])
    points = packed.reshape(batch, nodes * points_per_node, 3)
    queried = (
        query_world(field, points)
        if bool(cfg.solver.compile_kernels) and points.is_cuda
        else query_world_maybe_compiled(field, points, enabled=False)
    )

    def section(start: int, stop: int) -> JointMpcTerrainQuery:
        def scalar(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch, nodes, points_per_node)[:, :, start:stop]

        def vector(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch, nodes, points_per_node, 2)[:, :, start:stop]

        return JointMpcTerrainQuery(
            height_w=scalar(queried.height_w),
            small_distance_m=scalar(queried.small_distance_m),
            large_distance_m=scalar(queried.large_distance_m),
            small_gradient_w=vector(queried.small_gradient_w),
            large_gradient_w=vector(queried.large_gradient_w),
            valid=scalar(queried.valid),
        )

    foot_stop = 4
    knee_stop = foot_stop + 4
    shank_stop = knee_stop + 12
    body_stop = shank_stop + int(rollout.body_samples_w.shape[2])
    return (
        section(0, foot_stop),
        section(foot_stop, knee_stop),
        section(knee_stop, shank_stop),
        section(shank_stop, body_stop),
        section(body_stop, body_stop + 1),
    )


def rollout_loss_breakdown(
    *,
    rollout: JointMpcRollout,
    nominal_rollout: JointMpcRollout | None = None,
    nominal_foot_pos_w: torch.Tensor | None = None,
    stance_anchor_w: torch.Tensor | None = None,
    contact_state: torch.Tensor,
    swing_weight: torch.Tensor,
    terrain_field: JointMpcTerrainField,
    command_body: torch.Tensor,
    joint_target: torch.Tensor,
    previous_control: torch.Tensor,
    cfg: JointMpcRtiCfg,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    state = rollout.state
    batch, nodes = int(state.shape[0]), int(state.shape[1])
    if nominal_foot_pos_w is None:
        if nominal_rollout is None:
            raise ValueError("nominal_rollout or nominal_foot_pos_w is required")
        nominal_foot = nominal_rollout.foot_pos_w
    else:
        nominal_foot = torch.as_tensor(nominal_foot_pos_w, dtype=state.dtype, device=state.device)
    stance_anchor = nominal_foot if stance_anchor_w is None else torch.as_tensor(
        stance_anchor_w, dtype=state.dtype, device=state.device
    )
    foot_query, knee_query, shank_query, body_query, root_query = _packed_geometry_queries(
        terrain_field,
        rollout,
        cfg,
    )
    foot_height = foot_query.height_w.reshape(batch, nodes, 4)
    knee_height = knee_query.height_w.reshape(batch, nodes, 4)
    shank_height = shank_query.height_w.reshape(batch, nodes, 4, 3)
    body_height = body_query.height_w.reshape(batch, nodes, -1)
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=state.device)
    swing_weight = torch.as_tensor(swing_weight, dtype=state.dtype, device=state.device)
    support_height = (foot_height * contact.to(state.dtype)).sum(dim=2) / contact.sum(dim=2).clamp_min(1).to(state.dtype)
    nominal_joint = constant_like(state, "nominal_joint_pos", cfg.gait.nominal_joint_pos)
    joint_lower = constant_like(state, "joint_lower", (-1.0472, -0.6632, -2.721) * 4)
    joint_upper = constant_like(state, "joint_upper", (1.0472, 2.966, -0.837) * 4)
    losses: dict[str, torch.Tensor] = {}
    losses.update(command_losses(state[..., :3], state[..., 3:6], rollout.control, command_body, dt=cfg.runtime.dt))
    losses.update(
        posture_losses(
            root_pos_w=state[:, :-1, :3],
            root_rpy_w=state[:, :-1, 3:6],
            joint_pos=state[:, :-1, 6:],
            joint_velocity=rollout.control[..., 6:],
            support_height=support_height[:, :-1],
            nominal_root_clearance=0.32,
            nominal_joint_pos=joint_target[:, :-1],
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            joint_velocity_limit=state.new_full((12,), 30.0),
            root_linear_velocity_b=rollout.control[..., :3],
            root_angular_velocity_b=rollout.control[..., 3:6],
            barrier_relaxation=cfg.solver.barrier_relaxation,
        )
    )
    losses.update(
        stance_losses(
            rollout.foot_pos_w,
            foot_height,
            contact,
            stance_anchor_w=stance_anchor,
            dt=cfg.runtime.dt,
        )
    )
    losses.update(
        swing_losses(
            foot_pos_w=rollout.foot_pos_w,
            nominal_foot_pos_w=nominal_foot,
            queried_height_w=foot_height,
            swing_mask=torch.logical_not(contact),
            swing_weight=swing_weight,
            dt=cfg.runtime.dt,
            terrain_margin=cfg.gait.nominal_swing_clearance,
            barrier_relaxation=cfg.solver.barrier_relaxation,
        )
    )
    losses.update(
        touchdown_losses(
            touchdown_pos_w=rollout.foot_pos_w[:, -1],
            queried_height_w=foot_height[:, -1],
            queried_valid=foot_query.valid.reshape(batch, nodes, 4)[:, -1],
        )
    )
    terminal_rotation = rpy_to_rotation_matrix(state[:, -1, 3:6])
    terminal_foot_root = torch.einsum(
        "bij,bkj->bki",
        terminal_rotation.transpose(-1, -2),
        rollout.foot_pos_w[:, -1] - state[:, -1, None, :3],
    )
    losses.update(
        touchdown_geometry_losses(
            terminal_foot_root,
            min_reach=0.20,
            max_reach=0.48,
            min_left_right_separation=0.12,
            barrier_relaxation=cfg.solver.barrier_relaxation,
        )
    )
    losses.update(
        clearance_losses(
            foot_pos_w=rollout.foot_pos_w,
            foot_height_w=foot_height,
            knee_pos_w=rollout.knee_pos_w,
            knee_height_w=knee_height,
            shank_pos_w=rollout.shank_samples_w,
            shank_height_w=shank_height,
            body_pos_w=rollout.body_samples_w,
            body_height_w=body_height,
            swing_mask=torch.logical_not(contact),
            barrier_relaxation=cfg.solver.barrier_relaxation,
        )
    )
    link_pos = torch.cat((rollout.knee_pos_w, rollout.shank_samples_w.reshape(batch, nodes, -1, 3)), dim=2)
    link_small_distance = torch.cat(
        (
            knee_query.small_distance_m.reshape(batch, nodes, 4),
            shank_query.small_distance_m.reshape(batch, nodes, -1),
        ),
        dim=2,
    )
    link_height = torch.cat((knee_height, shank_height.reshape(batch, nodes, -1)), dim=2)
    losses.update(
        small_object_losses(
            foot_pos_w=rollout.foot_pos_w,
            foot_small_distance=foot_query.small_distance_m.reshape(batch, nodes, 4),
            small_top_height=foot_height,
            small_distance_touchdown=foot_query.small_distance_m.reshape(batch, nodes, 4)[:, -1],
            link_pos_w=link_pos,
            link_small_distance=link_small_distance,
            link_top_height=link_height,
            swing_mask=torch.logical_not(contact),
            stance_mask=contact,
            extra_margin=cfg.gait.small_semantic_clearance,
            swing_weight=swing_weight,
            barrier_relaxation=cfg.solver.barrier_relaxation,
        )
    )
    root_large = body_query.large_distance_m.reshape(batch, nodes, -1)
    root_large_distance = root_query.large_distance_m.squeeze(-1)
    terminal_distance = root_large_distance[:, -1]
    terminal_approach = (terminal_distance - root_large_distance[:, -2]) / float(cfg.runtime.dt)
    knee_shank_large = torch.cat(
        (
            knee_query.large_distance_m.reshape(batch, nodes, 4),
            shank_query.large_distance_m.reshape(batch, nodes, -1),
        ),
        dim=2,
    )
    losses.update(
        large_obstacle_losses(
            root_footprint_distance=root_large,
            body_distance=root_large,
            foot_distance=foot_query.large_distance_m.reshape(batch, nodes, 4),
            knee_shank_distance=knee_shank_large,
            terminal_distance=terminal_distance,
            terminal_approach_speed=terminal_approach,
            barrier_relaxation=cfg.solver.barrier_relaxation,
        )
    )
    losses.update(smoothness_losses(rollout.control, previous_control=previous_control, dt=cfg.runtime.dt))
    losses.update(
        terminal_losses(
            terminal_control=rollout.control[:, -1],
            command_body=command_body,
            terminal_root_rpy=state[:, -1, 3:6],
            terminal_joint_pos=state[:, -1, 6:],
            nominal_joint_pos=nominal_joint,
            obstacle_distance=terminal_distance,
            obstacle_approach_speed=terminal_approach,
            contact_viability=contact[:, -1].to(state.dtype).mean(dim=1),
            barrier_relaxation=cfg.solver.barrier_relaxation,
        )
    )
    weights = {name: float(getattr(cfg.losses, name)) for name in losses}
    return losses, weighted_objective(losses, weights)


_COMPILED_ROLLOUT_LOSS_BREAKDOWN = torch.compile(
    rollout_loss_breakdown,
    fullgraph=True,
    dynamic=False,
    options={"triton.cudagraphs": False},
)


def rollout_loss_breakdown_maybe_compiled(**kwargs):
    rollout = kwargs["rollout"]
    cfg = kwargs["cfg"]
    if bool(cfg.solver.compile_kernels) and rollout.state.is_cuda:
        return _COMPILED_ROLLOUT_LOSS_BREAKDOWN(**kwargs)
    return rollout_loss_breakdown(**kwargs)


__all__ = ["rollout_loss_breakdown", "rollout_loss_breakdown_maybe_compiled"]
