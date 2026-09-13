from __future__ import annotations

import torch
from torch import Tensor

from extension.parallelism.candidates import build_candidates
from extension.parallelism.collision import official_collision_mask
from extension.parallelism.config import ParallelismCfg
from extension.parallelism.ik import ik_go2
from extension.parallelism.kinematics import JOINT_LOWER, JOINT_UPPER, fk_go2
from extension.parallelism.root import clamp_command, rollout_root
from extension.parallelism.robot_backend import RobotBackend, get_robot_backend
from extension.parallelism.swing import minimum_clearance_swing_curve, terrain_aware_swing_curve
from extension.parallelism.terrain import expanded_obstacle_mask, query_expanded_obstacle, query_height_semantic_valid
from extension.parallelism.types import (
    ParallelismDiagnostics,
    ParallelismState,
    ParallelismTerrain,
    ParallelismTrajectory,
)


def _joint_limit_mask(
    joint_pos: Tensor,
    robot_backend: RobotBackend | None = None,
    leg_indices: Tensor | None = None,
) -> Tensor:
    backend = robot_backend or get_robot_backend("go2")
    return backend.joint_limit_mask(joint_pos, leg_indices)


def _selected_take(values: Tensor, selected_index: Tensor) -> Tensor:
    gather_index = selected_index[..., None, None].expand(*selected_index.shape, 1, values.shape[-1])
    return values.gather(dim=2, index=gather_index).squeeze(2)


def _selected_score_take(values: Tensor, selected_index: Tensor) -> Tensor:
    return values.gather(dim=2, index=selected_index[..., None]).squeeze(2)


def _contact_safe_current_foot(
    state: ParallelismState,
    root_pos: Tensor,
    root_rpy: Tensor,
    terrain: ParallelismTerrain,
    cfg: ParallelismCfg,
    robot_backend: RobotBackend | None = None,
) -> Tensor:
    foot = _current_foot_pos(state, root_pos, root_rpy, robot_backend)
    query = query_height_semantic_valid(terrain, foot[..., :2])
    safe_z = query.height + float(cfg.foot_contact_offset_m)
    z = torch.where(query.valid, torch.maximum(foot[..., 2], safe_z), foot[..., 2])
    return torch.cat((foot[..., :2], z[..., None]), dim=-1)


def _candidate_targets(
    state: ParallelismState,
    root_pos: Tensor,
    root_rpy: Tensor,
    candidate_w: Tensor,
    terrain: ParallelismTerrain,
    cfg: ParallelismCfg,
    robot_backend: RobotBackend | None = None,
) -> Tensor:
    batch, leg_count, candidate_count, _ = candidate_w.shape
    joint = torch.as_tensor(state.joint_pos, dtype=root_pos.dtype, device=root_pos.device)
    foot0 = _contact_safe_current_foot(state, root_pos[:, 0], root_rpy[:, 0], terrain, cfg, robot_backend)
    base = foot0[:, None, None, :, :].expand(batch, leg_count, candidate_count, leg_count, 3)
    active_leg = torch.eye(leg_count, dtype=torch.bool, device=root_pos.device).view(1, leg_count, 1, leg_count, 1)
    return torch.where(active_leg, candidate_w[..., None, :], base)


def _leg_reference_root(
    root_pos: Tensor,
    root_rpy: Tensor,
    cfg: ParallelismCfg,
    robot_backend: RobotBackend | None = None,
) -> tuple[Tensor, Tensor]:
    leg_index = torch.arange(4, device=root_pos.device)
    frame_ref = torch.where(
        (leg_index == 0) | (leg_index == 3),
        torch.zeros_like(leg_index),
        torch.full_like(leg_index, int(cfg.half_cycle)),
    )
    if robot_backend is not None and str(robot_backend.name).lower() == "m1":
        frame_ref = torch.where(
            (leg_index == 0) | (leg_index == 3),
            torch.full_like(leg_index, max(int(cfg.half_cycle) - 1, 0)),
            torch.full_like(leg_index, max(int(cfg.horizon) - 1, 0)),
        )
    return root_pos[:, frame_ref], root_rpy[:, frame_ref]


def _collision_mask(
    terrain: ParallelismTerrain,
    geometry,
    cfg: ParallelismCfg,
    robot_backend: RobotBackend | None = None,
) -> tuple[Tensor, Tensor]:
    backend = robot_backend or get_robot_backend("go2")
    return backend.collision_mask(terrain, geometry, cfg)


def _contact_tolerant_indices(cfg: ParallelismCfg, *, device: torch.device) -> Tensor:
    names = tuple(spec.name for spec in cfg.official_collision_shapes)
    indices = [idx for idx, name in enumerate(names) if name in set(cfg.contact_tolerant_collision_shape_names)]
    return torch.tensor(indices, dtype=torch.long, device=device)


def _suppress_contact_tolerant(bits: Tensor, cfg: ParallelismCfg) -> Tensor:
    indices = _contact_tolerant_indices(cfg, device=bits.device)
    if int(indices.numel()) == 0:
        return bits
    suppressed = bits.clone()
    suppressed.index_fill_(dim=-1, index=indices, value=False)
    return suppressed


def _terrain_swing_curve(
    start_w: Tensor,
    touchdown_w: Tensor,
    terrain: ParallelismTerrain,
    cfg: ParallelismCfg,
    *,
    frames: int,
) -> Tensor:
    kwargs = {
        "frames": int(frames),
        "clearance_m": cfg.swing_clearance_m,
        "min_apex_m": cfg.min_swing_apex_m,
    }
    radius = float(getattr(cfg, "swing_terrain_query_radius_m", 0.0))
    if radius > 0.0:
        kwargs["terrain_query_radius_m"] = radius
    if str(getattr(cfg, "swing_profile", "parabolic")).lower() == "minimum_clearance":
        return minimum_clearance_swing_curve(start_w, touchdown_w, terrain, **kwargs)
    return terrain_aware_swing_curve(start_w, touchdown_w, terrain, **kwargs)


def _ground_swing_curve(start_w: Tensor, touchdown_w: Tensor, *, frames: int) -> Tensor:
    start = torch.as_tensor(start_w)
    touchdown = torch.as_tensor(touchdown_w, dtype=start.dtype, device=start.device)
    tau = torch.linspace(0.0, 1.0, int(frames), dtype=start.dtype, device=start.device)
    tau_view = tau.view(*((1,) * (start.ndim - 1)), int(frames), 1)
    curve = (1.0 - tau_view) * start[..., None, :] + tau_view * touchdown[..., None, :]
    curve = curve.clone()
    curve[..., 2] = start[..., 2, None]
    return curve


def _swing_collision_mask(
    state: ParallelismState,
    root_pos: Tensor,
    root_rpy: Tensor,
    candidates,
    terrain: ParallelismTerrain,
    cfg: ParallelismCfg,
    robot_backend: RobotBackend | None = None,
    candidate_needs_swing: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    backend = robot_backend or get_robot_backend("go2")
    batch, leg_count, candidate_count, _ = candidates.candidate_w.shape
    half_cycle = int(cfg.half_cycle)
    needs_swing = None
    if candidate_needs_swing is not None:
        needs_swing = torch.as_tensor(
            candidate_needs_swing,
            dtype=torch.bool,
            device=root_pos.device,
        )
        expected_shape = (batch, leg_count, candidate_count)
        if tuple(needs_swing.shape) != expected_shape:
            raise ValueError(
                f"candidate_needs_swing must have shape {expected_shape}, got {tuple(needs_swing.shape)}"
            )
    current_foot = _contact_safe_current_foot(state, root_pos[:, 0], root_rpy[:, 0], terrain, cfg, backend)
    swing_ok = torch.ones(batch, leg_count, candidate_count, dtype=torch.bool, device=root_pos.device)
    swing_bits = torch.zeros(
        batch,
        leg_count,
        candidate_count,
        len(tuple(cfg.official_collision_shapes)),
        dtype=torch.bool,
        device=root_pos.device,
    )

    for leg_idx in range(leg_count):
        phase_start = 0 if leg_idx in (0, 3) else half_cycle
        phase_stop = phase_start + half_cycle
        eval_stop = int(cfg.horizon) if leg_idx in (0, 3) else phase_stop
        eval_count = eval_stop - phase_start
        root_phase_pos = root_pos[:, phase_start:eval_stop]
        root_phase_rpy = root_rpy[:, phase_start:eval_stop]
        swing = _terrain_swing_curve(
            current_foot[:, None, leg_idx].expand(batch, candidate_count, 3),
            candidates.candidate_w[:, leg_idx],
            terrain,
            cfg,
            frames=half_cycle,
        )
        ground_swing = _ground_swing_curve(
            current_foot[:, None, leg_idx].expand(batch, candidate_count, 3),
            candidates.candidate_w[:, leg_idx],
            frames=half_cycle,
        )
        if eval_count > half_cycle:
            stance = candidates.candidate_w[:, leg_idx, :, None].expand(batch, candidate_count, eval_count - half_cycle, 3)
            swing_path = torch.cat((swing, stance), dim=2)
            ground_path = torch.cat((ground_swing, stance), dim=2)
        else:
            swing_path = swing
            ground_path = ground_swing
        if needs_swing is None:
            active_path = swing_path
        else:
            active_path = torch.where(
                needs_swing[:, leg_idx, :, None, None],
                swing_path,
                ground_path,
            )
        foot_targets = current_foot[:, None, None, :, :].expand(batch, candidate_count, eval_count, 4, 3).clone()
        foot_targets[..., leg_idx, :] = active_path
        root_eval_pos = root_phase_pos[:, None].expand(batch, candidate_count, eval_count, 3)
        root_eval_rpy = root_phase_rpy[:, None].expand(batch, candidate_count, eval_count, 3)
        joint_candidate, reachable = backend.ik(root_eval_pos, root_eval_rpy, foot_targets)
        flat_count = batch * candidate_count * eval_count
        geometry = backend.fk(
            root_eval_pos.reshape(flat_count, 3),
            root_eval_rpy.reshape(flat_count, 3),
            joint_candidate.reshape(flat_count, 12),
        )
        sample_count = int(geometry.calf_samples_w.shape[-2])

        def _expand_candidate_leg(value: Tensor, *tail: int) -> Tensor:
            return value.reshape(batch, candidate_count * eval_count, 4, *tail).unsqueeze(1).expand(
                batch,
                leg_count,
                candidate_count * eval_count,
                4,
                *tail,
            )

        geometry = type(geometry)(
            hip_pos_w=_expand_candidate_leg(geometry.hip_pos_w, 3),
            hip_rot_w=_expand_candidate_leg(geometry.hip_rot_w, 3, 3),
            foot_pos_w=_expand_candidate_leg(geometry.foot_pos_w, 3),
            knee_pos_w=_expand_candidate_leg(geometry.knee_pos_w, 3),
            calf_samples_w=geometry.calf_samples_w.reshape(batch, candidate_count * eval_count, 4, sample_count, 3)
            .unsqueeze(1)
            .expand(batch, leg_count, candidate_count * eval_count, 4, sample_count, 3),
            thigh_samples_w=geometry.thigh_samples_w.reshape(batch, candidate_count * eval_count, 4, sample_count, 3)
            .unsqueeze(1)
            .expand(batch, leg_count, candidate_count * eval_count, 4, sample_count, 3),
            thigh_pos_w=_expand_candidate_leg(geometry.thigh_pos_w, 3),
            thigh_rot_w=_expand_candidate_leg(geometry.thigh_rot_w, 3, 3),
            calf_pos_w=_expand_candidate_leg(geometry.calf_pos_w, 3),
            calf_rot_w=_expand_candidate_leg(geometry.calf_rot_w, 3, 3),
            foot_rot_w=_expand_candidate_leg(geometry.foot_rot_w, 3, 3),
        )
        collision_ok, collision_bits = _collision_mask(terrain, geometry, cfg, backend)
        active_collision_bits = collision_bits[:, leg_idx].reshape(
            batch,
            candidate_count,
            eval_count,
            -1,
        ).clone()
        start_tolerant_names = set(getattr(cfg, "swing_start_tolerant_collision_shape_names", ()))
        if start_tolerant_names:
            shape_indices = [
                shape_idx
                for shape_idx, spec in enumerate(cfg.official_collision_shapes)
                if spec.name in start_tolerant_names
            ]
            if shape_indices:
                start_tolerant = torch.zeros(
                    batch,
                    candidate_count,
                    len(tuple(cfg.official_collision_shapes)),
                    dtype=torch.bool,
                    device=active_collision_bits.device,
                )
                start_tolerant[..., shape_indices] = True
                if needs_swing is not None:
                    start_tolerant &= needs_swing[:, leg_idx, :, None]
                active_collision_bits[:, :, 0] = torch.where(
                    start_tolerant,
                    torch.zeros_like(active_collision_bits[:, :, 0]),
                    active_collision_bits[:, :, 0],
                )
        contact_indices = _contact_tolerant_indices(cfg, device=active_collision_bits.device)
        if int(contact_indices.numel()) > 0:
            active_collision_bits[:, :, 0].index_fill_(dim=-1, index=contact_indices, value=False)
            active_collision_bits[:, :, half_cycle - 1].index_fill_(dim=-1, index=contact_indices, value=False)
        ignore_frames = min(max(int(getattr(cfg, "swing_collision_start_ignore_frames", 0)), 0), eval_count)
        if ignore_frames > 0 and phase_start == 0:
            if needs_swing is None:
                active_collision_bits[:, :, :ignore_frames] = False
            else:
                rolling_candidate = ~needs_swing[:, leg_idx, :, None, None]
                active_collision_bits[:, :, :ignore_frames] = torch.where(
                    rolling_candidate,
                    torch.zeros_like(active_collision_bits[:, :, :ignore_frames]),
                    active_collision_bits[:, :, :ignore_frames],
                )
        active_collision_ok = ~active_collision_bits.any(dim=-1)
        swing_ok[:, leg_idx] = active_collision_ok.all(dim=-1)
        swing_bits[:, leg_idx] = active_collision_bits.any(dim=2)
    return swing_ok, swing_bits


def _tracking_score(candidates, command: Tensor, cfg: ParallelismCfg) -> Tensor:
    displacement_error = candidates.offset_body.view(1, 1, int(cfg.candidates_per_leg), 2) - candidates.score_target_body[:, :, None, :]
    return displacement_error.square().sum(dim=-1)


def _current_foot_pos(
    state: ParallelismState,
    root_pos: Tensor,
    root_rpy: Tensor,
    robot_backend: RobotBackend | None = None,
) -> Tensor:
    root = torch.as_tensor(root_pos)
    if state.foot_pos_w is not None:
        return torch.as_tensor(state.foot_pos_w, dtype=root.dtype, device=root.device)
    joint = torch.as_tensor(state.joint_pos, dtype=root.dtype, device=root.device)
    rpy = torch.as_tensor(root_rpy, dtype=root.dtype, device=root.device)
    backend = robot_backend or get_robot_backend("go2")
    return backend.fk(root, rpy, joint).foot_pos_w


def _semantic_ok(semantic: Tensor, cfg: ParallelismCfg) -> Tensor:
    obstacle_ids = torch.tensor(
        tuple(cfg.obstacle_semantic_ids),
        dtype=semantic.dtype,
        device=semantic.device,
    )
    return ~(semantic[..., None] == obstacle_ids).any(dim=-1)


def _candidate_needs_swing(
    state: ParallelismState,
    candidates,
    terrain: ParallelismTerrain,
    cfg: ParallelismCfg,
    robot_backend: RobotBackend,
    terrain_following_mask: Tensor | None = None,
    root_pos_w: Tensor | None = None,
    root_rpy_w: Tensor | None = None,
) -> Tensor:
    """Mark each candidate whose wheel-center path requires a leg swing."""

    batch, leg_count, candidate_count, _ = candidates.candidate_w.shape
    result = torch.zeros(batch, leg_count, candidate_count, dtype=torch.bool, device=candidates.candidate_w.device)
    semantic_ids = tuple(getattr(robot_backend, "swing_semantic_ids", ()))
    if semantic_ids:
        ids = torch.tensor(semantic_ids, dtype=candidates.candidate_w.dtype, device=candidates.candidate_w.device)
        large_ids = torch.tensor((2,), dtype=candidates.candidate_w.dtype, device=candidates.candidate_w.device)
        large_present = (terrain.semantic_id.to(device=result.device) == 2).any(dim=(-1, -2))
        query_radius = float(getattr(cfg, "swing_terrain_query_radius_m", 0.0))
        angle_count = max(int(getattr(cfg, "cylinder_angles", 4)), 4)
        if query_radius > 0.0:
            angles = torch.arange(
                angle_count,
                dtype=candidates.candidate_w.dtype,
                device=candidates.candidate_w.device,
            ) * (2.0 * torch.pi / angle_count)
            ring = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1) * query_radius
            path_offsets = torch.cat((torch.zeros(1, 2, dtype=ring.dtype, device=ring.device), ring), dim=0)
        else:
            path_offsets = torch.zeros(1, 2, dtype=candidates.candidate_w.dtype, device=candidates.candidate_w.device)

        def semantic_path_hit(path_xy: Tensor, query_ids: Tensor) -> Tensor:
            points = path_xy.unsqueeze(-2) + path_offsets.view(
                *((1,) * (path_xy.ndim - 1)), path_offsets.shape[0], 2
            )
            query = query_height_semantic_valid(terrain, points)
            semantic = query.semantic
            return (semantic[..., None] == query_ids).any(dim=(-1, -2)).any(dim=-1)

        current_foot = _contact_safe_current_foot(
            state,
            state.root_pos_w,
            state.root_rpy_w,
            terrain,
            cfg,
            robot_backend,
        )
        tau = torch.linspace(
            0.0,
            1.0,
            int(cfg.half_cycle),
            dtype=candidates.candidate_w.dtype,
            device=candidates.candidate_w.device,
        )
        path_xy = (
            current_foot[:, :, None, None, :2] * (1.0 - tau.view(1, 1, 1, -1, 1))
            + candidates.candidate_w[:, :, :, None, :2] * tau.view(1, 1, 1, -1, 1)
        )
        result = semantic_path_hit(path_xy, ids)
        large_path_hit = semantic_path_hit(path_xy, large_ids)

        if root_pos_w is not None:
            root_pos = torch.as_tensor(root_pos_w, dtype=current_foot.dtype, device=current_foot.device)
            horizon = int(root_pos.shape[1])
            root_rpy = torch.as_tensor(
                root_rpy_w if root_rpy_w is not None else state.root_rpy_w,
                dtype=root_pos.dtype,
                device=root_pos.device,
            )
            if root_rpy.ndim == 2:
                root_rpy = root_rpy[:, None].expand(-1, horizon, -1)
            joint = torch.as_tensor(state.joint_pos, dtype=root_pos.dtype, device=root_pos.device)
            rolling_foot = robot_backend.fk(
                root_pos.reshape(batch * horizon, 3),
                root_rpy.reshape(batch * horizon, 3),
                joint[:, None].expand(-1, horizon, -1).reshape(batch * horizon, 12),
            ).foot_pos_w.reshape(batch, horizon, leg_count, 3)
            for leg_idx in range(leg_count):
                phase_start = 0 if leg_idx in (0, 3) else int(cfg.half_cycle)
                phase_stop = horizon if leg_idx in (0, 3) else min(horizon, phase_start + int(cfg.half_cycle))
                rolling_path = rolling_foot[:, phase_start:phase_stop, leg_idx, :2]
                rolling_hit = semantic_path_hit(rolling_path, ids)
                rolling_large_hit = semantic_path_hit(
                    rolling_path,
                    large_ids,
                )
                result[:, leg_idx, :] = result[:, leg_idx, :] | rolling_hit[:, None]
                large_path_hit[:, leg_idx, :] = large_path_hit[:, leg_idx, :] | rolling_large_hit[:, None]

        if terrain_following_mask is not None and str(robot_backend.name).lower() == "m1":
            following = torch.as_tensor(
                terrain_following_mask,
                dtype=torch.bool,
                device=result.device,
            ).reshape(batch, 1, 1)
            result = result | (following & ~large_present[:, None, None])
        # Large boxes are handled by root lateral avoidance.  They must not
        # turn a terrain-following candidate into a leg swing that tries to
        # cross the box with the M1 body collision envelope.
        result = result & ~large_path_hit
    return result


def _assemble_foot_targets(
    state: ParallelismState,
    root_pos: Tensor,
    root_rpy: Tensor,
    selected_foothold_w: Tensor,
    terrain: ParallelismTerrain,
    cfg: ParallelismCfg,
    robot_backend: RobotBackend | None = None,
    leg_swing: Tensor | None = None,
) -> Tensor:
    batch = root_pos.shape[0]
    backend = robot_backend or get_robot_backend("go2")
    foot0 = _contact_safe_current_foot(state, root_pos[:, 0], root_rpy[:, 0], terrain, cfg, backend)
    if leg_swing is not None:
        leg_swing = torch.as_tensor(leg_swing, dtype=torch.bool, device=root_pos.device)
        if leg_swing.shape != (batch, 4):
            raise ValueError(f"leg_swing must have shape {(batch, 4)}, got {tuple(leg_swing.shape)}")
        horizon = int(cfg.horizon)
        target = foot0[:, None].expand(-1, horizon, -1, -1).clone()
        half_cycle = int(cfg.half_cycle)
        first_swing = _terrain_swing_curve(
            foot0[:, (0, 3)],
            selected_foothold_w[:, (0, 3)],
            terrain,
            cfg,
            frames=half_cycle,
        ).transpose(1, 2)
        first_ground_swing = _ground_swing_curve(
            foot0[:, (0, 3)],
            selected_foothold_w[:, (0, 3)],
            frames=half_cycle,
        ).transpose(1, 2)
        second_swing = _terrain_swing_curve(
            foot0[:, (1, 2)],
            selected_foothold_w[:, (1, 2)],
            terrain,
            cfg,
            frames=half_cycle,
        ).transpose(1, 2)
        second_ground_swing = _ground_swing_curve(
            foot0[:, (1, 2)],
            selected_foothold_w[:, (1, 2)],
            frames=half_cycle,
        ).transpose(1, 2)
        first_mask = leg_swing[:, (0, 3)][:, None, :, None]
        second_mask = leg_swing[:, (1, 2)][:, None, :, None]
        target[:, :half_cycle, (0, 3)] = torch.where(
            first_mask,
            first_swing,
            first_ground_swing,
        )
        target[:, half_cycle:, (0, 3)] = selected_foothold_w[:, None, (0, 3)]
        target[:, half_cycle:, (1, 2)] = torch.where(
            second_mask,
            second_swing,
            second_ground_swing,
        )
        target[:, :half_cycle, (1, 2)] = foot0[:, None, (1, 2)]
        return target

    first = foot0[:, None].expand(-1, int(cfg.half_cycle), -1, -1).clone()
    second = first.clone()
    first_swing = _terrain_swing_curve(
        foot0[:, (0, 3)],
        selected_foothold_w[:, (0, 3)],
        terrain,
        cfg,
        frames=int(cfg.half_cycle),
    )
    second_swing = _terrain_swing_curve(
        foot0[:, (1, 2)],
        selected_foothold_w[:, (1, 2)],
        terrain,
        cfg,
        frames=int(cfg.half_cycle),
    )
    first[:, :, (0, 3)] = first_swing.transpose(1, 2)
    second[:, :, (0, 3)] = selected_foothold_w[:, None, (0, 3)]
    second[:, :, (1, 2)] = second_swing.transpose(1, 2)
    return torch.cat((first, second), dim=1).reshape(batch, int(cfg.horizon), 4, 3)


def _contact_state(
    root_pos: Tensor,
    cfg: ParallelismCfg,
    leg_swing: Tensor | None = None,
) -> Tensor:
    contact = torch.ones(root_pos.shape[0], int(cfg.horizon), 4, dtype=torch.bool, device=root_pos.device)
    contact[:, : int(cfg.half_cycle), (0, 3)] = False
    contact[:, int(cfg.half_cycle) :, (1, 2)] = False
    if leg_swing is not None:
        leg_swing = torch.as_tensor(leg_swing, dtype=torch.bool, device=root_pos.device)
        contact[:, : int(cfg.half_cycle), (0, 3)] = ~leg_swing[:, None, (0, 3)]
        contact[:, int(cfg.half_cycle) :, (1, 2)] = ~leg_swing[:, None, (1, 2)]
    return contact


def _hold_trajectory(
    state: ParallelismState,
    cfg: ParallelismCfg,
    robot_backend: RobotBackend,
) -> ParallelismTrajectory:
    root_pos = torch.as_tensor(state.root_pos_w)
    root_rpy = torch.as_tensor(state.root_rpy_w, dtype=root_pos.dtype, device=root_pos.device)
    joint_pos = torch.as_tensor(state.joint_pos, dtype=root_pos.dtype, device=root_pos.device)
    foot_pos = _current_foot_pos(state, root_pos, root_rpy, robot_backend)
    batch = int(root_pos.shape[0])
    candidate_center = foot_pos.clone()
    candidate_count = int(cfg.candidates_per_leg)
    candidate_w = foot_pos[:, :, None, :].expand(-1, -1, candidate_count, -1).clone()
    candidate_valid = torch.ones(batch, 4, candidate_count, dtype=torch.bool, device=root_pos.device)
    collision_shape_count = len(tuple(cfg.official_collision_shapes))
    return ParallelismTrajectory(
        root_pos_w=root_pos[:, None].expand(-1, int(cfg.horizon), -1),
        root_rpy_w=root_rpy[:, None].expand(-1, int(cfg.horizon), -1),
        joint_pos=joint_pos[:, None].expand(-1, int(cfg.horizon), -1),
        foot_pos_w=foot_pos[:, None].expand(-1, int(cfg.horizon), -1, -1),
        contact_state=torch.ones(batch, int(cfg.horizon), 4, dtype=torch.bool, device=root_pos.device),
        valid=torch.ones(batch, dtype=torch.bool, device=root_pos.device),
        selected_foothold_w=foot_pos,
        selected_score=torch.zeros(batch, 4, dtype=root_pos.dtype, device=root_pos.device),
        diagnostics=ParallelismDiagnostics(
            candidate_center_w=candidate_center,
            candidate_w=candidate_w,
            candidate_score=torch.zeros(batch, 4, candidate_count, dtype=root_pos.dtype, device=root_pos.device),
            candidate_valid=candidate_valid,
            candidate_reject_bits=torch.zeros(batch, 4, candidate_count, 6, dtype=torch.bool, device=root_pos.device),
            candidate_collision_bits=torch.zeros(
                batch,
                4,
                1,
                collision_shape_count,
                dtype=torch.bool,
                device=root_pos.device,
            ),
            collision_shape_names=tuple(spec.name for spec in cfg.official_collision_shapes),
            collision_surface_point_count=max(
                int(cfg.box_surface_points),
                int(cfg.cylinder_layers) * int(cfg.cylinder_angles) + 2,
                int(cfg.sphere_surface_points),
            ),
            candidate_semantic=torch.zeros(batch, 4, candidate_count, dtype=torch.long, device=root_pos.device),
            fk_touchdown_semantic=torch.zeros(batch, 4, candidate_count, dtype=torch.long, device=root_pos.device),
            selected_index=torch.zeros(batch, 4, dtype=torch.long, device=root_pos.device),
            candidate_radius_m=float(cfg.candidate_radius_m),
            candidate_needs_swing=torch.zeros(batch, 4, candidate_count, dtype=torch.bool, device=root_pos.device),
            selected_needs_swing=torch.zeros(batch, 4, dtype=torch.bool, device=root_pos.device),
            touchdown_collision_bits=torch.zeros(
                batch,
                4,
                candidate_count,
                collision_shape_count,
                dtype=torch.bool,
                device=root_pos.device,
            ),
            swing_collision_bits=torch.zeros(
                batch,
                4,
                candidate_count,
                collision_shape_count,
                dtype=torch.bool,
                device=root_pos.device,
            ),
        ),
    )


def plan_trajectory(
    state: ParallelismState,
    command_body: Tensor,
    terrain: ParallelismTerrain,
    cfg: ParallelismCfg | None = None,
    terrain_following_mask: Tensor | None = None,
    robot_backend: RobotBackend | None = None,
) -> ParallelismTrajectory:
    backend = robot_backend or get_robot_backend("go2")
    cfg = cfg or backend.cfg
    command = clamp_command(command_body, cfg)
    if not bool(torch.any(command.abs() > 1.0e-6)):
        return _hold_trajectory(state, cfg, backend)
    root = rollout_root(state, command, terrain, cfg, terrain_following_mask=terrain_following_mask)
    candidates = build_candidates(root, state, command, terrain, cfg, robot_backend=backend)
    batch, leg_count, candidate_count, _ = candidates.candidate_w.shape
    candidate_needs_swing = _candidate_needs_swing(
        state,
        candidates,
        terrain,
        cfg,
        backend,
        terrain_following_mask,
        root_pos_w=root.root_pos_w,
        root_rpy_w=root.root_rpy_w,
    )

    root_ref_pos, root_ref_rpy = _leg_reference_root(
        root.root_pos_w,
        root.root_rpy_w,
        cfg,
        robot_backend=backend,
    )
    root_eval_pos = root_ref_pos[:, :, None, :].expand(batch, leg_count, candidate_count, 3)
    root_eval_rpy = root_ref_rpy[:, :, None, :].expand(batch, leg_count, candidate_count, 3)
    target = _candidate_targets(
        state,
        root.root_pos_w,
        root.root_rpy_w,
        candidates.candidate_w,
        terrain,
        cfg,
        backend,
    )
    joint_candidate, reachable = backend.ik(root_eval_pos, root_eval_rpy, target)
    geometry = backend.fk(
        root_eval_pos.reshape(batch * leg_count * candidate_count, 3),
        root_eval_rpy.reshape(batch * leg_count * candidate_count, 3),
        joint_candidate.reshape(batch * leg_count * candidate_count, 12),
    )
    sample_count = int(geometry.calf_samples_w.shape[-2])
    geometry = type(geometry)(
        hip_pos_w=geometry.hip_pos_w.reshape(batch, leg_count, candidate_count, 4, 3),
        hip_rot_w=geometry.hip_rot_w.reshape(batch, leg_count, candidate_count, 4, 3, 3),
        foot_pos_w=geometry.foot_pos_w.reshape(batch, leg_count, candidate_count, 4, 3),
        knee_pos_w=geometry.knee_pos_w.reshape(batch, leg_count, candidate_count, 4, 3),
        calf_samples_w=geometry.calf_samples_w.reshape(batch, leg_count, candidate_count, 4, sample_count, 3),
        thigh_samples_w=geometry.thigh_samples_w.reshape(batch, leg_count, candidate_count, 4, sample_count, 3),
        thigh_pos_w=geometry.thigh_pos_w.reshape(batch, leg_count, candidate_count, 4, 3),
        thigh_rot_w=geometry.thigh_rot_w.reshape(batch, leg_count, candidate_count, 4, 3, 3),
        calf_pos_w=geometry.calf_pos_w.reshape(batch, leg_count, candidate_count, 4, 3),
        calf_rot_w=geometry.calf_rot_w.reshape(batch, leg_count, candidate_count, 4, 3, 3),
        foot_rot_w=geometry.foot_rot_w.reshape(batch, leg_count, candidate_count, 4, 3, 3),
    )
    leg_select = torch.arange(leg_count, device=root.root_pos_w.device).view(1, leg_count, 1, 1, 1).expand(batch, leg_count, candidate_count, 1, 3)
    active_reachable = reachable.gather(3, leg_select[..., 0]).squeeze(3)
    active_joint = joint_candidate.gather(3, leg_select.expand(batch, leg_count, candidate_count, 1, 3)).squeeze(3)
    fk_touchdown = geometry.foot_pos_w.gather(3, leg_select).squeeze(3)

    valid_map_ok = candidates.candidate_valid_map
    active_leg_indices = torch.arange(leg_count, device=active_joint.device).view(1, leg_count, 1).expand(batch, leg_count, candidate_count)
    joint_ok = active_reachable & _joint_limit_mask(active_joint, backend, active_leg_indices)
    landing_query = query_height_semantic_valid(terrain, fk_touchdown[..., :2].reshape(batch, leg_count * candidate_count, 2))
    landing_height = landing_query.height.reshape(batch, leg_count, candidate_count)
    landing_target_height = landing_height + float(cfg.foot_contact_offset_m)
    landing_ok = landing_query.valid.reshape(batch, leg_count, candidate_count) & (
        (fk_touchdown[..., 2] - landing_target_height).abs() <= float(cfg.landing_tolerance_m)
    )
    _touchdown_collision_ok, touchdown_collision_bits = _collision_mask(terrain, geometry, cfg, backend)
    touchdown_collision_bits = _suppress_contact_tolerant(touchdown_collision_bits, cfg)
    touchdown_collision_ok = ~touchdown_collision_bits.any(dim=-1)
    swing_collision_ok, swing_collision_bits = _swing_collision_mask(
        state,
        root.root_pos_w,
        root.root_rpy_w,
        candidates,
        terrain,
        cfg,
        backend,
        candidate_needs_swing=(candidate_needs_swing if str(backend.name).lower() == "m1" else None),
    )
    collision_ok = touchdown_collision_ok & swing_collision_ok
    collision_bits = touchdown_collision_bits | swing_collision_bits
    semantic_obstacle = expanded_obstacle_mask(
        terrain,
        tuple(cfg.obstacle_semantic_ids),
        margin_m=float(cfg.semantic_touchdown_margin_m),
    )
    candidate_semantic_ok = ~query_expanded_obstacle(
        terrain,
        candidates.candidate_w[..., :2].reshape(batch, leg_count * candidate_count, 2),
        semantic_obstacle,
    ).reshape(batch, leg_count, candidate_count)
    fk_touchdown_semantic = landing_query.semantic.reshape(batch, leg_count, candidate_count)
    fk_touchdown_semantic_ok = ~query_expanded_obstacle(
        terrain,
        fk_touchdown[..., :2].reshape(batch, leg_count * candidate_count, 2),
        semantic_obstacle,
    ).reshape(batch, leg_count, candidate_count)
    candidate_valid = (
        valid_map_ok
        & joint_ok
        & landing_ok
        & collision_ok
        & candidate_semantic_ok
        & fk_touchdown_semantic_ok
    )
    reject_bits = torch.stack(
        (
            ~valid_map_ok,
            ~joint_ok,
            ~landing_ok,
            ~collision_ok,
            ~candidate_semantic_ok,
            ~fk_touchdown_semantic_ok,
        ),
        dim=-1,
    )
    score_raw = _tracking_score(candidates, command, cfg)
    score = torch.where(candidate_valid, score_raw, torch.full_like(score_raw, torch.inf))
    selected_index = score.argmin(dim=-1)
    selected_score = _selected_score_take(score, selected_index)
    selected_foothold = _selected_take(candidates.candidate_w, selected_index)
    selected_needs_swing = candidate_needs_swing.gather(
        dim=-1,
        index=selected_index.unsqueeze(-1),
    ).squeeze(-1)
    per_leg_has_valid = candidate_valid.any(dim=-1)
    selected_valid = per_leg_has_valid.all(dim=-1)

    leg_swing = selected_needs_swing if str(backend.name).lower() == "m1" else None
    foot_targets = _assemble_foot_targets(
        state,
        root.root_pos_w,
        root.root_rpy_w,
        selected_foothold,
        terrain,
        cfg,
        backend,
        leg_swing=leg_swing,
    )
    joint_traj, _reachable = backend.ik(root.root_pos_w, root.root_rpy_w, foot_targets)
    joint_pos = joint_traj.reshape(batch, int(cfg.horizon), 12)
    fk = backend.fk(
        root.root_pos_w.reshape(batch * int(cfg.horizon), 3),
        root.root_rpy_w.reshape(batch * int(cfg.horizon), 3),
        joint_pos.reshape(batch * int(cfg.horizon), 12),
    )
    foot_pos = fk.foot_pos_w.reshape(batch, int(cfg.horizon), 4, 3)
    current_root_pos = torch.as_tensor(state.root_pos_w, dtype=root.root_pos_w.dtype, device=root.root_pos_w.device)
    current_root_rpy = torch.as_tensor(state.root_rpy_w, dtype=root.root_pos_w.dtype, device=root.root_pos_w.device)
    current_joint = torch.as_tensor(state.joint_pos, dtype=root.root_pos_w.dtype, device=root.root_pos_w.device)
    current_foot = _current_foot_pos(state, current_root_pos, current_root_rpy, backend)
    output_planned = selected_valid | (torch.full_like(selected_valid, True) if not bool(cfg.standstill_fallback_enabled) else torch.zeros_like(selected_valid))
    env_mask_3 = output_planned[:, None, None]
    root_pos_out = torch.where(env_mask_3, root.root_pos_w, current_root_pos[:, None].expand(-1, int(cfg.horizon), -1))
    root_rpy_out = torch.where(env_mask_3, root.root_rpy_w, current_root_rpy[:, None].expand(-1, int(cfg.horizon), -1))
    joint_pos_out = torch.where(env_mask_3, joint_pos, current_joint[:, None].expand(-1, int(cfg.horizon), -1))
    foot_pos_out = torch.where(
        output_planned[:, None, None, None],
        foot_pos,
        current_foot[:, None].expand(-1, int(cfg.horizon), -1, -1),
    )
    contact_state = _contact_state(root.root_pos_w, cfg, leg_swing)
    contact_state_out = torch.where(
        output_planned[:, None, None],
        contact_state,
        torch.ones_like(contact_state),
    )
    selected_foothold_out = torch.where(
        output_planned[:, None, None],
        selected_foothold,
        current_foot,
    )
    selected_score_out = torch.where(
        selected_valid[:, None],
        selected_score,
        torch.full_like(selected_score, torch.inf),
    )

    return ParallelismTrajectory(
        root_pos_w=root_pos_out,
        root_rpy_w=root_rpy_out,
        joint_pos=joint_pos_out,
        foot_pos_w=foot_pos_out,
        contact_state=contact_state_out,
        valid=selected_valid,
        selected_foothold_w=selected_foothold_out,
        selected_score=selected_score_out,
        diagnostics=ParallelismDiagnostics(
            candidate_center_w=candidates.candidate_center_w,
            candidate_w=candidates.candidate_w,
            candidate_score=score,
            candidate_valid=candidate_valid,
            candidate_reject_bits=reject_bits,
            candidate_collision_bits=collision_bits,
            collision_shape_names=tuple(spec.name for spec in cfg.official_collision_shapes),
            collision_surface_point_count=max(
                int(cfg.box_surface_points),
                int(cfg.cylinder_layers) * int(cfg.cylinder_angles) + 2,
                int(cfg.sphere_surface_points),
            ),
            candidate_semantic=candidates.candidate_semantic,
            fk_touchdown_semantic=fk_touchdown_semantic,
            selected_index=selected_index,
            candidate_radius_m=float(cfg.candidate_radius_m),
            candidate_needs_swing=candidate_needs_swing,
            selected_needs_swing=selected_needs_swing,
            touchdown_collision_bits=touchdown_collision_bits,
            swing_collision_bits=swing_collision_bits,
        ),
    )
