from __future__ import annotations

import torch
from torch import Tensor

from extension.parallelism.candidates import build_candidates
from extension.parallelism.collision import official_collision_mask
from extension.parallelism.config import ParallelismCfg
from extension.parallelism.ik import ik_go2
from extension.parallelism.kinematics import JOINT_LOWER, JOINT_UPPER, fk_go2
from extension.parallelism.root import clamp_command, rollout_root
from extension.parallelism.swing import terrain_aware_swing_curve
from extension.parallelism.terrain import expanded_obstacle_mask, query_expanded_obstacle, query_height_semantic_valid
from extension.parallelism.types import (
    ParallelismDiagnostics,
    ParallelismState,
    ParallelismTerrain,
    ParallelismTrajectory,
)


def _joint_limit_mask(joint_pos: Tensor) -> Tensor:
    lower = torch.tensor(JOINT_LOWER, dtype=joint_pos.dtype, device=joint_pos.device).view(
        *((1,) * (joint_pos.ndim - 1)),
        3,
    )
    upper = torch.tensor(JOINT_UPPER, dtype=joint_pos.dtype, device=joint_pos.device).view(
        *((1,) * (joint_pos.ndim - 1)),
        3,
    )
    return ((joint_pos >= lower) & (joint_pos <= upper)).all(dim=-1)


def _selected_take(values: Tensor, selected_index: Tensor) -> Tensor:
    gather_index = selected_index[..., None, None].expand(*selected_index.shape, 1, values.shape[-1])
    return values.gather(dim=2, index=gather_index).squeeze(2)


def _selected_score_take(values: Tensor, selected_index: Tensor) -> Tensor:
    return values.gather(dim=2, index=selected_index[..., None]).squeeze(2)


def _candidate_targets(state: ParallelismState, root_pos: Tensor, root_rpy: Tensor, candidate_w: Tensor) -> Tensor:
    batch, leg_count, candidate_count, _ = candidate_w.shape
    joint = torch.as_tensor(state.joint_pos, dtype=root_pos.dtype, device=root_pos.device)
    foot0 = _current_foot_pos(state, root_pos[:, 0], root_rpy[:, 0])
    base = foot0[:, None, None, :, :].expand(batch, leg_count, candidate_count, leg_count, 3)
    active_leg = torch.eye(leg_count, dtype=torch.bool, device=root_pos.device).view(1, leg_count, 1, leg_count, 1)
    return torch.where(active_leg, candidate_w[..., None, :], base)


def _leg_reference_root(root_pos: Tensor, root_rpy: Tensor, cfg: ParallelismCfg) -> tuple[Tensor, Tensor]:
    leg_index = torch.arange(4, device=root_pos.device)
    frame_ref = torch.where(
        (leg_index == 0) | (leg_index == 3),
        torch.zeros_like(leg_index),
        torch.full_like(leg_index, int(cfg.half_cycle)),
    )
    return root_pos[:, frame_ref], root_rpy[:, frame_ref]


def _collision_mask(
    terrain: ParallelismTerrain,
    geometry,
    cfg: ParallelismCfg,
) -> tuple[Tensor, Tensor]:
    return official_collision_mask(terrain, geometry, cfg)


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


def _swing_collision_mask(
    state: ParallelismState,
    root_pos: Tensor,
    root_rpy: Tensor,
    candidates,
    terrain: ParallelismTerrain,
    cfg: ParallelismCfg,
) -> tuple[Tensor, Tensor]:
    batch, leg_count, candidate_count, _ = candidates.candidate_w.shape
    half_cycle = int(cfg.half_cycle)
    current_foot = _current_foot_pos(state, root_pos[:, 0], root_rpy[:, 0])
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
        root_phase_pos = root_pos[:, phase_start:phase_stop]
        root_phase_rpy = root_rpy[:, phase_start:phase_stop]
        swing = terrain_aware_swing_curve(
            current_foot[:, None, leg_idx].expand(batch, candidate_count, 3),
            candidates.candidate_w[:, leg_idx],
            terrain,
            frames=half_cycle,
            clearance_m=cfg.swing_clearance_m,
            min_apex_m=cfg.min_swing_apex_m,
        )
        foot_targets = current_foot[:, None, None, :, :].expand(batch, candidate_count, half_cycle, 4, 3).clone()
        foot_targets[..., leg_idx, :] = swing
        root_eval_pos = root_phase_pos[:, None].expand(batch, candidate_count, half_cycle, 3)
        root_eval_rpy = root_phase_rpy[:, None].expand(batch, candidate_count, half_cycle, 3)
        joint_candidate, reachable = ik_go2(root_eval_pos, root_eval_rpy, foot_targets)
        flat_count = batch * candidate_count * half_cycle
        geometry = fk_go2(
            root_eval_pos.reshape(flat_count, 3),
            root_eval_rpy.reshape(flat_count, 3),
            joint_candidate.reshape(flat_count, 12),
        )
        sample_count = int(geometry.calf_samples_w.shape[-2])

        def _expand_candidate_leg(value: Tensor, *tail: int) -> Tensor:
            return value.reshape(batch, candidate_count * half_cycle, 4, *tail).unsqueeze(1).expand(
                batch,
                leg_count,
                candidate_count * half_cycle,
                4,
                *tail,
            )

        geometry = type(geometry)(
            hip_pos_w=_expand_candidate_leg(geometry.hip_pos_w, 3),
            foot_pos_w=_expand_candidate_leg(geometry.foot_pos_w, 3),
            knee_pos_w=_expand_candidate_leg(geometry.knee_pos_w, 3),
            calf_samples_w=geometry.calf_samples_w.reshape(batch, candidate_count * half_cycle, 4, sample_count, 3)
            .unsqueeze(1)
            .expand(batch, leg_count, candidate_count * half_cycle, 4, sample_count, 3),
            thigh_samples_w=geometry.thigh_samples_w.reshape(batch, candidate_count * half_cycle, 4, sample_count, 3)
            .unsqueeze(1)
            .expand(batch, leg_count, candidate_count * half_cycle, 4, sample_count, 3),
            thigh_pos_w=_expand_candidate_leg(geometry.thigh_pos_w, 3),
            thigh_rot_w=_expand_candidate_leg(geometry.thigh_rot_w, 3, 3),
            calf_pos_w=_expand_candidate_leg(geometry.calf_pos_w, 3),
            calf_rot_w=_expand_candidate_leg(geometry.calf_rot_w, 3, 3),
            foot_rot_w=_expand_candidate_leg(geometry.foot_rot_w, 3, 3),
        )
        collision_ok, collision_bits = _collision_mask(terrain, geometry, cfg)
        active_collision_bits = collision_bits[:, leg_idx].reshape(
            batch,
            candidate_count,
            half_cycle,
            -1,
        ).clone()
        contact_indices = _contact_tolerant_indices(cfg, device=active_collision_bits.device)
        if int(contact_indices.numel()) > 0:
            active_collision_bits[:, :, 0].index_fill_(dim=-1, index=contact_indices, value=False)
            active_collision_bits[:, :, -1].index_fill_(dim=-1, index=contact_indices, value=False)
        active_collision_ok = ~active_collision_bits.any(dim=-1)
        swing_ok[:, leg_idx] = active_collision_ok.all(dim=-1)
        swing_bits[:, leg_idx] = active_collision_bits.any(dim=2)
    return swing_ok, swing_bits


def _tracking_score(candidates, command: Tensor, cfg: ParallelismCfg) -> Tensor:
    displacement_error = candidates.offset_body.view(1, 1, int(cfg.candidates_per_leg), 2) - candidates.score_target_body[:, :, None, :]
    return displacement_error.square().sum(dim=-1)


def _current_foot_pos(state: ParallelismState, root_pos: Tensor, root_rpy: Tensor) -> Tensor:
    root = torch.as_tensor(root_pos)
    if state.foot_pos_w is not None:
        return torch.as_tensor(state.foot_pos_w, dtype=root.dtype, device=root.device)
    joint = torch.as_tensor(state.joint_pos, dtype=root.dtype, device=root.device)
    rpy = torch.as_tensor(root_rpy, dtype=root.dtype, device=root.device)
    return fk_go2(root, rpy, joint).foot_pos_w


def _semantic_ok(semantic: Tensor, cfg: ParallelismCfg) -> Tensor:
    obstacle_ids = torch.tensor(
        tuple(cfg.obstacle_semantic_ids),
        dtype=semantic.dtype,
        device=semantic.device,
    )
    return ~(semantic[..., None] == obstacle_ids).any(dim=-1)


def _assemble_foot_targets(
    state: ParallelismState,
    root_pos: Tensor,
    root_rpy: Tensor,
    selected_foothold_w: Tensor,
    terrain: ParallelismTerrain,
    cfg: ParallelismCfg,
) -> Tensor:
    batch = root_pos.shape[0]
    foot0 = _current_foot_pos(state, root_pos[:, 0], root_rpy[:, 0])
    first = foot0[:, None].expand(-1, int(cfg.half_cycle), -1, -1).clone()
    second = first.clone()
    first_swing = terrain_aware_swing_curve(
        foot0[:, (0, 3)],
        selected_foothold_w[:, (0, 3)],
        terrain,
        frames=int(cfg.half_cycle),
        clearance_m=cfg.swing_clearance_m,
        min_apex_m=cfg.min_swing_apex_m,
    )
    second_swing = terrain_aware_swing_curve(
        foot0[:, (1, 2)],
        selected_foothold_w[:, (1, 2)],
        terrain,
        frames=int(cfg.half_cycle),
        clearance_m=cfg.swing_clearance_m,
        min_apex_m=cfg.min_swing_apex_m,
    )
    first[:, :, (0, 3)] = first_swing.transpose(1, 2)
    second[:, :, (0, 3)] = selected_foothold_w[:, None, (0, 3)]
    second[:, :, (1, 2)] = second_swing.transpose(1, 2)
    return torch.cat((first, second), dim=1).reshape(batch, int(cfg.horizon), 4, 3)


def _contact_state(root_pos: Tensor, cfg: ParallelismCfg) -> Tensor:
    contact = torch.ones(root_pos.shape[0], int(cfg.horizon), 4, dtype=torch.bool, device=root_pos.device)
    contact[:, : int(cfg.half_cycle), (0, 3)] = False
    contact[:, int(cfg.half_cycle) :, (1, 2)] = False
    return contact


def plan_trajectory(
    state: ParallelismState,
    command_body: Tensor,
    terrain: ParallelismTerrain,
    cfg: ParallelismCfg | None = None,
) -> ParallelismTrajectory:
    cfg = cfg or ParallelismCfg()
    command = clamp_command(command_body, cfg)
    root = rollout_root(state, command, terrain, cfg)
    candidates = build_candidates(root, state, command, terrain, cfg)
    batch, leg_count, candidate_count, _ = candidates.candidate_w.shape

    root_ref_pos, root_ref_rpy = _leg_reference_root(root.root_pos_w, root.root_rpy_w, cfg)
    root_eval_pos = root_ref_pos[:, :, None, :].expand(batch, leg_count, candidate_count, 3)
    root_eval_rpy = root_ref_rpy[:, :, None, :].expand(batch, leg_count, candidate_count, 3)
    target = _candidate_targets(state, root.root_pos_w, root.root_rpy_w, candidates.candidate_w)
    joint_candidate, reachable = ik_go2(root_eval_pos, root_eval_rpy, target)
    geometry = fk_go2(
        root_eval_pos.reshape(batch * leg_count * candidate_count, 3),
        root_eval_rpy.reshape(batch * leg_count * candidate_count, 3),
        joint_candidate.reshape(batch * leg_count * candidate_count, 12),
    )
    sample_count = int(geometry.calf_samples_w.shape[-2])
    geometry = type(geometry)(
        hip_pos_w=geometry.hip_pos_w.reshape(batch, leg_count, candidate_count, 4, 3),
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
    joint_ok = active_reachable & _joint_limit_mask(active_joint)
    landing_query = query_height_semantic_valid(terrain, fk_touchdown[..., :2].reshape(batch, leg_count * candidate_count, 2))
    landing_height = landing_query.height.reshape(batch, leg_count, candidate_count)
    landing_ok = landing_query.valid.reshape(batch, leg_count, candidate_count) & (
        (fk_touchdown[..., 2] - landing_height).abs() <= float(cfg.landing_tolerance_m)
    )
    _touchdown_collision_ok, touchdown_collision_bits = _collision_mask(terrain, geometry, cfg)
    touchdown_collision_bits = _suppress_contact_tolerant(touchdown_collision_bits, cfg)
    touchdown_collision_ok = ~touchdown_collision_bits.any(dim=-1)
    swing_collision_ok, swing_collision_bits = _swing_collision_mask(state, root.root_pos_w, root.root_rpy_w, candidates, terrain, cfg)
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
    per_leg_has_valid = candidate_valid.any(dim=-1)
    selected_valid = per_leg_has_valid.all(dim=-1)

    foot_targets = _assemble_foot_targets(state, root.root_pos_w, root.root_rpy_w, selected_foothold, terrain, cfg)
    joint_traj, _reachable = ik_go2(root.root_pos_w, root.root_rpy_w, foot_targets)
    joint_pos = joint_traj.reshape(batch, int(cfg.horizon), 12)
    fk = fk_go2(
        root.root_pos_w.reshape(batch * int(cfg.horizon), 3),
        root.root_rpy_w.reshape(batch * int(cfg.horizon), 3),
        joint_pos.reshape(batch * int(cfg.horizon), 12),
    )
    foot_pos = fk.foot_pos_w.reshape(batch, int(cfg.horizon), 4, 3)
    current_root_pos = torch.as_tensor(state.root_pos_w, dtype=root.root_pos_w.dtype, device=root.root_pos_w.device)
    current_root_rpy = torch.as_tensor(state.root_rpy_w, dtype=root.root_pos_w.dtype, device=root.root_pos_w.device)
    current_joint = torch.as_tensor(state.joint_pos, dtype=root.root_pos_w.dtype, device=root.root_pos_w.device)
    current_foot = _current_foot_pos(state, current_root_pos, current_root_rpy)
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
    contact_state = _contact_state(root.root_pos_w, cfg)
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
        ),
    )
