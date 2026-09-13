from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos

import torch
from torch import Tensor

from extension.parallelism.config import OfficialCollisionShapeSpec, ParallelismCfg
from extension.parallelism.kinematics import rpy_to_rotation_matrix


M1_LEG_NAMES = ("FL", "FR", "RL", "RR")
M1_PLANNER_JOINT_NAMES = tuple(
    name
    for prefix in ("FBL", "FAR", "RBL", "RAR")
    for name in (f"{prefix}_ABAD_JOINT", f"{prefix}_HIP_JOINT", f"{prefix}_KNEE_JOINT")
)
M1_ASSET_JOINT_NAMES = tuple(
    name
    for prefix in ("FBL", "FAR", "RBL", "RAR")
    for name in (
        f"{prefix}_ABAD_JOINT",
        f"{prefix}_HIP_JOINT",
        f"{prefix}_KNEE_JOINT",
        f"{prefix}_FOOT_JOINT",
    )
)
M1_WHEEL_JOINT_NAMES = tuple(name for name in M1_ASSET_JOINT_NAMES if name.endswith("FOOT_JOINT"))

M1_WHEEL_RADIUS_M = 0.095958
M1_WHEEL_THICKNESS_M = 0.04650
M1_WHEEL_HORIZONTAL_ENVELOPE_M = M1_WHEEL_RADIUS_M + M1_WHEEL_THICKNESS_M / 2.0
M1_SWING_CLEARANCE_M = 0.105
M1_ROOT_FOOTPRINT_OFFSETS_M = tuple(
    (x, y)
    for x in (-0.45, -0.225, 0.0, 0.225, 0.45)
    for y in (-0.45, -0.225, 0.0, 0.225, 0.45)
)
M1_DEFAULT_HIP_ANGLE_RAD = -0.15
M1_DEFAULT_KNEE_ANGLE_RAD = 1.10
M1_DEFAULT_JOINT_POS = tuple(
    value
    for _ in range(4)
    for value in (0.0, M1_DEFAULT_HIP_ANGLE_RAD, M1_DEFAULT_KNEE_ANGLE_RAD)
)
M1_DEFAULT_ASSET_JOINT_POS = tuple(
    value
    for _ in range(4)
    for value in (0.0, M1_DEFAULT_HIP_ANGLE_RAD, M1_DEFAULT_KNEE_ANGLE_RAD, 0.0)
)
M1_ROOT_Z_M = M1_WHEEL_RADIUS_M + 0.26 * cos(M1_DEFAULT_HIP_ANGLE_RAD) + 0.28 * cos(
    M1_DEFAULT_HIP_ANGLE_RAD + M1_DEFAULT_KNEE_ANGLE_RAD
)
M1_ABAD_LOWER = (-0.523, -0.697, -0.523, -0.697)
M1_ABAD_UPPER = (0.697, 0.523, 0.697, 0.523)
M1_HIP_LOWER = (-2.443,) * 4
M1_HIP_UPPER = (2.443,) * 4
M1_KNEE_LOWER = (-2.801,) * 4
M1_KNEE_UPPER = (2.801,) * 4


@dataclass(frozen=True)
class M1ParallelGeometry:
    hip_pos_w: Tensor
    hip_rot_w: Tensor
    foot_pos_w: Tensor
    knee_pos_w: Tensor
    calf_samples_w: Tensor
    thigh_samples_w: Tensor
    thigh_pos_w: Tensor
    thigh_rot_w: Tensor
    calf_pos_w: Tensor
    calf_rot_w: Tensor
    foot_rot_w: Tensor


def _rotation_x(angle: Tensor) -> Tensor:
    cosine, sine = torch.cos(angle), torch.sin(angle)
    zeros, ones = torch.zeros_like(angle), torch.ones_like(angle)
    return torch.stack(
        (
            torch.stack((ones, zeros, zeros), dim=-1),
            torch.stack((zeros, cosine, -sine), dim=-1),
            torch.stack((zeros, sine, cosine), dim=-1),
        ),
        dim=-2,
    )


def _rotation_y(angle: Tensor) -> Tensor:
    cosine, sine = torch.cos(angle), torch.sin(angle)
    zeros, ones = torch.zeros_like(angle), torch.ones_like(angle)
    return torch.stack(
        (
            torch.stack((cosine, zeros, sine), dim=-1),
            torch.stack((zeros, ones, zeros), dim=-1),
            torch.stack((-sine, zeros, cosine), dim=-1),
        ),
        dim=-2,
    )


def _constants(reference: Tensor, values) -> Tensor:
    return torch.tensor(values, dtype=reference.dtype, device=reference.device)


def _m1_fk_body(joint_pos: Tensor):
    leading = joint_pos.shape[:-1]
    angles = joint_pos.reshape(*leading, 4, 3)
    abad, hip, knee = angles.unbind(dim=-1)
    front = _constants(joint_pos, (1.0, 1.0, -1.0, -1.0)).view(*((1,) * len(leading)), 4)
    side = _constants(joint_pos, (1.0, -1.0, 1.0, -1.0)).view(*((1,) * len(leading)), 4)

    base_to_abad = torch.stack((0.272 * front, 0.065 * side, torch.zeros_like(front)), dim=-1)
    abad_to_hip = torch.stack((0.0575 * front, 0.039 * side, torch.zeros_like(front)), dim=-1)
    hip_to_knee = torch.stack((torch.zeros_like(front), 0.0442 * side, torch.full_like(front, -0.26)), dim=-1)
    knee_to_wheel = torch.stack((torch.zeros_like(front), 0.0592 * side, torch.full_like(front, -0.28)), dim=-1)

    abad_rot = _rotation_x(abad)
    hip_rot = torch.matmul(abad_rot, _rotation_y(hip))
    knee_rot = torch.matmul(hip_rot, _rotation_y(knee))
    wheel_rot = knee_rot
    abad_pos = base_to_abad
    hip_pos = abad_pos + torch.matmul(abad_rot, abad_to_hip[..., None]).squeeze(-1)
    knee_pos = hip_pos + torch.matmul(hip_rot, hip_to_knee[..., None]).squeeze(-1)
    wheel_pos = knee_pos + torch.matmul(knee_rot, knee_to_wheel[..., None]).squeeze(-1)
    return abad_pos, hip_pos, knee_pos, wheel_pos, abad_rot, hip_rot, knee_rot, wheel_rot


def m1_fk(root_pos_w: Tensor, root_rpy_w: Tensor, joint_pos: Tensor, *, capsule_samples: int = 5) -> M1ParallelGeometry:
    root_pos = torch.as_tensor(root_pos_w)
    root_rpy = torch.as_tensor(root_rpy_w, dtype=root_pos.dtype, device=root_pos.device)
    joint = torch.as_tensor(joint_pos, dtype=root_pos.dtype, device=root_pos.device)
    if root_pos.shape[-1] != 3 or root_rpy.shape != root_pos.shape or joint.shape[:-1] != root_pos.shape[:-1] or joint.shape[-1] != 12:
        raise ValueError("root_pos_w/root_rpy_w/joint_pos must have shapes [...,3], [...,3], [...,12]")

    body_values = _m1_fk_body(joint)
    rotation = rpy_to_rotation_matrix(root_rpy)

    def world_pos(value: Tensor) -> Tensor:
        return torch.einsum("...ij,...lj->...li", rotation, value) + root_pos.unsqueeze(-2)

    def world_rot(value: Tensor) -> Tensor:
        return torch.einsum("...ij,...ljk->...lik", rotation, value)

    abad_body, hip_body, knee_body, wheel_body, abad_r, hip_r, knee_r, wheel_r = body_values
    abad_w, hip_w, knee_w, wheel_w = tuple(world_pos(value) for value in (abad_body, hip_body, knee_body, wheel_body))
    abad_rot_w, hip_rot_w, knee_rot_w, wheel_rot_w = tuple(world_rot(value) for value in (abad_r, hip_r, knee_r, wheel_r))
    alpha = torch.linspace(0.0, 1.0, int(capsule_samples), dtype=root_pos.dtype, device=root_pos.device)
    alpha = alpha.view(*((1,) * (wheel_w.ndim - 1)), int(capsule_samples), 1)
    calf_samples = knee_w.unsqueeze(-2) * (1.0 - alpha) + wheel_w.unsqueeze(-2) * alpha
    thigh_samples = hip_w.unsqueeze(-2) * (1.0 - alpha) + knee_w.unsqueeze(-2) * alpha
    return M1ParallelGeometry(
        hip_pos_w=hip_w,
        hip_rot_w=hip_rot_w,
        foot_pos_w=wheel_w,
        knee_pos_w=knee_w,
        calf_samples_w=calf_samples,
        thigh_samples_w=thigh_samples,
        thigh_pos_w=abad_w,
        thigh_rot_w=abad_rot_w,
        calf_pos_w=knee_w,
        calf_rot_w=knee_rot_w,
        foot_rot_w=wheel_rot_w,
    )


def _m1_joint_limits(reference: Tensor) -> tuple[Tensor, Tensor]:
    lower = _constants(reference, tuple(value for triple in zip(M1_ABAD_LOWER, M1_HIP_LOWER, M1_KNEE_LOWER) for value in triple))
    upper = _constants(reference, tuple(value for triple in zip(M1_ABAD_UPPER, M1_HIP_UPPER, M1_KNEE_UPPER) for value in triple))
    return lower, upper


def m1_joint_limit_mask(joint_pos: Tensor, leg_indices: Tensor | None = None) -> Tensor:
    joint = torch.as_tensor(joint_pos)
    if joint.shape[-1] == 12:
        lower, upper = _m1_joint_limits(joint)
    elif joint.shape[-1] == 3:
        if leg_indices is None:
            lower = joint.new_tensor((min(M1_ABAD_LOWER), min(M1_HIP_LOWER), min(M1_KNEE_LOWER)))
            upper = joint.new_tensor((max(M1_ABAD_UPPER), max(M1_HIP_UPPER), max(M1_KNEE_UPPER)))
        else:
            indices = torch.as_tensor(leg_indices, dtype=torch.long, device=joint.device)
            lower_all, upper_all = _m1_joint_limits(joint)
            lower = lower_all.reshape(4, 3).index_select(0, indices.reshape(-1)).reshape(*indices.shape, 3)
            upper = upper_all.reshape(4, 3).index_select(0, indices.reshape(-1)).reshape(*indices.shape, 3)
    else:
        raise ValueError("M1 joint positions must end in 3 or 12")
    return ((joint >= lower) & (joint <= upper)).all(dim=-1)


def m1_ik(root_pos_w: Tensor, root_rpy_w: Tensor, foot_target_w: Tensor) -> tuple[Tensor, Tensor]:
    root_pos = torch.as_tensor(root_pos_w)
    root_rpy = torch.as_tensor(root_rpy_w, dtype=root_pos.dtype, device=root_pos.device)
    target = torch.as_tensor(foot_target_w, dtype=root_pos.dtype, device=root_pos.device)
    if target.shape != root_pos.shape[:-1] + (4, 3):
        raise ValueError("foot_target_w must have shape [...,4,3]")
    flat_count = target[..., 0, 0].numel()
    root_flat = root_pos.reshape(flat_count, 3)
    rpy_flat = root_rpy.reshape(flat_count, 3)
    target_flat = target.reshape(flat_count, 4, 3)
    rotation = rpy_to_rotation_matrix(rpy_flat)
    target_body = torch.einsum(
        "nij,nlj->nli",
        rotation.transpose(-1, -2),
        target_flat - root_flat[:, None, :],
    )

    front = _constants(root_flat, (1.0, 1.0, -1.0, -1.0)).view(1, 4)
    side = _constants(root_flat, (1.0, -1.0, 1.0, -1.0)).view(1, 4)
    base_to_abad = torch.stack(
        (0.272 * front, 0.065 * side, torch.zeros_like(front)),
        dim=-1,
    )
    target_from_abad = target_body - base_to_abad

    # After removing the ABAD rotation, the two planar links have a fixed
    # lateral offset and lengths of 0.26 m and 0.28 m.
    py = target_from_abad[..., 1]
    pz = target_from_abad[..., 2]
    lateral_offset = 0.1424 * side
    lateral_sq = py.square() + pz.square() - lateral_offset.square()
    lateral = lateral_sq.clamp_min(0.0).sqrt()
    lateral_norm = (py.square() + pz.square()).clamp_min(1.0e-12).sqrt()
    abad = torch.atan2(pz, py) + torch.acos(
        (lateral_offset / lateral_norm).clamp(-1.0, 1.0)
    )

    cosine = torch.cos(abad)
    sine = torch.sin(abad)
    planar_x = target_from_abad[..., 0] - 0.0575 * front
    planar_z = -sine * py + cosine * pz
    reach_sq = planar_x.square() + planar_z.square()
    knee_cosine = (reach_sq - 0.26**2 - 0.28**2) / (2.0 * 0.26 * 0.28)
    knee = torch.acos(knee_cosine.clamp(-1.0, 1.0))
    target_angle = torch.atan2(-planar_x, -planar_z)
    hip = target_angle - torch.atan2(
        0.28 * torch.sin(knee),
        0.26 + 0.28 * torch.cos(knee),
    )

    joint_by_leg = torch.stack((abad, hip, knee), dim=-1)
    joint = joint_by_leg.reshape(flat_count, 12)
    lower, upper = _m1_joint_limits(joint)
    joint_in_limits = ((joint >= lower) & (joint <= upper)).reshape(flat_count, 4, 3).all(dim=-1)
    tolerance = 2.0e-3
    reach = reach_sq.clamp_min(0.0).sqrt()
    reachable = (
        (lateral_sq >= -(tolerance**2))
        & (reach >= (0.28 - 0.26) - tolerance)
        & (reach <= (0.28 + 0.26) + tolerance)
        & torch.isfinite(joint_by_leg).all(dim=-1)
        & joint_in_limits
    )

    # Keep a single FK check as the final geometric and numerical guard.
    residual = torch.linalg.vector_norm(
        m1_fk(root_flat, rpy_flat, joint).foot_pos_w - target_flat,
        dim=-1,
    )
    reachable = reachable & (residual <= 2.0e-3)
    return joint_by_leg.reshape(*target.shape[:-2], 4, 3), reachable.reshape(*target.shape[:-2], 4)


def _m1_specs() -> tuple[OfficialCollisionShapeSpec, ...]:
    specs = []
    prefixes = ("FBL", "FAR", "RBL", "RAR")
    legs = ("FL", "FR", "RL", "RR")
    for prefix, leg, side in zip(prefixes, legs, (1.0, -1.0, 1.0, -1.0)):
        specs.extend(
            (
                OfficialCollisionShapeSpec(f"{prefix}_abad_box", leg, "thigh", (0.044875, 0.003375 * side, 0.0), (1.0, 0.0, 0.0, 0.0), "box", size_l=(0.08975, 0.07125, 0.084187)),
                OfficialCollisionShapeSpec(f"{prefix}_hip_box", leg, "hip", (0.0, 0.056 * side, -0.12875), (1.0, 0.0, 0.0, 0.0), "box", size_l=(0.096987, 0.1246, 0.3545)),
                OfficialCollisionShapeSpec(f"{prefix}_knee_box", leg, "calf", (0.0, 0.0323 * side, -0.1375), (1.0, 0.0, 0.0, 0.0), "box", size_l=(0.09999, 0.0966, 0.375)),
                OfficialCollisionShapeSpec(f"{prefix}_wheel", leg, "foot", (0.0, 0.0073 * side, 0.0), (1.0, 0.0, 0.0, 0.0), "cylinder", radius_m=M1_WHEEL_RADIUS_M, height_m=M1_WHEEL_THICKNESS_M),
            )
        )
    return tuple(specs)


M1_CFG = replace(
    ParallelismCfg(),
    flat_root_clearance_m=M1_ROOT_Z_M,
    root_motion_scale=0.60,
    # M1's hip/knee envelope is wider than Go2's.  Keep the Go2 avoidance
    # rule, but give the M1 root enough lateral authority to clear a box.
    vy_limit=1.5,
    large_obstacle_lateral_speed_max_mps=1.5,
    # Match the terrain-relative root offset used by the M1 asset.  The root
    # is sampled at the centerline, as in Go2; taking the maximum over the
    # whole body footprint makes a descending stair lift the root from a rear
    # step and can move the front legs outside their IK workspace.
    terrain_following_root_clearance_m=0.34,
    terrain_following_root_max_drop_m=0.06,
    terrain_following_hold_support_on_large_obstacles=True,
    terrain_following_root_footprint_offsets_m=M1_ROOT_FOOTPRINT_OFFSETS_M,
    terrain_following_root_leading_footprint_only=True,
    swing_clearance_m=M1_SWING_CLEARANCE_M,
    min_swing_apex_m=0.0,
    swing_profile="minimum_clearance",
    swing_terrain_query_radius_m=M1_WHEEL_HORIZONTAL_ENVELOPE_M,
    foot_contact_offset_m=M1_WHEEL_RADIUS_M,
    # The M1 touchdown is validated against its wheel/body geometry below;
    # the inherited Go2 semantic expansion otherwise removes valid nearby landings.
    semantic_touchdown_margin_m=0.0,
    collision_margin_m=0.003,
    box_surface_points=26,
    cylinder_layers=3,
    cylinder_angles=12,
    contact_tolerant_collision_shape_names=(),
    contact_tolerant_collision_point_indices=(),
    contact_tolerant_support_shape_names=tuple(
        f"{prefix}_wheel" for prefix in ("FBL", "FAR", "RBL", "RAR")
    ),
    swing_start_tolerant_collision_shape_names=tuple(
        name
        for prefix in ("FBL", "FAR", "RBL", "RAR")
        for name in (f"{prefix}_wheel", f"{prefix}_knee_box")
    ),
    swing_collision_start_ignore_frames=12,
    official_collision_shapes=_m1_specs(),
)
