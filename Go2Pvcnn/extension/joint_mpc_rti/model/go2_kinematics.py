"""Analytic Go2 forward kinematics and joint Jacobians."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from extension.joint_mpc_rti.tensor_constants import constant_like


HIP_OFFSETS = (
    (0.1934, 0.0465, 0.0),
    (0.1934, -0.0465, 0.0),
    (-0.1934, 0.0465, 0.0),
    (-0.1934, -0.0465, 0.0),
)
LEG_SIDE_SIGNS = (1.0, -1.0, 1.0, -1.0)
THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213
HIP_OFFSET_Y = 0.0955
FOOT_RADIUS = 0.022
SOLE_HALF_EXTENTS = (0.030, 0.020)
BASE_HALF_EXTENTS = (0.320, 0.090, 0.080)


@dataclass(frozen=True)
class Go2Geometry:
    foot_pos_w: Tensor
    knee_pos_w: Tensor
    shank_samples_w: Tensor
    thigh_samples_w: Tensor
    body_samples_w: Tensor


@dataclass(frozen=True)
class Go2LinkJacobians:
    calf_samples: Tensor
    thigh_samples: Tensor


@dataclass(frozen=True)
class Go2CollisionGeometry:
    foot_center_w: Tensor
    sole_corners_w: Tensor
    knee_center_w: Tensor
    calf_endpoints_w: Tensor
    thigh_endpoints_w: Tensor
    base_center_w: Tensor
    base_rotation_w: Tensor
    base_half_extents: Tensor
    base_corners_w: Tensor
    base_bottom_samples_w: Tensor


@dataclass(frozen=True)
class Go2SelectedLegCollisionGeometry:
    foot_center_w: Tensor
    knee_center_w: Tensor
    calf_endpoints_w: Tensor
    thigh_endpoints_w: Tensor


def rpy_to_rotation_matrix(root_rpy_w: Tensor) -> Tensor:
    """Return body-to-world rotation matrices for XYZ fixed-axis RPY."""
    rpy = torch.as_tensor(root_rpy_w)
    if rpy.ndim < 2 or int(rpy.shape[-1]) != 3:
        raise ValueError("root_rpy_w must have shape [...,3]")
    roll = rpy[..., 0]
    pitch = rpy[..., 1]
    yaw = rpy[..., 2]
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    row0 = torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), dim=-1)
    row1 = torch.stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), dim=-1)
    row2 = torch.stack((-sp, cp * sr, cp * cr), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _rpy_rotation_derivatives(root_rpy_w: Tensor) -> Tensor:
    """Return dR/dr, dR/dp, dR/dy for the XYZ fixed-axis RPY convention."""
    rpy = torch.as_tensor(root_rpy_w)
    roll, pitch, yaw = rpy.unbind(dim=-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    zero = torch.zeros_like(roll)

    droll = torch.stack(
        (
            torch.stack((zero, cy * sp * cr + sy * sr, -cy * sp * sr + sy * cr), dim=-1),
            torch.stack((zero, sy * sp * cr - cy * sr, -sy * sp * sr - cy * cr), dim=-1),
            torch.stack((zero, cp * cr, -cp * sr), dim=-1),
        ),
        dim=-2,
    )
    dpitch = torch.stack(
        (
            torch.stack((-cy * sp, cy * cp * sr, cy * cp * cr), dim=-1),
            torch.stack((-sy * sp, sy * cp * sr, sy * cp * cr), dim=-1),
            torch.stack((-cp, -sp * sr, -sp * cr), dim=-1),
        ),
        dim=-2,
    )
    dyaw = torch.stack(
        (
            torch.stack((-sy * cp, -sy * sp * sr - cy * cr, -sy * sp * cr + cy * sr), dim=-1),
            torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), dim=-1),
            torch.stack((zero, zero, zero), dim=-1),
        ),
        dim=-2,
    )
    return torch.stack((droll, dpitch, dyaw), dim=1)


def complete_root_point_jacobian(root_rpy_w: Tensor, point_body: Tensor) -> Tensor:
    """Return point Jacobians with respect to root translation and XYZ RPY."""
    rpy = torch.as_tensor(root_rpy_w)
    point = torch.as_tensor(point_body, dtype=rpy.dtype, device=rpy.device)
    if rpy.ndim != 2 or rpy.shape[-1] != 3 or point.ndim != 3 or point.shape[0] != rpy.shape[0]:
        raise ValueError("root_rpy_w and point_body must have shapes [B,3] and [B,N,3]")
    batch, points = int(point.shape[0]), int(point.shape[1])
    output = torch.zeros(batch, points, 3, 6, dtype=point.dtype, device=point.device)
    output[..., :3] = torch.eye(3, dtype=point.dtype, device=point.device).view(1, 1, 3, 3)
    output[..., 3:6] = torch.einsum("bqij,bkj->bkiq", _rpy_rotation_derivatives(rpy), point)
    return output


def _validate_inputs(root_pos_w: Tensor, root_rpy_w: Tensor, joint_pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    root_pos = torch.as_tensor(root_pos_w)
    root_rpy = torch.as_tensor(root_rpy_w, dtype=root_pos.dtype, device=root_pos.device)
    joint = torch.as_tensor(joint_pos, dtype=root_pos.dtype, device=root_pos.device)
    if root_pos.ndim != 2 or int(root_pos.shape[-1]) != 3:
        raise ValueError("root_pos_w must have shape [B,3]")
    if root_rpy.shape != root_pos.shape:
        raise ValueError("root_rpy_w must match root_pos_w")
    if joint.ndim != 2 or tuple(joint.shape) != (int(root_pos.shape[0]), 12):
        raise ValueError("joint_pos must have shape [B,12]")
    return root_pos, root_rpy, joint


def _leg_points_body(joint_pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    batch = int(joint_pos.shape[0])
    angles = joint_pos.reshape(batch, 4, 3)
    abad = angles[..., 0]
    thigh_angle = angles[..., 1]
    calf_angle = angles[..., 2]
    side = constant_like(joint_pos, "leg_side_signs", LEG_SIDE_SIGNS).view(1, 4)
    lateral = HIP_OFFSET_Y * side
    thigh = THIGH_LENGTH
    calf = CALF_LENGTH
    knee_x = -thigh * torch.sin(thigh_angle)
    knee_z = -thigh * torch.cos(thigh_angle)
    calf_absolute = thigh_angle + calf_angle
    foot_x = knee_x - calf * torch.sin(calf_absolute)
    foot_z = knee_z - calf * torch.cos(calf_absolute)
    cosine = torch.cos(abad)
    sine = torch.sin(abad)
    hip = constant_like(joint_pos, "hip_offsets", HIP_OFFSETS).view(1, 4, 3).expand(batch, -1, -1)
    upper_body = torch.stack(
        (
            hip[..., 0],
            hip[..., 1] + cosine * lateral,
            hip[..., 2] + sine * lateral,
        ),
        dim=-1,
    )
    knee_body = torch.stack(
        (
            hip[..., 0] + knee_x,
            hip[..., 1] + cosine * lateral - sine * knee_z,
            hip[..., 2] + sine * lateral + cosine * knee_z,
        ),
        dim=-1,
    )
    foot_body = torch.stack(
        (
            hip[..., 0] + foot_x,
            hip[..., 1] + cosine * lateral - sine * foot_z,
            hip[..., 2] + sine * lateral + cosine * foot_z,
        ),
        dim=-1,
    )
    return upper_body, knee_body, foot_body


def _body_collision_samples(dtype: torch.dtype, device: torch.device) -> Tensor:
    x = 0.32
    y = 0.09
    z_bottom = -0.08
    reference = torch.empty((), dtype=dtype, device=device)
    return constant_like(
        reference,
        "body_collision_samples",
        (
            (x, y, z_bottom),
            (x, -y, z_bottom),
            (-x, y, z_bottom),
            (-x, -y, z_bottom),
            (0.0, y, z_bottom),
            (0.0, -y, z_bottom),
            (x, 0.0, z_bottom),
            (-x, 0.0, z_bottom),
            (0.0, 0.0, z_bottom),
        ),
    )


def go2_fk(root_pos_w: Tensor, root_rpy_w: Tensor, joint_pos: Tensor) -> Go2Geometry:
    """Compute planner-order foot, knee, shank, and body samples in world coordinates."""
    root_input = torch.as_tensor(root_pos_w)
    root_rpy_input = torch.as_tensor(root_rpy_w, dtype=root_input.dtype, device=root_input.device)
    joint_input = torch.as_tensor(joint_pos, dtype=root_input.dtype, device=root_input.device)
    if root_input.ndim < 2 or root_input.shape[-1] != 3:
        raise ValueError("root_pos_w must have shape [...,3]")
    if root_rpy_input.shape != root_input.shape:
        raise ValueError("root_rpy_w must match root_pos_w")
    if joint_input.shape != root_input.shape[:-1] + (12,):
        raise ValueError("joint_pos must have shape [...,12]")
    leading_shape = root_input.shape[:-1]
    root_pos = root_input.reshape(-1, 3)
    root_rpy = root_rpy_input.reshape(-1, 3)
    joint = joint_input.reshape(-1, 12)
    upper_body, knee_body, foot_body = _leg_points_body(joint)
    alpha = constant_like(joint, "shank_sample_alpha", (0.25, 0.5, 0.75)).view(1, 1, 3, 1)
    shank_body = knee_body.unsqueeze(2) * (1.0 - alpha) + foot_body.unsqueeze(2) * alpha
    thigh_body = upper_body.unsqueeze(2) * (1.0 - alpha) + knee_body.unsqueeze(2) * alpha
    body_samples = _body_collision_samples(joint.dtype, joint.device).unsqueeze(0).expand(joint.shape[0], -1, -1)
    rotation = rpy_to_rotation_matrix(root_rpy)
    knee_world = torch.einsum("bij,bkj->bki", rotation, knee_body) + root_pos.unsqueeze(1)
    foot_world = torch.einsum("bij,bkj->bki", rotation, foot_body) + root_pos.unsqueeze(1)
    shank_world = torch.einsum("bij,bkqj->bkqi", rotation, shank_body) + root_pos[:, None, None, :]
    thigh_world = torch.einsum("bij,bkqj->bkqi", rotation, thigh_body) + root_pos[:, None, None, :]
    body_world = torch.einsum("bij,bkj->bki", rotation, body_samples) + root_pos.unsqueeze(1)
    return Go2Geometry(
        foot_pos_w=foot_world.reshape(*leading_shape, 4, 3),
        knee_pos_w=knee_world.reshape(*leading_shape, 4, 3),
        shank_samples_w=shank_world.reshape(*leading_shape, 4, 3, 3),
        thigh_samples_w=thigh_world.reshape(*leading_shape, 4, 3, 3),
        body_samples_w=body_world.reshape(*leading_shape, body_world.shape[-2], 3),
    )


def go2_collision_geometry(
    root_pos_w: Tensor,
    root_rpy_w: Tensor,
    joint_pos: Tensor,
) -> Go2CollisionGeometry:
    """Return world-coordinate sphere, capsule, sole, and base OBB primitives."""
    root_input = torch.as_tensor(root_pos_w)
    root_rpy_input = torch.as_tensor(
        root_rpy_w, dtype=root_input.dtype, device=root_input.device
    )
    joint_input = torch.as_tensor(
        joint_pos, dtype=root_input.dtype, device=root_input.device
    )
    if root_input.ndim < 2 or root_input.shape[-1] != 3:
        raise ValueError("root_pos_w must have shape [...,3]")
    if root_rpy_input.shape != root_input.shape:
        raise ValueError("root_rpy_w must match root_pos_w")
    if joint_input.shape != root_input.shape[:-1] + (12,):
        raise ValueError("joint_pos must have shape [...,12]")

    leading_shape = root_input.shape[:-1]
    root_pos = root_input.reshape(-1, 3)
    root_rpy = root_rpy_input.reshape(-1, 3)
    joint = joint_input.reshape(-1, 12)
    upper_body, knee_body, foot_body = _leg_points_body(joint)
    rotation = rpy_to_rotation_matrix(root_rpy)

    def to_world(points_body: Tensor) -> Tensor:
        return torch.einsum("bij,b...j->b...i", rotation, points_body) + root_pos.view(
            root_pos.shape[0], *((1,) * (points_body.ndim - 2)), 3
        )

    foot_world = to_world(foot_body)
    knee_world = to_world(knee_body)
    upper_world = to_world(upper_body)
    sole_offsets = constant_like(
        joint,
        "go2_sole_corner_offsets",
        (
            (SOLE_HALF_EXTENTS[0], SOLE_HALF_EXTENTS[1], -FOOT_RADIUS),
            (SOLE_HALF_EXTENTS[0], -SOLE_HALF_EXTENTS[1], -FOOT_RADIUS),
            (-SOLE_HALF_EXTENTS[0], SOLE_HALF_EXTENTS[1], -FOOT_RADIUS),
            (-SOLE_HALF_EXTENTS[0], -SOLE_HALF_EXTENTS[1], -FOOT_RADIUS),
        ),
    ).view(1, 1, 4, 3)
    sole_world = to_world(foot_body.unsqueeze(2) + sole_offsets)

    half_extents = constant_like(
        joint, "go2_base_half_extents", BASE_HALF_EXTENTS
    ).view(1, 3).expand(root_pos.shape[0], -1)
    base_corners_body = constant_like(
        joint,
        "go2_base_corners",
        tuple(
            (sx * BASE_HALF_EXTENTS[0], sy * BASE_HALF_EXTENTS[1], sz * BASE_HALF_EXTENTS[2])
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ),
    ).view(1, 8, 3)
    base_bottom_body = constant_like(
        joint,
        "go2_base_bottom_samples",
        tuple(
            (sx * BASE_HALF_EXTENTS[0], sy * BASE_HALF_EXTENTS[1], -BASE_HALF_EXTENTS[2])
            for sx in (-1.0, -0.5, 0.0, 0.5, 1.0)
            for sy in (-1.0, 0.0, 1.0)
        ),
    ).view(1, 15, 3)
    base_corners_world = to_world(base_corners_body)
    base_bottom_world = to_world(base_bottom_body)
    calf_endpoints = torch.stack((knee_world, foot_world), dim=-2)
    thigh_endpoints = torch.stack((upper_world, knee_world), dim=-2)

    return Go2CollisionGeometry(
        foot_center_w=foot_world.reshape(*leading_shape, 4, 3),
        sole_corners_w=sole_world.reshape(*leading_shape, 4, 4, 3),
        knee_center_w=knee_world.reshape(*leading_shape, 4, 3),
        calf_endpoints_w=calf_endpoints.reshape(*leading_shape, 4, 2, 3),
        thigh_endpoints_w=thigh_endpoints.reshape(*leading_shape, 4, 2, 3),
        base_center_w=root_input,
        base_rotation_w=rotation.reshape(*leading_shape, 3, 3),
        base_half_extents=half_extents.reshape(*leading_shape, 3),
        base_corners_w=base_corners_world.reshape(*leading_shape, 8, 3),
        base_bottom_samples_w=base_bottom_world.reshape(*leading_shape, 15, 3),
    )


def go2_selected_leg_collision_geometry(
    root_pos_w: Tensor,
    root_rpy_w: Tensor,
    joint_pos_leg: Tensor,
    leg_index: Tensor,
) -> Go2SelectedLegCollisionGeometry:
    """Return collision primitives for one selected leg per leading element."""
    root = torch.as_tensor(root_pos_w)
    rpy = torch.as_tensor(root_rpy_w, dtype=root.dtype, device=root.device)
    joint = torch.as_tensor(joint_pos_leg, dtype=root.dtype, device=root.device)
    index = torch.as_tensor(leg_index, dtype=torch.long, device=root.device)
    if root.ndim < 2 or root.shape[-1] != 3 or rpy.shape != root.shape:
        raise ValueError("root_pos_w and root_rpy_w must have matching [...,3] shapes")
    if joint.shape != root.shape or index.shape != root.shape[:-1]:
        raise ValueError("joint_pos_leg and leg_index must have shapes [...,3] and [...]")

    leading = root.shape[:-1]
    flat_root = root.reshape(-1, 3)
    flat_rpy = rpy.reshape(-1, 3)
    flat_joint = joint.reshape(-1, 3)
    flat_index = index.reshape(-1)
    side = torch.index_select(
        constant_like(flat_joint, "selected_leg_side_signs", LEG_SIDE_SIGNS),
        0,
        flat_index,
    )
    hip = torch.index_select(
        constant_like(flat_joint, "selected_leg_hip_offsets", HIP_OFFSETS),
        0,
        flat_index,
    )
    abad, thigh_angle, calf_angle = flat_joint.unbind(dim=-1)
    lateral = HIP_OFFSET_Y * side
    knee_x = -THIGH_LENGTH * torch.sin(thigh_angle)
    knee_z = -THIGH_LENGTH * torch.cos(thigh_angle)
    calf_absolute = thigh_angle + calf_angle
    foot_x = knee_x - CALF_LENGTH * torch.sin(calf_absolute)
    foot_z = knee_z - CALF_LENGTH * torch.cos(calf_absolute)
    cosine = torch.cos(abad)
    sine = torch.sin(abad)
    upper_body = torch.stack(
        (
            hip[:, 0],
            hip[:, 1] + cosine * lateral,
            hip[:, 2] + sine * lateral,
        ),
        dim=-1,
    )
    knee_body = torch.stack(
        (
            hip[:, 0] + knee_x,
            hip[:, 1] + cosine * lateral - sine * knee_z,
            hip[:, 2] + sine * lateral + cosine * knee_z,
        ),
        dim=-1,
    )
    foot_body = torch.stack(
        (
            hip[:, 0] + foot_x,
            hip[:, 1] + cosine * lateral - sine * foot_z,
            hip[:, 2] + sine * lateral + cosine * foot_z,
        ),
        dim=-1,
    )
    rotation = rpy_to_rotation_matrix(flat_rpy)

    def to_world(point_body: Tensor) -> Tensor:
        return torch.einsum("bij,bj->bi", rotation, point_body) + flat_root

    upper_w = to_world(upper_body)
    knee_w = to_world(knee_body)
    foot_w = to_world(foot_body)
    return Go2SelectedLegCollisionGeometry(
        foot_center_w=foot_w.reshape(*leading, 3),
        knee_center_w=knee_w.reshape(*leading, 3),
        calf_endpoints_w=torch.stack((knee_w, foot_w), dim=-2).reshape(
            *leading, 2, 3
        ),
        thigh_endpoints_w=torch.stack((upper_w, knee_w), dim=-2).reshape(
            *leading, 2, 3
        ),
    )


def go2_foot_pos(root_pos_w: Tensor, root_rpy_w: Tensor, joint_pos: Tensor) -> Tensor:
    """Compute only planner-order foot positions for nominal-shape references."""
    root_input = torch.as_tensor(root_pos_w)
    root_rpy_input = torch.as_tensor(root_rpy_w, dtype=root_input.dtype, device=root_input.device)
    joint_input = torch.as_tensor(joint_pos, dtype=root_input.dtype, device=root_input.device)
    if root_input.ndim < 2 or root_input.shape[-1] != 3:
        raise ValueError("root_pos_w must have shape [...,3]")
    if root_rpy_input.shape != root_input.shape:
        raise ValueError("root_rpy_w must match root_pos_w")
    if joint_input.shape != root_input.shape[:-1] + (12,):
        raise ValueError("joint_pos must have shape [...,12]")
    leading_shape = root_input.shape[:-1]
    root_pos = root_input.reshape(-1, 3)
    root_rpy = root_rpy_input.reshape(-1, 3)
    joint = joint_input.reshape(-1, 12)
    _, _, foot_body = _leg_points_body(joint)
    rotation = rpy_to_rotation_matrix(root_rpy)
    foot_world = torch.einsum("bij,bkj->bki", rotation, foot_body) + root_pos.unsqueeze(1)
    return foot_world.reshape(*leading_shape, 4, 3)


def foot_jacobian_leg(root_pos_w: Tensor, root_rpy_w: Tensor, joint_pos: Tensor) -> Tensor:
    """Return one dense 3x3 world-foot Jacobian for each independent leg."""
    root_pos, root_rpy, joint = _validate_inputs(root_pos_w, root_rpy_w, joint_pos)
    del root_pos
    batch = int(joint.shape[0])
    angles = joint.reshape(batch, 4, 3)
    abad = angles[..., 0]
    thigh_angle = angles[..., 1]
    calf_angle = angles[..., 2]
    side = constant_like(joint, "leg_side_signs", LEG_SIDE_SIGNS).view(1, 4)
    lateral = HIP_OFFSET_Y * side
    thigh = THIGH_LENGTH
    calf = CALF_LENGTH
    absolute = thigh_angle + calf_angle
    foot_z = -thigh * torch.cos(thigh_angle) - calf * torch.cos(absolute)
    cosine = torch.cos(abad)
    sine = torch.sin(abad)
    dfoot_x_thigh = -thigh * torch.cos(thigh_angle) - calf * torch.cos(absolute)
    dfoot_z_thigh = thigh * torch.sin(thigh_angle) + calf * torch.sin(absolute)
    dfoot_x_calf = -calf * torch.cos(absolute)
    dfoot_z_calf = calf * torch.sin(absolute)
    zero = torch.zeros_like(abad)
    derivative_abad = torch.stack(
        (
            zero,
            -sine * lateral - cosine * foot_z,
            cosine * lateral - sine * foot_z,
        ),
        dim=-1,
    )
    derivative_thigh = torch.stack(
        (dfoot_x_thigh, -sine * dfoot_z_thigh, cosine * dfoot_z_thigh),
        dim=-1,
    )
    derivative_calf = torch.stack(
        (dfoot_x_calf, -sine * dfoot_z_calf, cosine * dfoot_z_calf),
        dim=-1,
    )
    jacobian_body = torch.stack((derivative_abad, derivative_thigh, derivative_calf), dim=-1)
    rotation = rpy_to_rotation_matrix(root_rpy)
    return torch.einsum("bij,bkjq->bkiq", rotation, jacobian_body)


def foot_jacobian_joint(root_pos_w: Tensor, root_rpy_w: Tensor, joint_pos: Tensor) -> Tensor:
    """Return analytic world-foot Jacobians with respect to all 12 joint positions."""
    joint = torch.as_tensor(joint_pos)
    jacobian_world_leg = foot_jacobian_leg(root_pos_w, root_rpy_w, joint)
    selector = torch.eye(12, dtype=joint.dtype, device=joint.device).reshape(4, 3, 12)
    return torch.einsum("bkiq,kqr->bkir", jacobian_world_leg, selector)


def complete_foot_jacobian(root_pos_w: Tensor, root_rpy_w: Tensor, joint_pos: Tensor) -> Tensor:
    """Return world-foot Jacobians with respect to root position/RPY and all joints."""
    root_pos, root_rpy, joint = _validate_inputs(root_pos_w, root_rpy_w, joint_pos)
    del root_pos
    batch = int(joint.shape[0])
    foot_body = _leg_points_body(joint)[2]
    rotation_derivatives = _rpy_rotation_derivatives(root_rpy)
    root_rotation = torch.einsum("bqij,bkj->bkiq", rotation_derivatives, foot_body)
    output = torch.zeros(batch, 4, 3, 18, dtype=joint.dtype, device=joint.device)
    identity = torch.eye(3, dtype=joint.dtype, device=joint.device).view(1, 1, 3, 3)
    output[..., :3] = identity
    output[..., 3:6] = root_rotation
    local = foot_jacobian_leg(
        torch.zeros(batch, 3, dtype=joint.dtype, device=joint.device),
        root_rpy,
        joint,
    )
    for leg in range(4):
        output[:, leg, :, 6 + 3 * leg : 9 + 3 * leg] = local[:, leg]
    return output


def complete_knee_jacobian(root_pos_w: Tensor, root_rpy_w: Tensor, joint_pos: Tensor) -> Tensor:
    """Return world-knee Jacobians with respect to root position/RPY and all joints."""
    root_pos, root_rpy, joint = _validate_inputs(root_pos_w, root_rpy_w, joint_pos)
    del root_pos
    batch = int(joint.shape[0])
    knee_body = _leg_points_body(joint)[1]
    output = torch.zeros(batch, 4, 3, 18, dtype=joint.dtype, device=joint.device)
    output[..., :6] = complete_root_point_jacobian(root_rpy, knee_body)

    angles = joint.reshape(batch, 4, 3)
    abad = angles[..., 0]
    thigh_angle = angles[..., 1]
    side = constant_like(joint, "complete_knee_side_signs", LEG_SIDE_SIGNS).view(1, 4)
    lateral = HIP_OFFSET_Y * side
    cosine = torch.cos(abad)
    sine = torch.sin(abad)
    knee_z = -THIGH_LENGTH * torch.cos(thigh_angle)
    zero = torch.zeros_like(abad)
    knee_abad = torch.stack(
        (
            zero,
            -sine * lateral - cosine * knee_z,
            cosine * lateral - sine * knee_z,
        ),
        dim=-1,
    )
    dknee_x_thigh = -THIGH_LENGTH * torch.cos(thigh_angle)
    dknee_z_thigh = THIGH_LENGTH * torch.sin(thigh_angle)
    knee_thigh = torch.stack(
        (dknee_x_thigh, -sine * dknee_z_thigh, cosine * dknee_z_thigh),
        dim=-1,
    )
    local_body = torch.stack((knee_abad, knee_thigh, torch.zeros_like(knee_abad)), dim=-1)
    local_world = torch.einsum("bij,bkjq->bkiq", rpy_to_rotation_matrix(root_rpy), local_body)
    for leg in range(4):
        output[:, leg, :, 6 + 3 * leg : 9 + 3 * leg] = local_world[:, leg]
    return output


def complete_body_sample_jacobian(root_rpy_w: Tensor, body_samples_w: Tensor, root_pos_w: Tensor) -> Tensor:
    """Return body-sample Jacobians with zero joint columns."""
    root_rpy = torch.as_tensor(root_rpy_w)
    body_world = torch.as_tensor(body_samples_w, dtype=root_rpy.dtype, device=root_rpy.device)
    root_pos = torch.as_tensor(root_pos_w, dtype=root_rpy.dtype, device=root_rpy.device)
    rotation = rpy_to_rotation_matrix(root_rpy)
    body_local = torch.einsum(
        "bij,bkj->bki",
        rotation.transpose(-1, -2),
        body_world - root_pos.unsqueeze(1),
    )
    output = body_world.new_zeros((*body_world.shape[:-1], 3, 18))
    output[..., :6] = complete_root_point_jacobian(root_rpy, body_local)
    return output


def link_sample_jacobians(root_pos_w: Tensor, root_rpy_w: Tensor, joint_pos: Tensor) -> Go2LinkJacobians:
    """Return analytic local joint Jacobians for fixed calf and thigh samples."""
    root_pos, root_rpy, joint = _validate_inputs(root_pos_w, root_rpy_w, joint_pos)
    del root_pos
    batch = int(joint.shape[0])
    angles = joint.reshape(batch, 4, 3)
    abad = angles[..., 0]
    thigh_angle = angles[..., 1]
    side = constant_like(joint, "leg_side_signs", LEG_SIDE_SIGNS).view(1, 4)
    lateral = HIP_OFFSET_Y * side
    cosine = torch.cos(abad)
    sine = torch.sin(abad)
    knee_z = -THIGH_LENGTH * torch.cos(thigh_angle)
    dknee_x_thigh = -THIGH_LENGTH * torch.cos(thigh_angle)
    dknee_z_thigh = THIGH_LENGTH * torch.sin(thigh_angle)
    zero = torch.zeros_like(abad)
    upper_abad = torch.stack((zero, -sine * lateral, cosine * lateral), dim=-1)
    knee_abad = torch.stack(
        (
            zero,
            -sine * lateral - cosine * knee_z,
            cosine * lateral - sine * knee_z,
        ),
        dim=-1,
    )
    knee_thigh = torch.stack(
        (dknee_x_thigh, -sine * dknee_z_thigh, cosine * dknee_z_thigh),
        dim=-1,
    )
    upper_jacobian_body = torch.stack((upper_abad, torch.zeros_like(upper_abad), torch.zeros_like(upper_abad)), dim=-1)
    knee_jacobian_body = torch.stack((knee_abad, knee_thigh, torch.zeros_like(knee_abad)), dim=-1)
    rotation = rpy_to_rotation_matrix(root_rpy)
    upper_jacobian = torch.einsum("bij,bkjq->bkiq", rotation, upper_jacobian_body)
    knee_jacobian = torch.einsum("bij,bkjq->bkiq", rotation, knee_jacobian_body)
    foot_jacobian = foot_jacobian_leg(
        torch.zeros(batch, 3, dtype=joint.dtype, device=joint.device),
        root_rpy,
        joint,
    )
    alpha = constant_like(joint, "link_sample_alpha", (0.25, 0.5, 0.75)).view(1, 1, 3, 1, 1)
    thigh_samples = upper_jacobian.unsqueeze(2) * (1.0 - alpha) + knee_jacobian.unsqueeze(2) * alpha
    calf_samples = knee_jacobian.unsqueeze(2) * (1.0 - alpha) + foot_jacobian.unsqueeze(2) * alpha
    return Go2LinkJacobians(calf_samples=calf_samples, thigh_samples=thigh_samples)


def complete_link_sample_jacobians(
    root_pos_w: Tensor,
    root_rpy_w: Tensor,
    joint_pos: Tensor,
) -> Go2LinkJacobians:
    """Return calf/thigh sample Jacobians with respect to all 18 planner states."""
    root_pos, root_rpy, joint = _validate_inputs(root_pos_w, root_rpy_w, joint_pos)
    del root_pos
    batch = int(joint.shape[0])
    upper_body, knee_body, foot_body = _leg_points_body(joint)
    alpha = constant_like(joint, "complete_link_sample_alpha", (0.25, 0.5, 0.75)).view(
        1, 1, 3, 1
    )
    calf_body = knee_body.unsqueeze(2) * (1.0 - alpha) + foot_body.unsqueeze(2) * alpha
    thigh_body = upper_body.unsqueeze(2) * (1.0 - alpha) + knee_body.unsqueeze(2) * alpha
    local = link_sample_jacobians(
        torch.zeros(batch, 3, dtype=joint.dtype, device=joint.device),
        root_rpy,
        joint,
    )

    def assemble(points_body: Tensor, local_jacobian: Tensor) -> Tensor:
        root_jacobian = complete_root_point_jacobian(
            root_rpy,
            points_body.reshape(batch, 12, 3),
        ).reshape(batch, 4, 3, 3, 6)
        output = torch.zeros(batch, 4, 3, 3, 18, dtype=joint.dtype, device=joint.device)
        output[..., :6] = root_jacobian
        for leg in range(4):
            output[:, leg, :, :, 6 + 3 * leg : 9 + 3 * leg] = local_jacobian[:, leg]
        return output

    return Go2LinkJacobians(
        calf_samples=assemble(calf_body, local.calf_samples),
        thigh_samples=assemble(thigh_body, local.thigh_samples),
    )


__all__ = [
    "BASE_HALF_EXTENTS",
    "FOOT_RADIUS",
    "Go2CollisionGeometry",
    "Go2Geometry",
    "Go2LinkJacobians",
    "Go2SelectedLegCollisionGeometry",
    "CALF_LENGTH",
    "HIP_OFFSETS",
    "HIP_OFFSET_Y",
    "LEG_SIDE_SIGNS",
    "THIGH_LENGTH",
    "complete_foot_jacobian",
    "complete_knee_jacobian",
    "complete_body_sample_jacobian",
    "complete_link_sample_jacobians",
    "complete_root_point_jacobian",
    "foot_jacobian_joint",
    "foot_jacobian_leg",
    "go2_fk",
    "go2_collision_geometry",
    "go2_selected_leg_collision_geometry",
    "go2_foot_pos",
    "link_sample_jacobians",
    "rpy_to_rotation_matrix",
]
