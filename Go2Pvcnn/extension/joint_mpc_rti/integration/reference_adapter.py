"""Convert joint MPC RTI trajectories to the project reference-cache ABI."""

from __future__ import annotations

import torch

from extension.convention import euler_to_quat_batch
from extension.joint_mpc_rti.model.go2_kinematics import rpy_to_rotation_matrix
from extension.joint_mpc_rti.types import JointMpcRtiTrajectory
from extension.reference.cache import ReferenceTrajectoryCache


def trajectory_to_reference_cache(trajectory: JointMpcRtiTrajectory) -> ReferenceTrajectoryCache:
    state = torch.as_tensor(trajectory.state)
    foot_pos_w = torch.as_tensor(trajectory.foot_pos_w, dtype=state.dtype, device=state.device)
    contact = torch.as_tensor(trajectory.contact_state, dtype=torch.bool, device=state.device)
    if state.ndim != 3 or int(state.shape[-1]) != 18:
        raise ValueError("trajectory.state must have shape [B,H+1,18]")
    batch, nodes = int(state.shape[0]), int(state.shape[1])
    root_pos = state[..., :3]
    root_rpy = state[..., 3:6]
    root_quat = euler_to_quat_batch(root_rpy[..., 0], root_rpy[..., 1], root_rpy[..., 2])
    rotation = rpy_to_rotation_matrix(root_rpy.reshape(batch * nodes, 3)).reshape(batch, nodes, 3, 3)
    foot_delta_w = foot_pos_w - root_pos.unsqueeze(2)
    foot_pos_root = torch.einsum("btij,btkj->btki", rotation.transpose(-1, -2), foot_delta_w)
    phase = torch.arange(nodes, dtype=torch.long, device=state.device).unsqueeze(0).expand(batch, -1).contiguous()
    valid = torch.as_tensor(trajectory.valid, dtype=torch.bool, device=state.device)[:, None].expand(batch, nodes).contiguous()
    return ReferenceTrajectoryCache(
        root_pos_w=root_pos.contiguous(),
        root_quat_w=root_quat.contiguous(),
        joint_angles=state[..., 6:].contiguous(),
        foot_pos_w=foot_pos_w.contiguous(),
        foot_pos_root=foot_pos_root.contiguous(),
        contact_state=contact.contiguous(),
        planned_touchdown_w=foot_pos_w.contiguous(),
        phase_index=phase,
        valid_mask=valid,
    )


__all__ = ["trajectory_to_reference_cache"]
