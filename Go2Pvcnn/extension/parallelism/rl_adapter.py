from __future__ import annotations

from extension.parallelism.types import ParallelismReference, ParallelismTrajectory


def trajectory_to_reference(trajectory: ParallelismTrajectory) -> ParallelismReference:
    return ParallelismReference(
        root_pos_w=trajectory.root_pos_w,
        root_rpy_w=trajectory.root_rpy_w,
        joint_pos=trajectory.joint_pos,
        foot_pos_w=trajectory.foot_pos_w,
        contact_state=trajectory.contact_state,
        valid=trajectory.valid,
    )
