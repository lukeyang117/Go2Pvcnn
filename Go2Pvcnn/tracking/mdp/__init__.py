"""MDP terms for parallelism tracking."""

from tracking.mdp.curriculums import parallelism_velocity_curriculum
from tracking.mdp.observations import (
    parallelism_ref_joint_pos_rel_t,
    parallelism_ref_joint_vel_t,
    parallelism_ref_root_ang_vel_b_t,
    parallelism_ref_root_lin_vel_b_t,
)
from tracking.mdp.rewards import reference_joint_pos_reward, reference_joint_vel_reward
from tracking.mdp.terminations import (
    parallelism_ref_foot_z_too_far,
    parallelism_ref_joint_pos_too_far,
    parallelism_ref_projected_gravity_too_far,
    parallelism_ref_root_z_too_far,
)

__all__ = [
    "parallelism_ref_foot_z_too_far",
    "parallelism_ref_joint_pos_rel_t",
    "parallelism_ref_joint_pos_too_far",
    "parallelism_ref_joint_vel_t",
    "parallelism_ref_projected_gravity_too_far",
    "parallelism_ref_root_ang_vel_b_t",
    "parallelism_ref_root_lin_vel_b_t",
    "parallelism_ref_root_z_too_far",
    "parallelism_velocity_curriculum",
    "reference_joint_pos_reward",
    "reference_joint_vel_reward",
]
