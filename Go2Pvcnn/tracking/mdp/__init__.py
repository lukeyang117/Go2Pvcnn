"""MDP terms for parallelism tracking."""

from tracking.mdp.curriculums import parallelism_velocity_curriculum
from tracking.mdp.distillation import (
    parallelism_distillation_context,
    terrain_imitation_context_from_metadata,
)
from tracking.mdp.observations import (
    parallelism_plan_valid,
    parallelism_ref_joint_pos_rel_t,
    parallelism_ref_joint_vel_t,
    parallelism_ref_root_pos_b_t,
    parallelism_ref_root_rot_b_t,
)
from tracking.mdp.rewards import (
    active_swing_foot_on_small_obstacle_reward,
    parallelism_geometry_collision_penalty,
    parallelism_obstacle_episode_metrics,
    reset_parallelism_obstacle_stats,
    reference_active_swing_foot_max_reward,
    reference_joint_pos_reward,
    reference_joint_vel_reward,
    reference_foot_pos_reward,
    reference_joint_max_reward,
    reference_root_pos_reward,
    reference_root_rot_reward,
)
from tracking.mdp.policy_geometry_rewards import policy_geometry_collision_penalty
from tracking.mdp.terminations import (
    parallelism_consecutive_standstill,
    parallelism_ref_foot_z_too_far,
    parallelism_ref_joint_pos_too_far,
    parallelism_ref_projected_gravity_too_far,
    parallelism_ref_root_z_too_far,
)

__all__ = [
    "parallelism_consecutive_standstill",
    "parallelism_distillation_context",
    "parallelism_plan_valid",
    "parallelism_ref_foot_z_too_far",
    "parallelism_ref_joint_pos_rel_t",
    "parallelism_ref_joint_pos_too_far",
    "parallelism_ref_joint_vel_t",
    "parallelism_ref_projected_gravity_too_far",
    "parallelism_ref_root_pos_b_t",
    "parallelism_ref_root_rot_b_t",
    "parallelism_ref_root_z_too_far",
    "parallelism_velocity_curriculum",
    "terrain_imitation_context_from_metadata",
    "active_swing_foot_on_small_obstacle_reward",
    "parallelism_geometry_collision_penalty",
    "policy_geometry_collision_penalty",
    "parallelism_obstacle_episode_metrics",
    "reset_parallelism_obstacle_stats",
    "reference_active_swing_foot_max_reward",
    "reference_joint_pos_reward",
    "reference_joint_vel_reward",
    "reference_foot_pos_reward",
    "reference_joint_max_reward",
    "reference_root_pos_reward",
    "reference_root_rot_reward",
]
