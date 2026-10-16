"""Planner-free Go2 mixed-terrain environment for the AME baseline."""

from __future__ import annotations

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ame_baseline.ame_observations import downsampled_ame_scan
import tracking.mdp as tracking_mdp
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
    TeacherElevationTrajectoryMpcSemanticSceneCfg,
)
from tracking.cross_large_complex_ppo_env_cfg import CrossLargeComplexPpoEnvCfg


@configclass
class AmeObservationsCfg:
    """AME map with PPO-aligned continuous state observations."""

    @configclass
    class MapCfg(ObsGroup):
        elevation_semantic_map = ObsTerm(
            func=downsampled_ame_scan,
            params={"sensor_cfg": SceneEntityCfg("semantic_height_scanner"), "target_size": 16},
            noise=None,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PolicyStateCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        velocity_commands = ObsTerm(
            func=isaac_mdp.generated_commands,
            params={"command_name": "base_velocity"},
            noise=None,
        )
        actions = ObsTerm(func=isaac_mdp.last_action, noise=None)
        fixed_gait_phase = ObsTerm(func=tracking_mdp.fixed_gait_phase)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticStateCfg(PolicyStateCfg):
        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))

    policy_elevation_semantic_map: MapCfg = MapCfg()
    policy_state: PolicyStateCfg = PolicyStateCfg()
    critic_elevation_semantic_map: MapCfg = MapCfg()
    critic_state: CriticStateCfg = CriticStateCfg()


@configclass
class AmeCrossLargeComplexEnvCfg(CrossLargeComplexPpoEnvCfg):
    """Existing PPO task with only its observation representation replaced."""

    experiment_name: str = "cross_large_complex_ame"
    scene: TeacherElevationTrajectoryMpcSemanticSceneCfg = TeacherElevationTrajectoryMpcSemanticSceneCfg(
        num_envs=1024,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: AmeObservationsCfg = AmeObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "cross_large_complex_ame"
        self.curriculum.terrain_levels.params["excluded_terrain_names"] = (
            "flat_dense_small_obstacles",
            "flat",
        )
        self.scene.terrain.semantic_course_scale_profile_overrides = {
            "small": (0.12, 0.10),
            "large": (0.45, 0.55),
        }


@configclass
class AmeCrossLargeComplexEnvCfg_PLAY(AmeCrossLargeComplexEnvCfg):
    scene: TeacherElevationTrajectoryMpcSemanticSceneCfg = TeacherElevationTrajectoryMpcSemanticSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    def __post_init__(self):
        super().__post_init__()
        self.terminations.time_out = None
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


__all__ = ["AmeCrossLargeComplexEnvCfg", "AmeCrossLargeComplexEnvCfg_PLAY", "AmeObservationsCfg"]
