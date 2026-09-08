"""Planner-free Go2 mixed-terrain environment for the AME baseline."""

from __future__ import annotations

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ame_baseline.ame_observations import downsampled_ame_scan
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
    TeacherElevationTrajectoryMpcSemanticSceneCfg,
)
from tracking.cross_large_complex_ppo_env_cfg import CrossLargeComplexPpoEnvCfg


@configclass
class AmeObservationsCfg:
    """AME map and state groups with all observation corruption disabled."""

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
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=None)
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=None)
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=None)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=None)
        velocity_commands = ObsTerm(
            func=isaac_mdp.generated_commands,
            params={"command_name": "base_velocity"},
            noise=None,
        )
        actions = ObsTerm(func=isaac_mdp.last_action, noise=None)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticStateCfg(PolicyStateCfg):
        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel, noise=None)

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
