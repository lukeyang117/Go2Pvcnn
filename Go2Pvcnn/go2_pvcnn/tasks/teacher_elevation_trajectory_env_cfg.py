"""Teacher elevation trajectory env with batched GPU reference trajectory settings."""

from __future__ import annotations

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from extension.mdp.observations import downsampled_height_scan
from extension.mdp.rewards_reference import (
    reference_contact_reward,
    reference_foot_pos_reward,
    reference_joint_pos_reward,
    reference_root_pose_reward,
    reference_touchdown_reward,
)
from go2_pvcnn.tasks.teacher_elevation_env_cfg import TeacherElevationEnvCfg, TeacherElevationSceneCfg
from go2_pvcnn.tasks.teacher_semantic_env_cfg import COBBLESTONE_ROAD_CFG as SEMANTIC_TERRAIN_CFG
from go2_pvcnn.tasks.teacher_without_semantic_env_cfg import (
    RewardsCfg as BaseRewardsCfg,
    TeacherWithoutSemanticEnvCfg_PLAY,
)


@configclass
class TeacherElevationTrajectorySceneCfg(TeacherElevationSceneCfg):
    """Single high-resolution height scanner for planner + observation downsampling."""

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[1.5, 1.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class TeacherElevationTrajectoryObservationsCfg:
    """Use one high-res scanner and downsample it for actor/critic CNN stacks."""

    @configclass
    class PolicyElevationMapCfg(ObsGroup):
        elevation_map = ObsTerm(
            func=downsampled_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "target_size": 16},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PolicyStateCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        velocity_commands = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "base_velocity"})
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticElevationMapCfg(ObsGroup):
        elevation_map = ObsTerm(
            func=downsampled_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "target_size": 16},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticStateCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        velocity_commands = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "base_velocity"})
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy_elevation_map: PolicyElevationMapCfg = PolicyElevationMapCfg()
    policy_state: PolicyStateCfg = PolicyStateCfg()
    critic_elevation_map: CriticElevationMapCfg = CriticElevationMapCfg()
    critic_state: CriticStateCfg = CriticStateCfg()


@configclass
class TeacherElevationTrajectoryRewardsCfg(BaseRewardsCfg):
    reference_root_pose = RewTerm(func=reference_root_pose_reward, weight=0.2)
    reference_joint_pos = RewTerm(func=reference_joint_pos_reward, weight=0.15)
    reference_foot_pos = RewTerm(
        func=reference_foot_pos_reward,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
    )
    reference_contact = RewTerm(
        func=reference_contact_reward,
        weight=0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    reference_touchdown = RewTerm(
        func=reference_touchdown_reward,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
    )


@configclass
class TeacherElevationTrajectoryEnvCfg(TeacherElevationEnvCfg):
    scene: TeacherElevationTrajectorySceneCfg = TeacherElevationTrajectorySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: TeacherElevationTrajectoryObservationsCfg = TeacherElevationTrajectoryObservationsCfg()
    rewards: TeacherElevationTrajectoryRewardsCfg = TeacherElevationTrajectoryRewardsCfg()

    # Planner-only runtime: training must attach BatchedTrajectoryManager and must not
    # fall back to placeholder/reference-generator cache creation.
    planner_owned_reference_cache: bool = True
    use_batched_reference_trajectory: bool = True
    reference_trajectory_horizon: int = 50
    reference_replan_interval_steps: int = 250
    replan_velocity_scales: list[float] = [1.0, 0.8, 0.6]
    replan_yaw_biases: list[float] = [0.0, 0.15, -0.15]
    replan_vy_biases: list[float] = [0.0, 0.05, -0.05]
    replan_stop_speed: float = 0.05
    gait_name: str = "trot"
    step_freq: float = 2.0
    duty_factor: float = 0.6
    step_height: float = 0.08
    foothold_search_radius: float = 0.15
    foothold_search_step: float = 0.03
    max_step_down: float = float("inf")
    max_roughness: float = 0.5
    max_touchdown_xy_reach: float = 0.22
    # Optional: planner-owned timing/diagnostics (quiet by default).
    verbose_planner: bool = False
    verbose_planner_interval_steps: int = 250

    def __post_init__(self):
        super().__post_init__()
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class TeacherElevationTrajectoryEnvCfg_PLAY(TeacherWithoutSemanticEnvCfg_PLAY):
    scene: TeacherElevationTrajectorySceneCfg = TeacherElevationTrajectorySceneCfg(num_envs=32, env_spacing=2.5)
    observations: TeacherElevationTrajectoryObservationsCfg = TeacherElevationTrajectoryObservationsCfg()
    rewards: TeacherElevationTrajectoryRewardsCfg = TeacherElevationTrajectoryRewardsCfg()

    # Planner-only runtime: the play path also consumes the planner-owned manager/cache.
    planner_owned_reference_cache: bool = True
    use_batched_reference_trajectory: bool = True
    reference_trajectory_horizon: int = 50
    reference_replan_interval_steps: int = 250
    replan_velocity_scales: list[float] = [1.0, 0.8, 0.6]
    replan_yaw_biases: list[float] = [0.0, 0.15, -0.15]
    replan_vy_biases: list[float] = [0.0, 0.05, -0.05]
    replan_stop_speed: float = 0.05
    gait_name: str = "trot"
    step_freq: float = 2.0
    duty_factor: float = 0.6
    step_height: float = 0.08
    foothold_search_radius: float = 0.15
    foothold_search_step: float = 0.03
    max_step_down: float = float("inf")
    max_roughness: float = 0.5
    max_touchdown_xy_reach: float = 0.22
    verbose_planner: bool = False
    verbose_planner_interval_steps: int = 250

    def __post_init__(self):
        super().__post_init__()
        tg = self.scene.terrain.terrain_generator
        if tg is not None:
            tg.num_rows = SEMANTIC_TERRAIN_CFG.num_rows
            tg.num_cols = SEMANTIC_TERRAIN_CFG.num_cols
            tg.curriculum = SEMANTIC_TERRAIN_CFG.curriculum
        self.observations.policy_elevation_map.enable_corruption = False
        self.observations.policy_state.enable_corruption = False
        self.observations.critic_elevation_map.enable_corruption = False
        self.observations.critic_state.enable_corruption = False
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
