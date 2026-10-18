"""M1 AME pure-PPO environment configuration."""

from __future__ import annotations

import torch

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ame_baseline.ame_env_cfg import AmeObservationsCfg
from extension.parallelism.m1_kinematics import M1_ROOT_Z_M
from go2_pvcnn.assets.m1 import M1_CFG, M1_USD_JOINT_NAMES
from go2_pvcnn.sensor.semantic_raycaster import SemanticGridRayCasterCfg
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
    SEMANTIC_COURSE_LARGE_ROOT,
    SEMANTIC_COURSE_SMALL_ROOT,
    TeacherElevationTrajectoryMpcSemanticSceneCfg,
    TeacherElevationTrajectoryMpcSemanticEventCfg,
)
from m1_ame.m1_ame_rewards import M1AmeRewardsCfg, M1AmeTerminationsCfg
from m1_ame.m1_ame_curriculum import M1AmeCurriculumCfg
from tracking.cross_large_complex_ppo_env_cfg import CrossLargeComplexPpoEnvCfg


M1_AME_JOINT_SELECTOR = tuple(M1_USD_JOINT_NAMES)


def m1_reset_root_state_uniform(env, env_ids, pose_range, velocity_range, asset_cfg=SceneEntityCfg("robot")):
    """Synchronize the semantic terrain origin before resetting the M1 root pose."""

    if env_ids is None:
        ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    else:
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    terrain = getattr(env.scene, "terrain", None)
    if terrain is not None and all(
        hasattr(terrain, name) for name in ("terrain_origins", "terrain_levels", "terrain_types")
    ):
        no_move = torch.zeros(ids.shape, dtype=torch.bool, device=ids.device)
        terrain.update_env_origins(ids, no_move, no_move)
    return isaac_mdp.reset_root_state_uniform(
        env,
        ids,
        pose_range=pose_range,
        velocity_range=velocity_range,
        asset_cfg=asset_cfg,
    )


@configclass
class M1AmeSceneCfg(TeacherElevationTrajectoryMpcSemanticSceneCfg):
    """Go2 AME scene with only the robot and scanner body path replaced."""

    robot: ArticulationCfg = M1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    semantic_height_scanner = SemanticGridRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BASE_LINK",
        offset=SemanticGridRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[1.5, 1.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground", SEMANTIC_COURSE_SMALL_ROOT, SEMANTIC_COURSE_LARGE_ROOT],
        mesh_semantic_ids={
            "/World/ground": 0,
            SEMANTIC_COURSE_SMALL_ROOT: 1,
            SEMANTIC_COURSE_LARGE_ROOT: 2,
        },
        height_scan_offset=0.5,
        max_update_envs_per_call=512,
    )


@configclass
class M1AmeActionsCfg:
    """Sixteen M1 joint-position actions in USD order."""

    JointPositionAction = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(M1_AME_JOINT_SELECTOR),
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class M1AmeEventsCfg(TeacherElevationTrajectoryMpcSemanticEventCfg):
    """Go2 event schedule with M1 base body selectors."""

    add_base_mass = EventTerm(
        func=isaac_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["BASE_LINK"]),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    sync_root_to_terrain = EventTerm(
        func=m1_reset_root_state_uniform,
        mode="startup",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    base_external_force_torque = EventTerm(
        func=isaac_mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["BASE_LINK"]),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=m1_reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class M1AmeObservationsCfg(AmeObservationsCfg):
    """The Go2 AME observation schema with explicit sixteen-joint selectors."""

    @configclass
    class PolicyStateCfg(AmeObservationsCfg.PolicyStateCfg):
        joint_pos = ObsTerm(
            func=isaac_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(M1_AME_JOINT_SELECTOR))},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=isaac_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(M1_AME_JOINT_SELECTOR))},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=isaac_mdp.last_action, noise=None)

    @configclass
    class CriticStateCfg(PolicyStateCfg):
        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel, noise=None)

    policy_state: PolicyStateCfg = PolicyStateCfg()
    critic_state: CriticStateCfg = CriticStateCfg()


@configclass
class M1AmeCrossLargeComplexEnvCfg(CrossLargeComplexPpoEnvCfg):
    """M1 AME pure-PPO task with the original Go2 terrain curriculum."""

    experiment_name: str = "m1_cross_large_complex_ame"
    scene: M1AmeSceneCfg = M1AmeSceneCfg(num_envs=1024, env_spacing=2.5, replicate_physics=True)
    observations: M1AmeObservationsCfg = M1AmeObservationsCfg()
    actions: M1AmeActionsCfg = M1AmeActionsCfg()
    events: M1AmeEventsCfg = M1AmeEventsCfg()
    rewards: M1AmeRewardsCfg = M1AmeRewardsCfg()
    terminations: M1AmeTerminationsCfg = M1AmeTerminationsCfg()
    curriculum: M1AmeCurriculumCfg = M1AmeCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "m1_cross_large_complex_ame"
        self.scene.robot.init_state.pos = (0.0, 0.0, M1_ROOT_Z_M)
        self.planner_owned_reference_cache = False
        self.use_batched_reference_trajectory = False
        self.commands.base_velocity.debug_vis = False


@configclass
class M1AmeCrossLargeComplexEnvCfg_PLAY(M1AmeCrossLargeComplexEnvCfg):
    """Single M1 environment for visual policy playback."""

    scene: M1AmeSceneCfg = M1AmeSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    def __post_init__(self):
        super().__post_init__()
        self.terminations.time_out = None
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.observations.policy_elevation_semantic_map.enable_corruption = False
        self.observations.policy_state.enable_corruption = False
        self.observations.critic_elevation_semantic_map.enable_corruption = False
        self.observations.critic_state.enable_corruption = False


__all__ = [
    "M1AmeActionsCfg",
    "M1AmeCrossLargeComplexEnvCfg",
    "M1AmeCrossLargeComplexEnvCfg_PLAY",
    "M1AmeEventsCfg",
    "M1AmeObservationsCfg",
    "M1AmeSceneCfg",
]
