"""Flat RL tracking config driven by the parallelism planner."""

from __future__ import annotations

import math
from dataclasses import field
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import bind_visual_material, clone
from pxr import Usd, UsdGeom, UsdPhysics

from extension.mdp.observations import downsampled_elevation_semantic_scan
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
    SemanticObstacleCount,
    SemanticObstacleCurriculumCfg,
    TeacherElevationTrajectoryMpcSemanticEnvCfg,
    TeacherElevationTrajectoryMpcSemanticObservationsCfg,
    TeacherElevationTrajectoryMpcSemanticRewardsCfg,
    TeacherElevationTrajectoryMpcSemanticSceneCfg,
    _flat_small_avoidance_terrain_cfg,
)
import go2_pvcnn.mdp as go2_mdp
from go2_pvcnn.assets import UNITREE_GO2_CFG
import tracking.mdp as tracking_mdp


@clone
def _spawn_pale_reference_go2(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    """Spawn a kinematic, collision-free pale reference Go2."""

    prim = spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
    )
    stage = prim.GetStage()
    # The Go2 USD is instanceable. Detach visual and collision subtrees so
    # their per-instance USD properties can be changed safely.
    for child in Usd.PrimRange(prim):
        if child.IsInstance() and (
            "/visuals" in child.GetPath().pathString or "/collisions" in child.GetPath().pathString
        ):
            child.SetInstanceable(False)

    # The spawn config's collision_props cannot reach instanceable collision
    # prims. Disable every reference collider explicitly after detaching it.
    for child in Usd.PrimRange(prim):
        path = child.GetPath().pathString
        if "/collisions" not in path:
            continue
        if child.IsA(UsdGeom.Gprim):
            collision_api = UsdPhysics.CollisionAPI.Get(stage, child.GetPath())
            if not collision_api:
                collision_api = UsdPhysics.CollisionAPI.Apply(child)
            collision_api.CreateCollisionEnabledAttr().Set(False)

    material_path = "/World/Looks/ParallelismReferencePaleBlue"
    if not stage.GetPrimAtPath(material_path).IsValid():
        cfg.visual_material.func(material_path, cfg.visual_material)
    for child in Usd.PrimRange(prim):
        path = child.GetPath().pathString
        if "/visuals" in path and child.IsA(UsdGeom.Gprim):
            bind_visual_material(path, material_path)
    return prim


@configclass
class ParallelismTrackingCommandsCfg:
    base_velocity = go2_mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.48, 0.48),
        rel_standing_envs=0.05,
        debug_vis=True,
        ranges=go2_mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-0.05, 0.05),
            ang_vel_z=(-0.2, 0.2),
        ),
        limit_ranges=go2_mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class ParallelismTrackingObservationsCfg(TeacherElevationTrajectoryMpcSemanticObservationsCfg):
    @configclass
    class PolicyStateCfg(TeacherElevationTrajectoryMpcSemanticObservationsCfg.PolicyStateCfg):
        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        velocity_commands = None
        parallelism_ref_joint_pos = ObsTerm(func=tracking_mdp.parallelism_ref_joint_pos_rel_t)
        parallelism_ref_joint_vel = ObsTerm(func=tracking_mdp.parallelism_ref_joint_vel_t)
        parallelism_ref_root_lin_vel = ObsTerm(func=tracking_mdp.parallelism_ref_root_lin_vel_b_t)
        parallelism_ref_root_ang_vel = ObsTerm(func=tracking_mdp.parallelism_ref_root_ang_vel_b_t)

    @configclass
    class CriticStateCfg(TeacherElevationTrajectoryMpcSemanticObservationsCfg.CriticStateCfg):
        velocity_commands = None
        parallelism_ref_joint_pos = ObsTerm(func=tracking_mdp.parallelism_ref_joint_pos_rel_t)
        parallelism_ref_joint_vel = ObsTerm(func=tracking_mdp.parallelism_ref_joint_vel_t)
        parallelism_ref_root_lin_vel = ObsTerm(func=tracking_mdp.parallelism_ref_root_lin_vel_b_t)
        parallelism_ref_root_ang_vel = ObsTerm(func=tracking_mdp.parallelism_ref_root_ang_vel_b_t)

    @configclass
    class PolicyElevationSemanticMapCfg(TeacherElevationTrajectoryMpcSemanticObservationsCfg.PolicyElevationSemanticMapCfg):
        elevation_semantic_map = ObsTerm(
            func=downsampled_elevation_semantic_scan,
            params={"sensor_cfg": SceneEntityCfg("semantic_height_scanner"), "target_size": 16},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 2.0),
        )

    @configclass
    class CriticElevationSemanticMapCfg(TeacherElevationTrajectoryMpcSemanticObservationsCfg.CriticElevationSemanticMapCfg):
        elevation_semantic_map = ObsTerm(
            func=downsampled_elevation_semantic_scan,
            params={"sensor_cfg": SceneEntityCfg("semantic_height_scanner"), "target_size": 16},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 2.0),
        )

    policy_elevation_semantic_map: PolicyElevationSemanticMapCfg = PolicyElevationSemanticMapCfg()
    policy_state: PolicyStateCfg = PolicyStateCfg()
    critic_elevation_semantic_map: CriticElevationSemanticMapCfg = CriticElevationSemanticMapCfg()
    critic_state: CriticStateCfg = CriticStateCfg()


@configclass
class ParallelismTrackingRewardsCfg(TeacherElevationTrajectoryMpcSemanticRewardsCfg):
    track_lin_vel_xy = RewTerm(
        func=tracking_mdp.reference_root_lin_vel_reward,
        weight=1.5,
        params={"std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=tracking_mdp.reference_root_ang_vel_reward,
        weight=0.75,
        params={"std": math.sqrt(0.25)},
    )
    reference_joint_pos = RewTerm(
        func=tracking_mdp.reference_joint_pos_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "std": 0.7,
            "tracking_tolerance": 0.0,
        },
    )
    reference_joint_vel = RewTerm(
        func=tracking_mdp.reference_joint_vel_reward,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "std": 2.0,
            "tracking_tolerance": 0.0,
        },
    )


@configclass
class ParallelismTrackingTerminationsCfg:
    time_out = DoneTerm(func=go2_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=go2_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=go2_mdp.bad_orientation, params={"limit_angle": 0.8})
    parallelism_ref_root_z_too_far = DoneTerm(
        func=tracking_mdp.parallelism_ref_root_z_too_far,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.25},
    )
    parallelism_ref_projected_gravity_too_far = DoneTerm(
        func=tracking_mdp.parallelism_ref_projected_gravity_too_far,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.8, "z_only": False},
    )
    parallelism_ref_foot_z_too_far = DoneTerm(
        func=tracking_mdp.parallelism_ref_foot_z_too_far,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"), "threshold": 0.25},
    )
    parallelism_ref_joint_pos_too_far = DoneTerm(
        func=tracking_mdp.parallelism_ref_joint_pos_too_far,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "threshold": 0.8},
    )


@configclass
class ParallelismTrackingCurriculumCfg:
    parallelism_velocity = CurrTerm(
        func=tracking_mdp.parallelism_velocity_curriculum,
        params={
            "command_name": "base_velocity",
            "max_level": 10,
            "lin_vel_threshold": 0.25,
            "ang_vel_threshold": 0.35,
            "joint_mean_threshold": 0.20,
            "joint_max_threshold": 0.45,
        },
    )


@configclass
class ParallelismTrackingFlatEnvCfg(TeacherElevationTrajectoryMpcSemanticEnvCfg):
    """Flat Go2 RL tracking environment for parallelism references."""

    experiment_name: str = "parallelism_tracking_flat"
    scene: TeacherElevationTrajectoryMpcSemanticSceneCfg = TeacherElevationTrajectoryMpcSemanticSceneCfg(
        num_envs=1024,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: ParallelismTrackingObservationsCfg = ParallelismTrackingObservationsCfg()
    commands: ParallelismTrackingCommandsCfg = ParallelismTrackingCommandsCfg()
    rewards: ParallelismTrackingRewardsCfg = ParallelismTrackingRewardsCfg()
    terminations: ParallelismTrackingTerminationsCfg = ParallelismTrackingTerminationsCfg()
    curriculum: ParallelismTrackingCurriculumCfg = ParallelismTrackingCurriculumCfg()
    planner_owned_reference_cache: bool = False
    use_batched_reference_trajectory: bool = False
    parallelism_plan_batch_size: int = 64
    semantic_obstacle_curriculum: SemanticObstacleCurriculumCfg = field(
        default_factory=lambda: SemanticObstacleCurriculumCfg(
            enabled=False,
            plane_terrain_names=("flat",),
            plane_counts=(SemanticObstacleCount(small=0, large=0),),
            non_plane_counts=(SemanticObstacleCount(small=0, large=0),),
            center_safety_half_extent_m=(0.85,),
            min_spacing_clearance_m=(0.15,),
            tile_margin_m=(0.50,),
            collision_force_threshold=1.0,
        )
    )

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parallelism_tracking_flat"
        self.planner_owned_reference_cache = False
        self.use_batched_reference_trajectory = False
        self.scene.terrain.terrain_generator = _flat_small_avoidance_terrain_cfg()
        self.scene.terrain.semantic_obstacle_curriculum = self.semantic_obstacle_curriculum
        self.commands.base_velocity.resampling_time_range = (0.48, 0.48)
        self.commands.base_velocity.ranges = go2_mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-0.05, 0.05),
            ang_vel_z=(-0.2, 0.2),
        )
        self.commands.base_velocity.limit_ranges = go2_mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        )
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.3)
        self.rewards.reference_foot_pos = None
        self.rewards.semantic_contact_collision = None
        self.rewards.semantic_body_part_clearance = None
        self.rewards.semantic_foot_over_clearance = None
        self.scene.semantic_contact_small = None
        self.scene.semantic_contact_large = None
        self.events.push_robot = None
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation


@configclass
class ParallelismTrackingPlaySceneCfg(TeacherElevationTrajectoryMpcSemanticSceneCfg):
    """Play-only scene with a visual reference articulation outside policy physics."""

    reference_robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ParallelismReferenceGo2",
        spawn=UNITREE_GO2_CFG.spawn.replace(
            func=_spawn_pale_reference_go2,
            activate_contact_sensors=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            rigid_props=UNITREE_GO2_CFG.spawn.rigid_props.replace(disable_gravity=True),
            articulation_props=UNITREE_GO2_CFG.spawn.articulation_props.replace(
                enabled_self_collisions=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.35, 0.72, 1.0),
                emissive_color=(0.02, 0.05, 0.12),
                roughness=0.45,
                opacity=0.65,
            ),
        ),
    )


@configclass
class ParallelismTrackingFlatEnvCfg_PLAY(ParallelismTrackingFlatEnvCfg):
    scene: ParallelismTrackingPlaySceneCfg = ParallelismTrackingPlaySceneCfg(
        num_envs=32,
        env_spacing=2.5,
        replicate_physics=True,
    )

    def __post_init__(self):
        super().__post_init__()
        self.terminations.time_out = None
        self.curriculum.parallelism_velocity = None
        self.observations.policy_elevation_semantic_map.enable_corruption = False
        self.observations.policy_state.enable_corruption = False
        self.observations.critic_elevation_semantic_map.enable_corruption = False
        self.observations.critic_state.enable_corruption = False
