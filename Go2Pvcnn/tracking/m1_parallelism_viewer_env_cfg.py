"""Minimal Isaac Lab viewer environment for M1 Parallelism playback."""

from __future__ import annotations

from dataclasses import field

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.utils import configclass

from extension.semantic_course import SemanticCourseLayoutCfg, SemanticCourseTerrainImporter
from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg
from go2_pvcnn.assets.m1 import M1_CFG
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
    ISAAC_NUCLEUS_DIR,
    SEMANTIC_COURSE_LARGE_ROOT,
    SEMANTIC_COURSE_SMALL_ROOT,
)
from tracking.parallelism_cross_large_complex_env_cfg import _cross_large_complex_terrain_cfg
from go2_pvcnn.sensor.semantic_raycaster import SemanticGridRayCasterCfg
from isaaclab.terrains import TerrainImporterCfg


M1_SUPPORT_BODY_NAMES = (
    "FBL_FOOT_LINK",
    "FAR_FOOT_LINK",
    "RBL_FOOT_LINK",
    "RAR_FOOT_LINK",
)


@configclass
class M1ParallelismViewerSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=_cross_large_complex_terrain_cfg(),
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        debug_vis=False,
    )
    robot = M1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
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
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class M1ParallelismViewerActionsCfg:
    JointPositionAction = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class M1ParallelismViewerCommandsCfg:
    base_velocity = isaac_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(100.0, 100.0),
        rel_standing_envs=1.0,
        debug_vis=False,
        ranges=isaac_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
    )


@configclass
class M1ParallelismViewerEventsCfg:
    reset_base = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )
    reset_robot_joints = EventTerm(
        func=isaac_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    push_robot = None


@configclass
class M1ParallelismViewerTerminationsCfg:
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)


@configclass
class M1ParallelismViewerObservationsCfg:
    """Minimal observations required by ManagerBasedRLEnv during playback."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class M1ParallelismViewerEnvCfg(ManagerBasedRLEnvCfg):
    scene: M1ParallelismViewerSceneCfg = M1ParallelismViewerSceneCfg(num_envs=32, env_spacing=6.0, replicate_physics=True)
    observations: M1ParallelismViewerObservationsCfg = M1ParallelismViewerObservationsCfg()
    actions: M1ParallelismViewerActionsCfg = M1ParallelismViewerActionsCfg()
    commands: M1ParallelismViewerCommandsCfg = M1ParallelismViewerCommandsCfg()
    rewards = None
    events: M1ParallelismViewerEventsCfg = M1ParallelismViewerEventsCfg()
    terminations: M1ParallelismViewerTerminationsCfg = M1ParallelismViewerTerminationsCfg()
    curriculum = None
    planner_backend: str = "parallelism"
    robot_name: str = "m1"
    reference_height_scanner_name: str = "semantic_height_scanner"
    planner_owned_reference_cache: bool = False
    use_batched_reference_trajectory: bool = False
    semantic_obstacle_curriculum: SemanticObstacleCurriculumCfg = field(
        default_factory=lambda: SemanticObstacleCurriculumCfg(
            enabled=True,
            plane_terrain_names=("flat_dense_small_obstacles", "flat"),
            plane_counts=(SemanticObstacleCount(small=0, large=2),),
            non_plane_counts=(SemanticObstacleCount(small=5, large=2),),
            terrain_obstacle_count_overrides={
                "flat": SemanticObstacleCount(small=0, large=2),
                "flat_dense_small_obstacles": SemanticObstacleCount(small=40, large=0),
            },
            center_safety_half_extent_m=(0.25,),
            min_spacing_clearance_m=(0.08,),
            tile_margin_m=(0.50,),
            collision_force_threshold=1.0,
        )
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _cross_large_complex_terrain_cfg()
        self.scene.terrain.class_type = SemanticCourseTerrainImporter
        self.scene.terrain.semantic_obstacle_curriculum = self.semantic_obstacle_curriculum
        self.scene.terrain.semantic_course_layout_cfg = SemanticCourseLayoutCfg(
            tile_margin_m=0.50,
            center_safety_half_extent_m=0.25,
            center_safety_radius_m=0.30,
            min_spacing_clearance_m=0.08,
        )
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.episode_length_s = 100000.0
