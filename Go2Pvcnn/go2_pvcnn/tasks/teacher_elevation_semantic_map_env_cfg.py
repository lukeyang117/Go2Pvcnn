"""Teacher elevation + semantic grid: Isaac Lab multi-mesh ray caster, dual-channel CNN + PPO.

Uses :class:`~go2_pvcnn.sensor.semantic_raycaster.semantic_ray_caster.SemanticGridRayCaster` (extends Isaac
Lab ``RayCaster``) with the same grid pattern / ``height_scan`` offset as native height scanners.
Static cuboids use ``/World/...`` prim paths (same namespace as ``/World/ground`` terrain).
Semantic ids: terrain=0, small cube=1, big cube=2.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.envs import mdp as isaac_mdp

import go2_pvcnn.mdp as mdp
from go2_pvcnn.sensor.semantic_raycaster import SemanticGridRayCasterCfg

from go2_pvcnn.tasks.teacher_semantic_env_cfg import (
    COBBLESTONE_ROAD_CFG as SEMANTIC_TERRAIN_CFG,
    _TEACHER_OBJECTS_DIR,
)
from go2_pvcnn.tasks.teacher_without_semantic_env_cfg import (
    RobotSceneCfg,
    TeacherWithoutSemanticEnvCfg,
    TeacherWithoutSemanticEnvCfg_PLAY,
)


# Shared world space (same pattern as terrain prim_path="/World/ground").
# Use AssetBaseCfg (scene ``_extras``), not RigidObjectCfg: a single /World prim has
# num_instances=1 while scene.reset passes env_ids 0..num_envs-1, which breaks RigidObject buffers.
_SEM_SMALL = "/World/semantic_map_small_cube"
_SEM_BIG = "/World/semantic_map_big_cube"


@configclass
class TeacherElevationSemanticMapSceneCfg(RobotSceneCfg):
    """Terrain (teacher_semantic style) + static cuboids + semantic grid ray caster."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SEMANTIC_TERRAIN_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=str(
                _TEACHER_OBJECTS_DIR
                / "Materials"
                / "TilesMarbleSpiderWhiteBrickBondHoned"
                / "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    static_obstacle_small: AssetBaseCfg = AssetBaseCfg(
        prim_path=_SEM_SMALL,
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 0.12, 0.22),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.4, 0.25, 0.14)),
        collision_group=-1,
    )

    static_obstacle_big: AssetBaseCfg = AssetBaseCfg(
        prim_path=_SEM_BIG,
        spawn=sim_utils.CuboidCfg(
            size=(0.45, 0.45, 0.55),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.2, -0.35, 0.32)),
        collision_group=-1,
    )

    semantic_height_scanner: SemanticGridRayCasterCfg = SemanticGridRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=SemanticGridRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.5, 1.5]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground", _SEM_SMALL, _SEM_BIG],
        mesh_semantic_ids={
            "/World/ground": 0,
            _SEM_SMALL: 1,
            _SEM_BIG: 2,
        },
        height_scan_offset=0.5,
    )


@configclass
class TeacherElevationSemanticMapObservationsCfg:
    """Dual-channel grid (elevation + semantic) for actor/critic CNN stacks."""

    @configclass
    class PolicyGridCfg(ObsGroup):
        elevation_semantic = ObsTerm(
            func=mdp.elevation_semantic_dual_map,
            params={"sensor_cfg": SceneEntityCfg("semantic_height_scanner")},
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
        velocity_commands = ObsTerm(
            func=isaac_mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticGridCfg(ObsGroup):
        elevation_semantic = ObsTerm(
            func=mdp.elevation_semantic_dual_map,
            params={"sensor_cfg": SceneEntityCfg("semantic_height_scanner")},
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
        velocity_commands = ObsTerm(
            func=isaac_mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy_elevation_semantic_map: PolicyGridCfg = PolicyGridCfg()
    policy_state: PolicyStateCfg = PolicyStateCfg()
    critic_elevation_semantic_map: CriticGridCfg = CriticGridCfg()
    critic_state: CriticStateCfg = CriticStateCfg()


@configclass
class TeacherElevationSemanticMapEnvCfg(TeacherWithoutSemanticEnvCfg):
    """Elevation + semantic grid teacher (Isaac native ray cast, no project LiDAR)."""

    scene: TeacherElevationSemanticMapSceneCfg = TeacherElevationSemanticMapSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: TeacherElevationSemanticMapObservationsCfg = TeacherElevationSemanticMapObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.semantic_height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class TeacherElevationSemanticMapEnvCfg_PLAY(TeacherWithoutSemanticEnvCfg_PLAY):
    """Play config for elevation + semantic grid."""

    scene: TeacherElevationSemanticMapSceneCfg = TeacherElevationSemanticMapSceneCfg(num_envs=32, env_spacing=2.5)
    observations: TeacherElevationSemanticMapObservationsCfg = TeacherElevationSemanticMapObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        tg = self.scene.terrain.terrain_generator
        if tg is not None:
            tg.num_rows = SEMANTIC_TERRAIN_CFG.num_rows
            tg.num_cols = SEMANTIC_TERRAIN_CFG.num_cols
            tg.curriculum = SEMANTIC_TERRAIN_CFG.curriculum
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        self.observations.policy_elevation_semantic_map.enable_corruption = False
        self.observations.policy_state.enable_corruption = False
        self.observations.critic_elevation_semantic_map.enable_corruption = False
        self.observations.critic_state.enable_corruption = False
        self.scene.semantic_height_scanner.update_period = self.decimation * self.sim.dt
        print("[TeacherElevationSemanticMapEnvCfg_PLAY] Play mode (elevation + semantic grid)")
