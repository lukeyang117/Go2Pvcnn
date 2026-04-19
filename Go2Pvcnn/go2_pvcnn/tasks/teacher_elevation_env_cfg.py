"""Teacher elevation experiment: elevation map (height map) + CNN + PPO.

Inherits from teacher_without_semantic. Uses Isaac Lab velocity task height_scanner
(``isaaclab_tasks/.../locomotion/velocity/velocity_env_cfg.py``): same RayCasterCfg
layout with ``GridPatternCfg(resolution=0.1, size=[1.5, 1.5])`` (1.5 m footprint).
Observation: :func:`go2_pvcnn.mdp.elevation_map_height_scan` wrapping
``isaaclab.envs.mdp.height_scan``.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.envs import mdp as isaac_mdp

import go2_pvcnn.mdp as mdp

from go2_pvcnn.tasks.teacher_semantic_env_cfg import (
    COBBLESTONE_ROAD_CFG as SEMANTIC_TERRAIN_CFG,
    _TEACHER_OBJECTS_DIR,
)
from go2_pvcnn.tasks.teacher_without_semantic_env_cfg import (
    RobotSceneCfg,
    TeacherWithoutSemanticEnvCfg,
    TeacherWithoutSemanticEnvCfg_PLAY,
)


@configclass
class TeacherElevationSceneCfg(RobotSceneCfg):
    """Scene with official velocity-env height_scanner (RayCaster + grid), 1.5×1.5 m @ 0.1 m.

    ``height_scanner`` matches ``velocity_env_cfg.MySceneCfg.height_scanner`` except
    ``pattern_cfg.size`` is ``[1.5, 1.5]`` (same footprint as former LiDAR height map).
    """

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

    # Same structure as isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg.MySceneCfg
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.5, 1.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class ObservationsCfg:
    """Observation specifications for teacher elevation (elevation map + state)."""

    @configclass
    class PolicyElevationMapCfg(ObsGroup):
        """Elevation map observations for CNN."""

        elevation_map = ObsTerm(
            func=mdp.elevation_map_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PolicyStateCfg(ObsGroup):
        """State observations for MLP (aligned with teacher_semantic)."""

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
    class CriticElevationMapCfg(ObsGroup):
        """Elevation map observations for critic CNN."""

        elevation_map = ObsTerm(
            func=mdp.elevation_map_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticStateCfg(ObsGroup):
        """State observations for critic MLP (aligned with teacher_semantic)."""

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

    policy_elevation_map: PolicyElevationMapCfg = PolicyElevationMapCfg()
    policy_state: PolicyStateCfg = PolicyStateCfg()
    critic_elevation_map: CriticElevationMapCfg = CriticElevationMapCfg()
    critic_state: CriticStateCfg = CriticStateCfg()


@configclass
class TeacherElevationEnvCfg(TeacherWithoutSemanticEnvCfg):
    """Teacher elevation: elevation map + CNN + PPO, inherits teacher_without_semantic."""

    scene: TeacherElevationSceneCfg = TeacherElevationSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class TeacherElevationEnvCfg_PLAY(TeacherWithoutSemanticEnvCfg_PLAY):
    """Play config for teacher elevation."""

    scene: TeacherElevationSceneCfg = TeacherElevationSceneCfg(num_envs=32, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    
    def __post_init__(self):
        super().__post_init__()
        # Restore training terrain grid (parent PLAY uses 2×1 for fast debug)
        tg = self.scene.terrain.terrain_generator
        if tg is not None:
            tg.num_rows = SEMANTIC_TERRAIN_CFG.num_rows
            tg.num_cols = SEMANTIC_TERRAIN_CFG.num_cols
            tg.curriculum = SEMANTIC_TERRAIN_CFG.curriculum
        # Play: constant velocity command +x = 0.5 m/s (no lateral / yaw)
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        self.observations.policy_elevation_map.enable_corruption = False
        self.observations.policy_state.enable_corruption = False
        self.observations.critic_elevation_map.enable_corruption = False
        self.observations.critic_state.enable_corruption = False
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        print("[TeacherElevationEnvCfg_PLAY] Play mode (elevation map)")
