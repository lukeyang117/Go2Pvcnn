"""Teacher elevation experiment: elevation map (height map) + CNN + PPO.

Inherits from teacher_without_semantic. Adds lidar for height map only;
no cost map, no semantic objects.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.envs import mdp as isaac_mdp

import go2_pvcnn.mdp as mdp
from go2_pvcnn.sensor.lidar import SemanticLidarCfg, LivoxPatternCfg

from go2_pvcnn.tasks.teacher_without_semantic_env_cfg import (
    RobotSceneCfg,
    TeacherWithoutSemanticEnvCfg,
    TeacherWithoutSemanticEnvCfg_PLAY,
)


@configclass
class TeacherElevationSceneCfg(RobotSceneCfg):
    """Scene with lidar for elevation map only (terrain)."""

    lidar: SemanticLidarCfg = SemanticLidarCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=SemanticLidarCfg.OffsetCfg(
            pos=[0.3, 0.0, 0.2],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        ray_alignment="yaw",
        mesh_prim_paths=["/World/ground"],
        semantic_class_mapping={
            "terrain": ["ground", "wall", "floor", "plane"],
        },
        pattern_cfg=LivoxPatternCfg(
            sensor_type="mid360",
            use_simple_grid=True,
            horizontal_line_num=50,
            vertical_line_num=50,
            horizontal_fov_deg_min=-180.0,
            horizontal_fov_deg_max=180.0,
            vertical_fov_deg_min=-29.5,
            vertical_fov_deg_max=40.5,
        ),
        update_frequency=10.0,
        drift_range=(-0.0, 0.0),
        max_distance=1.5,
        min_range=0.1,
        return_pointcloud=True,
        pointcloud_in_world_frame=False,
        return_semantic_labels=False,
        enable_sensor_noise=False,
        debug_vis=False,
        return_height_map=True,
        height_map_size=(1.5, 1.5),
        height_map_resolution=0.1,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for teacher elevation (elevation map + state)."""

    @configclass
    class PolicyElevationMapCfg(ObsGroup):
        """Elevation map observations for CNN."""

        elevation_map = ObsTerm(
            func=mdp.elevation_map,
            params={"lidar_cfg": SceneEntityCfg("lidar")},
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
            func=mdp.elevation_map,
            params={"lidar_cfg": SceneEntityCfg("lidar")},
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


@configclass
class TeacherElevationEnvCfg_PLAY(TeacherWithoutSemanticEnvCfg_PLAY):
    """Play config for teacher elevation."""

    scene: TeacherElevationSceneCfg = TeacherElevationSceneCfg(num_envs=32, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy_elevation_map.enable_corruption = False
        self.observations.policy_state.enable_corruption = False
        self.observations.critic_elevation_map.enable_corruption = False
        self.observations.critic_state.enable_corruption = False
        print("[TeacherElevationEnvCfg_PLAY] Play mode (elevation map)")
