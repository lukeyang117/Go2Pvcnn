"""Isaac Lab asset configuration for the ZJ_V3 / M1 wheeled quadruped."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from extension.parallelism.m1_kinematics import M1_DEFAULT_ASSET_JOINT_POS, M1_ROOT_Z_M


M1_USD_PATH = Path(__file__).resolve().parents[4] / "m1" / "ZJ_V3_URDF_V1_0" / "ZJ_V3_URDF_V1_0.usd"
M1_USD_JOINT_NAMES = (
    "FBL_ABAD_JOINT", "FBL_HIP_JOINT", "FBL_KNEE_JOINT", "FBL_FOOT_JOINT",
    "FAR_ABAD_JOINT", "FAR_HIP_JOINT", "FAR_KNEE_JOINT", "FAR_FOOT_JOINT",
    "RBL_ABAD_JOINT", "RBL_HIP_JOINT", "RBL_KNEE_JOINT", "RBL_FOOT_JOINT",
    "RAR_ABAD_JOINT", "RAR_HIP_JOINT", "RAR_KNEE_JOINT", "RAR_FOOT_JOINT",
)
M1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(M1_USD_PATH),
        activate_contact_sensors=True,
        visible=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, M1_ROOT_Z_M),
        joint_pos=dict(zip(M1_USD_JOINT_NAMES, M1_DEFAULT_ASSET_JOINT_POS)),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=25.0,
            damping=0.5,
        ),
    },
)
