"""Dense MPC backend scaffold for planner-owned reference cache runtime."""

from .config import (
    MpcDiagnosticsCfg,
    MpcLossesCfg,
    MpcPlannerCfg,
    MpcRuntimeCfg,
    planner_cfg_from_task_cfg,
    validate_mpc_config,
)
from .manager import MpcTrajectoryManager
from .planner import plan_segment

__all__ = [
    "MpcDiagnosticsCfg",
    "MpcLossesCfg",
    "MpcPlannerCfg",
    "MpcRuntimeCfg",
    "MpcTrajectoryManager",
    "plan_segment",
    "planner_cfg_from_task_cfg",
    "validate_mpc_config",
]
