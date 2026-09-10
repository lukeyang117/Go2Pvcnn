"""AME observation configuration combined with the existing Parallelism AMP env."""

from __future__ import annotations

from isaaclab.utils import configclass

from .ame_env_cfg import AmeCrossLargeComplexEnvCfg


AME_AMP_EXPERIMENT_NAME = "parallelism_tracking_cross_large_complex_ame_amp"
AME_AMP_ENV_ID = "Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-AME-Go2-v0"


@configclass
class AmeParallelismAmpCrossLargeComplexEnvCfg(AmeCrossLargeComplexEnvCfg):
    """AME map/state groups with the planner-owned AMP transition payload."""

    experiment_name: str = AME_AMP_EXPERIMENT_NAME
    planner_owned_reference_cache: bool = True
    parallelism_plan_batch_size: int = 1024
    amp_window_frames: int = 24
    amp_dt: float = 0.02
    reference_command_name: str = "base_velocity"

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = AME_AMP_EXPERIMENT_NAME
        self.planner_owned_reference_cache = True
        self.parallelism_plan_batch_size = 1024
        self.amp_window_frames = 24
        self.amp_dt = 0.02


__all__ = [
    "AME_AMP_ENV_ID",
    "AME_AMP_EXPERIMENT_NAME",
    "AmeParallelismAmpCrossLargeComplexEnvCfg",
]
