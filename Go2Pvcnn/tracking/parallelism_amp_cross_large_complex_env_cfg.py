"""Environment configuration for the isolated Parallelism AMP experiment."""

from isaaclab.utils import configclass

from tracking.parallelism_cross_large_complex_env_cfg import ParallelismTrackingCrossLargeComplexEnvCfg


@configclass
class ParallelismAmpCrossLargeComplexEnvCfg(ParallelismTrackingCrossLargeComplexEnvCfg):
    experiment_name: str = "parallelism_tracking_cross_large_complex_amp"
    amp_window_frames: int = 24
    amp_dt: float = 0.02

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parallelism_tracking_cross_large_complex_amp"

