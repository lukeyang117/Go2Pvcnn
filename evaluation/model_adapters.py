"""Checkpoint and task metadata shared by benchmark launchers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    experiment: str
    task_id: str
    checkpoint: Path
    observation_kind: str
    requires_reference_manager: bool


_MODEL_TABLE = {
    "amp": ("parallelism_tracking_cross_large_complex_amp", "Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-AMP-v0", "deployment", True),
    "distillation": ("parallelism_tracking_cross_large_complex_distillation", "Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-Distillation-v0", "deployment", True),
    "ppo": ("cross_large_complex_ppo", "Isaac-Go2-Cross-Large-Complex-PPO-v0", "deployment", True),
    "teacher": ("parallelism_tracking_cross_large_complex", "Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-v0", "teacher", True),
}


def resolve_model_spec(kind: str, checkpoint: str | Path) -> ModelSpec:
    key = str(kind).lower()
    if key not in _MODEL_TABLE:
        raise ValueError(f"unknown experiment type: {kind}")
    path = Path(checkpoint).expanduser()
    if not path.exists() or not path.is_file():
        raise ValueError("checkpoint must be an explicit checkpoint file")
    experiment, task_id, observation_kind, requires_reference = _MODEL_TABLE[key]
    return ModelSpec(key, experiment, task_id, path.resolve(), observation_kind, requires_reference)


def build_play_mapping() -> dict[str, tuple[object, str]]:
    """Return the same task IDs as play.py, importing Isaac classes lazily."""

    from tracking.cross_large_complex_ppo_env_cfg import CrossLargeComplexPpoEnvCfg_PLAY
    from tracking.parallelism_amp_cross_large_complex_env_cfg import ParallelismAmpCrossLargeComplexEnvCfg
    from tracking.parallelism_cross_large_complex_distillation_env_cfg import ParallelismTrackingCrossLargeComplexDistillationEnvCfg
    from tracking.parallelism_cross_large_complex_env_cfg import ParallelismTrackingCrossLargeComplexEnvCfg

    return {
        "amp": (ParallelismAmpCrossLargeComplexEnvCfg, _MODEL_TABLE["amp"][1]),
        "distillation": (ParallelismTrackingCrossLargeComplexDistillationEnvCfg, _MODEL_TABLE["distillation"][1]),
        "ppo": (CrossLargeComplexPpoEnvCfg_PLAY, _MODEL_TABLE["ppo"][1]),
        "teacher": (ParallelismTrackingCrossLargeComplexEnvCfg, _MODEL_TABLE["teacher"][1]),
    }
