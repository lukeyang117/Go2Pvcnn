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
    adapter_kind: str


@dataclass
class BenchmarkRuntime:
    env: object
    base_env: object
    wrapped_env: object
    policy: object
    observations: object


_MODEL_TABLE = {
    "amp": ("parallelism_tracking_cross_large_complex_amp", "Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-AMP-v0", "deployment", True, "default"),
    "distillation": ("parallelism_tracking_cross_large_complex_distillation", "Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-Distillation-v0", "deployment", True, "default"),
    "ppo": ("cross_large_complex_ppo", "Isaac-Go2-Cross-Large-Complex-PPO-v0", "deployment", True, "default"),
    "teacher": ("parallelism_tracking_cross_large_complex", "Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-v0", "teacher", True, "default"),
    "ame": ("cross_large_complex_ame", "Isaac-Go2-Cross-Large-Complex-PPO-v0", "ame", True, "ame"),
    "ame_amp": ("parallelism_tracking_cross_large_complex_ame_amp", "Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-AME-Go2-v0", "ame", True, "ame_amp"),
    "semloco": ("cross_large_complex_semloco", "Isaac-Go2-Cross-Large-Complex-SemLoco-v0", "deployment", True, "default"),
}


def resolve_model_spec(kind: str, checkpoint: str | Path) -> ModelSpec:
    key = str(kind).lower()
    if key not in _MODEL_TABLE:
        raise ValueError(f"unknown experiment type: {kind}")
    path = Path(checkpoint).expanduser()
    if not path.exists() or not path.is_file():
        raise ValueError("checkpoint must be an explicit checkpoint file")
    experiment, task_id, observation_kind, requires_reference, adapter_kind = _MODEL_TABLE[key]
    return ModelSpec(key, experiment, task_id, path.resolve(), observation_kind, requires_reference, adapter_kind)


def build_play_mapping() -> dict[str, tuple[object, str]]:
    """Return the same task IDs as play.py, importing Isaac classes lazily."""

    from tracking.cross_large_complex_ppo_env_cfg import CrossLargeComplexPpoEnvCfg_PLAY
    from tracking.parallelism_amp_cross_large_complex_env_cfg import ParallelismAmpCrossLargeComplexEnvCfg
    from tracking.parallelism_cross_large_complex_distillation_env_cfg import ParallelismTrackingCrossLargeComplexDistillationEnvCfg
    from tracking.parallelism_cross_large_complex_env_cfg import ParallelismTrackingCrossLargeComplexEnvCfg
    from ame_baseline.ame_amp_env_cfg import AmeParallelismAmpCrossLargeComplexEnvCfg
    from ame_baseline.ame_env_cfg import AmeCrossLargeComplexEnvCfg_PLAY
    from semloco.semloco_env_cfg import SemlocoCrossLargeComplexEnvCfg_PLAY

    return {
        "amp": (ParallelismAmpCrossLargeComplexEnvCfg, _MODEL_TABLE["amp"][1]),
        "distillation": (ParallelismTrackingCrossLargeComplexDistillationEnvCfg, _MODEL_TABLE["distillation"][1]),
        "ppo": (CrossLargeComplexPpoEnvCfg_PLAY, _MODEL_TABLE["ppo"][1]),
        "teacher": (ParallelismTrackingCrossLargeComplexEnvCfg, _MODEL_TABLE["teacher"][1]),
        "ame": (AmeCrossLargeComplexEnvCfg_PLAY, _MODEL_TABLE["ame"][1]),
        "ame_amp": (AmeParallelismAmpCrossLargeComplexEnvCfg, _MODEL_TABLE["ame_amp"][1]),
        "semloco": (SemlocoCrossLargeComplexEnvCfg_PLAY, _MODEL_TABLE["semloco"][1]),
    }


class DefaultBenchmarkAdapter:
    """Build the existing RSL-RL runtime used by the original four models."""

    preserve_planner_owned_reference_cache = False

    def create_env_cfg(self, spec: ModelSpec):
        cfg_cls, _ = build_play_mapping()[spec.name]
        return cfg_cls()

    def create_runtime(self, spec: ModelSpec, env_cfg, gym_module) -> BenchmarkRuntime:
        from agent import get_train_cfg
        from rsl_rl.env import VecEnv
        from rsl_rl.runners import OnPolicyRunner
        from tensordict import TensorDict

        from Go2Pvcnn.scripts.play import _make_env_wrapper

        env = gym_module.make(spec.task_id, cfg=env_cfg, render_mode=None)
        base_env = env.unwrapped
        wrapped_env = _make_env_wrapper(
            base_env,
            gym_module=gym_module,
            vec_env_cls=VecEnv,
            tensor_dict_cls=TensorDict,
            clip_actions=100.0,
        )
        runner = OnPolicyRunner(
            wrapped_env,
            get_train_cfg(spec.experiment),
            log_dir=None,
            device=env_cfg.sim.device,
        )
        runner.load(str(spec.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=wrapped_env.device)
        observations, _ = wrapped_env.consume_initial_observations()
        return BenchmarkRuntime(env, base_env, wrapped_env, policy, observations)


class AmeBenchmarkAdapter:
    """Build a deterministic AME runtime with its map/state wrapper."""

    preserve_planner_owned_reference_cache = False

    def create_env_cfg(self, spec: ModelSpec):
        del spec
        from ame_baseline.ame_env_cfg import AmeCrossLargeComplexEnvCfg_PLAY

        return AmeCrossLargeComplexEnvCfg_PLAY()

    def create_runtime(self, spec: ModelSpec, env_cfg, gym_module) -> BenchmarkRuntime:
        import copy

        from ame_baseline.ame_env_wrapper import AmeRslRlEnvWrapper
        from ame_baseline.ame_runner import AmeOnPolicyRunner
        from ame_baseline.ame_train_cfg import get_ame_train_cfg

        env = gym_module.make(spec.task_id, cfg=env_cfg, render_mode=None)
        wrapped_env = AmeRslRlEnvWrapper(env, clip_actions=100.0)
        runner = AmeOnPolicyRunner(
            wrapped_env,
            copy.deepcopy(get_ame_train_cfg()),
            log_dir=None,
            device=env_cfg.sim.device,
        )
        runner.load(str(spec.checkpoint), load_optimizer=False, keep_std=True)
        policy = runner.get_inference_policy(device=wrapped_env.device)
        observations, _ = wrapped_env.get_observations()
        return BenchmarkRuntime(env, env.unwrapped, wrapped_env, policy, observations)


class AmeAmpBenchmarkAdapter(AmeBenchmarkAdapter):
    """Build AME-AMP and enforce loading a complete AMP checkpoint."""

    preserve_planner_owned_reference_cache = True

    @staticmethod
    def require_full_amp_resume(mode: str) -> None:
        if mode != "full_amp_resume":
            raise ValueError("benchmark requires a complete AME-AMP checkpoint")

    def create_env_cfg(self, spec: ModelSpec):
        del spec
        from ame_baseline.ame_amp_env_cfg import AmeParallelismAmpCrossLargeComplexEnvCfg

        return AmeParallelismAmpCrossLargeComplexEnvCfg()

    @staticmethod
    def _register_runtime_env(gym_module, env_id: str, env_cfg_cls) -> None:
        if env_id in gym_module.registry:
            return
        gym_module.register(
            id=env_id,
            entry_point="tracking.amp_env:ParallelismAmpEnv",
            kwargs={"env_cfg_entry_point": env_cfg_cls, "rsl_rl_cfg_entry_point": None},
            disable_env_checker=True,
        )

    def create_runtime(self, spec: ModelSpec, env_cfg, gym_module) -> BenchmarkRuntime:
        import copy

        from ame_baseline.ame_amp_env_cfg import AME_AMP_ENV_ID, AmeParallelismAmpCrossLargeComplexEnvCfg
        from ame_baseline.ame_amp_runner import AmeAmpOnPolicyRunner
        from ame_baseline.ame_amp_train_cfg import get_ame_amp_train_cfg
        from ame_baseline.ame_env_wrapper import AmeRslRlEnvWrapper
        from extension.trajectory_manager_factory import attach_trajectory_manager_if_enabled

        self._register_runtime_env(gym_module, AME_AMP_ENV_ID, AmeParallelismAmpCrossLargeComplexEnvCfg)
        env = gym_module.make(spec.task_id, cfg=env_cfg, render_mode=None)
        base_env = env.unwrapped
        attach_trajectory_manager_if_enabled(
            base_env,
            env_cfg,
            experiment_name=env_cfg.experiment_name,
            device=base_env.device,
        )
        wrapped_env = AmeRslRlEnvWrapper(base_env, clip_actions=100.0)
        runner = AmeAmpOnPolicyRunner(
            wrapped_env,
            copy.deepcopy(get_ame_amp_train_cfg()),
            log_dir=None,
            device=env_cfg.sim.device,
        )
        mode = runner.load_amp_checkpoint(spec.checkpoint, keep_std=True)
        self.require_full_amp_resume(mode)
        policy = runner.get_inference_policy(device=wrapped_env.device)
        observations, _ = wrapped_env.get_observations()
        return BenchmarkRuntime(env, base_env, wrapped_env, policy, observations)


def build_benchmark_adapter(spec: ModelSpec):
    adapters = {
        "default": DefaultBenchmarkAdapter,
        "ame": AmeBenchmarkAdapter,
        "ame_amp": AmeAmpBenchmarkAdapter,
    }
    try:
        return adapters[spec.adapter_kind]()
    except KeyError as exc:
        raise ValueError(f"unknown benchmark adapter: {spec.adapter_kind}") from exc
