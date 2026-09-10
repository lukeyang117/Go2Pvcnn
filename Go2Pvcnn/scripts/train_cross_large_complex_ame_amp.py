#!/usr/bin/env python3
"""Train AME with the existing Parallelism AMP algorithm and planner data."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
PACKAGE_ROOT = THIS_FILE.parent.parent
REPO_ROOT = PACKAGE_ROOT.parent
RSL_RL_ROOT = PACKAGE_ROOT / "rsl_rl"
for path in (PACKAGE_ROOT, RSL_RL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "logs/rsl_rl/cross_large_complex_ame/2026-09-08_15-40-23/bfb8b7f/model_19999.pt"
)


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=3500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep_std", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _git_hash() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _yaml_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(_yaml_safe(key)): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    if hasattr(value, "__module__") and hasattr(value, "__qualname__"):
        return f"{value.__module__}:{value.__qualname__}"
    return str(value)


def _register_runtime_env(gym, env_id, env_cfg_cls) -> None:
    if env_id in gym.registry:
        return
    gym.register(
        id=env_id,
        entry_point="tracking.amp_env:ParallelismAmpEnv",
        kwargs={"env_cfg_entry_point": env_cfg_cls, "rsl_rl_cfg_entry_point": None},
        disable_env_checker=True,
    )


def main() -> int:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = _parse_args()
    if args.num_envs <= 0 or args.max_iterations <= 0:
        raise ValueError("--num_envs and --max_iterations must be positive")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"AME checkpoint not found: {checkpoint_path}")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None
    try:
        import gymnasium as gym
        import torch

        import go2_pvcnn.tasks.register_envs  # noqa: F401 - registers scene dependencies
        from ame_baseline.ame_amp_env_cfg import (
            AME_AMP_ENV_ID,
            AME_AMP_EXPERIMENT_NAME,
            AmeParallelismAmpCrossLargeComplexEnvCfg,
        )
        from ame_baseline.ame_amp_runner import AmeAmpOnPolicyRunner
        from ame_baseline.ame_amp_train_cfg import get_ame_amp_train_cfg
        from ame_baseline.ame_env_wrapper import AmeRslRlEnvWrapper
        from extension.trajectory_manager_factory import attach_trajectory_manager_if_enabled
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils.io import dump_yaml

        _register_runtime_env(gym, AME_AMP_ENV_ID, AmeParallelismAmpCrossLargeComplexEnvCfg)
        device = f"cuda:{app_launcher.device_id}"
        torch.cuda.set_device(app_launcher.device_id)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        env_cfg = AmeParallelismAmpCrossLargeComplexEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = device
        env_cfg.seed = args.seed
        output_root = Path(
            os.environ.get(
                "AME_AMP_OUTPUT_ROOT",
                str(REPO_ROOT / "logs" / "rsl_rl" / AME_AMP_EXPERIMENT_NAME),
            )
        ).expanduser().resolve()
        log_dir = (
            output_root
            / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            / _git_hash()
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[AME-AMP] log_dir={log_dir}", flush=True)

        env = gym.make(AME_AMP_ENV_ID, cfg=env_cfg)
        assert isinstance(env.unwrapped, ManagerBasedRLEnv)
        base_env = env.unwrapped
        attach_trajectory_manager_if_enabled(
            base_env,
            env_cfg,
            experiment_name=AME_AMP_EXPERIMENT_NAME,
            device=base_env.device,
        )
        wrapped_env = AmeRslRlEnvWrapper(base_env, clip_actions=100.0)
        policy_obs, extras = wrapped_env.get_observations()
        critic_obs = extras["observations"]["critic"]
        print(
            f"[AME-AMP] observations policy={tuple(policy_obs.shape)} critic={tuple(critic_obs.shape)} "
            f"actions={wrapped_env.num_actions}",
            flush=True,
        )

        train_cfg = get_ame_amp_train_cfg()
        dump_yaml(str(log_dir / "env_cfg.yaml"), _yaml_safe(env_cfg.to_dict()))
        dump_yaml(str(log_dir / "train_cfg.yaml"), _yaml_safe(train_cfg))
        runner = AmeAmpOnPolicyRunner(
            wrapped_env,
            copy.deepcopy(train_cfg),
            log_dir=str(log_dir),
            device=device,
        )
        mode = runner.load_amp_checkpoint(checkpoint_path, keep_std=True)
        with (log_dir / "source_checkpoint.json").open("w", encoding="utf-8") as stream:
            json.dump(
                {"mode": mode, "checkpoint": str(checkpoint_path), **runner.source_metadata},
                stream,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[AME-AMP] checkpoint_mode={mode} checkpoint={checkpoint_path}", flush=True)
        print(f"Starting Training - {AME_AMP_EXPERIMENT_NAME}", flush=True)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
        print(f"Training Complete - {AME_AMP_EXPERIMENT_NAME}", flush=True)
        return 0
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
