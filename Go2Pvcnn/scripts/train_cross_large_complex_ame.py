#!/usr/bin/env python3
"""Train the isolated AME-XYZ-Semantic baseline on the Go2 mixed task."""

from __future__ import annotations

import argparse
import copy
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


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
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


def main() -> int:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = _parse_args()
    if args.num_envs <= 0 or args.max_iterations <= 0:
        raise ValueError("--num_envs and --max_iterations must be positive")
    if args.resume and not args.checkpoint:
        raise ValueError("--resume requires --checkpoint")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None
    try:
        import gymnasium as gym
        import torch

        from ame_baseline.actor_critic_ame import ActorCriticAME  # noqa: F401 - runner injection contract
        from ame_baseline.ame_env_cfg import AmeCrossLargeComplexEnvCfg
        from ame_baseline.ame_env_wrapper import AmeRslRlEnvWrapper
        from ame_baseline.ame_runner import AmeOnPolicyRunner
        from ame_baseline.ame_train_cfg import get_ame_train_cfg
        import go2_pvcnn.tasks.register_envs  # noqa: F401
        from isaaclab.utils.io import dump_yaml

        device = f"cuda:{app_launcher.device_id}"
        torch.cuda.set_device(app_launcher.device_id)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        env_cfg = AmeCrossLargeComplexEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = device
        env_cfg.seed = args.seed

        log_dir = (
            REPO_ROOT
            / "logs"
            / "rsl_rl"
            / "cross_large_complex_ame"
            / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            / _git_hash()
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[AME] log_dir={log_dir}", flush=True)

        env = gym.make("Isaac-Go2-Cross-Large-Complex-PPO-v0", cfg=env_cfg)
        wrapped_env = AmeRslRlEnvWrapper(env, clip_actions=100.0)
        policy_obs, extras = wrapped_env.get_observations()
        critic_obs = extras["observations"]["critic"]
        print(
            f"[AME] observations policy={tuple(policy_obs.shape)} critic={tuple(critic_obs.shape)}",
            flush=True,
        )

        train_cfg = get_ame_train_cfg()
        dump_yaml(str(log_dir / "env_cfg.yaml"), _yaml_safe(env_cfg.to_dict()))
        dump_yaml(str(log_dir / "train_cfg.yaml"), _yaml_safe(train_cfg))
        runner = AmeOnPolicyRunner(
            wrapped_env,
            copy.deepcopy(train_cfg),
            log_dir=str(log_dir),
            device=device,
        )
        if args.resume:
            checkpoint = Path(args.checkpoint).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"AME checkpoint not found: {checkpoint}")
            runner.load(str(checkpoint), keep_std=args.keep_std)
            print(f"[AME] resumed checkpoint={checkpoint}", flush=True)

        print("Starting Training - cross_large_complex_ame", flush=True)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
        print("Training Complete - cross_large_complex_ame", flush=True)
        return 0
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
