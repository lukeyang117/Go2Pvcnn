#!/usr/bin/env python3
"""Train the isolated AME-XYZ-Semantic baseline on the M1 mixed task."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import traceback
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
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--keep_std", action="store_true", default=True)
    parser.add_argument("--reset_std", action="store_false", dest="keep_std")
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


def _training_cfg_for_envs(train_cfg: dict, num_envs: int) -> dict:
    adjusted = copy.deepcopy(train_cfg)
    if num_envs >= 4096:
        # Bound the attention forward/backward minibatch to the 1024-env size.
        base_count = int(adjusted["algorithm"]["num_mini_batches"])
        adjusted["algorithm"]["num_mini_batches"] = max(
            base_count, base_count * ((num_envs + 1023) // 1024)
        )
    return adjusted


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
    training_failed = True
    try:
        import gymnasium as gym
        import torch

        from ame_baseline.actor_critic_ame import ActorCriticAME  # noqa: F401 - runner injection contract
        from m1_ame.m1_ame_env_cfg import M1AmeCrossLargeComplexEnvCfg
        from ame_baseline.ame_env_wrapper import AmeRslRlEnvWrapper
        from m1_ame.m1_ame_runner import M1AmeOnPolicyRunner
        from m1_ame.m1_ame_train_cfg import get_m1_ame_train_cfg
        from isaaclab.utils.io import dump_yaml

        env_id = "Isaac-M1-Cross-Large-Complex-AME-v0"
        if env_id not in gym.registry:
            gym.register(
                id=env_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                kwargs={"env_cfg_entry_point": M1AmeCrossLargeComplexEnvCfg, "rsl_rl_cfg_entry_point": None},
                disable_env_checker=True,
            )

        device = f"cuda:{app_launcher.device_id}"
        torch.cuda.set_device(app_launcher.device_id)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        env_cfg = M1AmeCrossLargeComplexEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = device
        env_cfg.seed = args.seed

        default_log_dir = (
            REPO_ROOT
            / "logs"
            / "rsl_rl"
            / "m1_cross_large_complex_ame"
            / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            / _git_hash()
        )
        log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else default_log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[AME] log_dir={log_dir}", flush=True)

        env = gym.make(env_id, cfg=env_cfg)
        wrapped_env = AmeRslRlEnvWrapper(env, clip_actions=1.0)
        policy_obs, extras = wrapped_env.get_observations()
        critic_obs = extras["observations"]["critic"]
        print(
            f"[AME] observations policy={tuple(policy_obs.shape)} critic={tuple(critic_obs.shape)}",
            flush=True,
        )

        train_cfg = _training_cfg_for_envs(get_m1_ame_train_cfg(), args.num_envs)
        dump_yaml(str(log_dir / "env_cfg.yaml"), _yaml_safe(env_cfg.to_dict()))
        dump_yaml(str(log_dir / "train_cfg.yaml"), _yaml_safe(train_cfg))
        runner = M1AmeOnPolicyRunner(
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
            runner.source_metadata = dict(runner.source_metadata, git_commit=_git_hash(), seed=args.seed)
            (log_dir / "source_checkpoint.json").write_text(
                json.dumps(runner.source_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[AME] resumed checkpoint={checkpoint}", flush=True)

        print("Starting Training - m1_cross_large_complex_ame", flush=True)
        runner.learn(
            num_learning_iterations=args.max_iterations,
            init_at_random_ep_len=not args.resume,
        )
        print("Training Complete - m1_cross_large_complex_ame", flush=True)
        training_failed = False
        return 0
    except SystemExit as exc:
        raise RuntimeError(f"IsaacLab exited during AME training (code={exc.code})") from exc
    finally:
        if not training_failed:
            if env is not None:
                env.close()
            try:
                simulation_app.close()
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise


def _run_cli() -> None:
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Kit may replace an uncaught Python exception with exit status zero.
        os._exit(1)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    _run_cli()
