#!/usr/bin/env python3
"""Play an M1 AME-XYZ-Semantic checkpoint with BatchNorm in evaluation mode."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
PACKAGE_ROOT = THIS_FILE.parent.parent
for path in (PACKAGE_ROOT, PACKAGE_ROOT / "rsl_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(args)
    simulation_app = launcher.app
    env = None
    try:
        import gymnasium as gym
        import torch

        from m1_ame.m1_ame_env_cfg import M1M1AmeCrossLargeComplexEnvCfg_PLAY
        from ame_baseline.ame_env_wrapper import AmeRslRlEnvWrapper
        from m1_ame.m1_ame_runner import M1M1AmeOnPolicyRunner
        from m1_ame.m1_ame_train_cfg import get_m1_ame_train_cfg

        env_id = "Isaac-M1-Cross-Large-Complex-AME-Play-v0"
        if env_id not in gym.registry:
            gym.register(
                id=env_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                kwargs={"env_cfg_entry_point": M1AmeCrossLargeComplexEnvCfg_PLAY, "rsl_rl_cfg_entry_point": None},
                disable_env_checker=True,
            )

        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"AME checkpoint not found: {checkpoint}")
        device = f"cuda:{launcher.device_id}"
        env_cfg = M1AmeCrossLargeComplexEnvCfg_PLAY()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = device
        env_cfg.seed = args.seed
        env = gym.make(env_id, cfg=env_cfg)
        wrapped = AmeRslRlEnvWrapper(env, clip_actions=100.0)
        runner = M1AmeOnPolicyRunner(
            wrapped,
            copy.deepcopy(get_m1_ame_train_cfg()),
            log_dir=None,
            device=device,
        )
        runner.load(str(checkpoint), load_optimizer=False, keep_std=True)
        policy = runner.get_inference_policy(device=device)
        observations, _ = wrapped.get_observations()
        for _ in range(args.max_steps):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                observations, _, _, _ = wrapped.step(policy(observations))
        return 0
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
