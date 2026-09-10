#!/usr/bin/env python3
"""Run a complete AME-AMP checkpoint in the planner-backed Go2 environment."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PACKAGE_ROOT = THIS_FILE.parent.parent
RSL_RL_ROOT = PACKAGE_ROOT / "rsl_rl"
for path in (PACKAGE_ROOT, RSL_RL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


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
    if args.num_steps <= 0 or args.num_envs <= 0:
        raise ValueError("--num_steps and --num_envs must be positive")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"AME-AMP checkpoint not found: {checkpoint}")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None
    try:
        import gymnasium as gym
        import torch

        import go2_pvcnn.tasks.register_envs  # noqa: F401
        from ame_baseline.ame_amp_env_cfg import (
            AME_AMP_ENV_ID,
            AmeParallelismAmpCrossLargeComplexEnvCfg,
        )
        from ame_baseline.ame_amp_runner import AmeAmpOnPolicyRunner
        from ame_baseline.ame_amp_train_cfg import get_ame_amp_train_cfg
        from ame_baseline.ame_env_wrapper import AmeRslRlEnvWrapper
        from extension.trajectory_manager_factory import attach_trajectory_manager_if_enabled

        _register_runtime_env(gym, AME_AMP_ENV_ID, AmeParallelismAmpCrossLargeComplexEnvCfg)
        device = f"cuda:{app_launcher.device_id}"
        torch.cuda.set_device(app_launcher.device_id)
        env_cfg = AmeParallelismAmpCrossLargeComplexEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = device
        env = gym.make(AME_AMP_ENV_ID, cfg=env_cfg)
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
            device=device,
        )
        mode = runner.load_amp_checkpoint(checkpoint, keep_std=True)
        if mode != "full_amp_resume":
            raise ValueError("Play requires a complete AME-AMP checkpoint with AMP head and discriminator")
        runner.alg.actor_critic.eval()
        obs, extras = wrapped_env.get_observations()
        critic_obs = extras["observations"]["critic"]
        with torch.no_grad():
            for _ in range(args.num_steps):
                actions = runner.alg.actor_critic.act_inference(obs.to(device))
                obs, _, dones, infos = wrapped_env.step(actions)
                obs = obs.to(device)
                if dones.any():
                    obs, extras = wrapped_env.get_observations()
                    critic_obs = extras["observations"]["critic"]
        return 0
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
