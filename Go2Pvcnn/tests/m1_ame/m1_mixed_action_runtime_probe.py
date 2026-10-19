#!/usr/bin/env python3
"""IsaacLab runtime probe for the M1 mixed leg/wheel action contract."""

from __future__ import annotations

import argparse
import sys
import traceback


def _all_finite(value) -> bool:
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(item) for item in value)
    return True


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.num_envs <= 0 or args.steps <= 0:
        raise ValueError("--num_envs and --steps must be positive")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    print(f"M1 mixed action probe: app_ready num_envs={args.num_envs}", flush=True)
    env = None
    try:
        import gymnasium as gym
        import torch

        from go2_pvcnn.assets.m1 import M1_USD_JOINT_NAMES
        from m1_ame.m1_ame_env_cfg import M1AmeCrossLargeComplexEnvCfg
        from m1_ame.m1_ame_rewards import M1_LEG_JOINT_NAMES, M1_WHEEL_JOINT_NAMES

        env_id = "Isaac-M1-Cross-Large-Complex-AME-Mixed-Action-Probe-v0"
        if env_id not in gym.registry:
            gym.register(
                id=env_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                kwargs={"env_cfg_entry_point": M1AmeCrossLargeComplexEnvCfg, "rsl_rl_cfg_entry_point": None},
                disable_env_checker=True,
            )

        device = f"cuda:{app_launcher.device_id}"
        torch.cuda.set_device(app_launcher.device_id)
        env_cfg = M1AmeCrossLargeComplexEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = device
        env = gym.make(env_id, cfg=env_cfg)
        print("M1 mixed action probe: env_created", flush=True)
        unwrapped = env.unwrapped
        observations, _ = env.reset()
        print("M1 mixed action probe: reset_done", flush=True)

        action_manager = unwrapped.action_manager
        if action_manager.total_action_dim != 16:
            raise AssertionError(f"expected action_dim=16, got {action_manager.total_action_dim}")
        if action_manager.active_terms != ["MixedJointAction"]:
            raise AssertionError(f"unexpected action terms: {action_manager.active_terms}")

        robot = unwrapped.scene["robot"]
        leg_ids = robot.find_joints(list(M1_LEG_JOINT_NAMES), preserve_order=True)[0]
        wheel_ids = robot.find_joints(list(M1_WHEEL_JOINT_NAMES), preserve_order=True)[0]
        wheel_action_indices = [M1_USD_JOINT_NAMES.index(name) for name in M1_WHEEL_JOINT_NAMES]
        wheel_position_before = robot.data.joint_pos_target[:, wheel_ids].clone()

        actions = torch.zeros((args.num_envs, 16), device=unwrapped.device)
        actions[:, wheel_action_indices] = 1.0
        for _ in range(args.steps):
            observations, rewards, terminated, truncated, _ = env.step(actions)
            tensors = (observations, rewards, terminated, truncated, robot.data.joint_vel_target)
            if not _all_finite(tensors):
                raise AssertionError("M1 mixed action probe observed a non-finite tensor")
        print("M1 mixed action probe: steps_done", flush=True)

        wheel_velocity_target = robot.data.joint_vel_target[:, wheel_ids]
        if not torch.allclose(wheel_velocity_target, torch.full_like(wheel_velocity_target, 8.0), atol=1e-5):
            raise AssertionError(
                f"wheel velocity target mismatch: min={wheel_velocity_target.min().item()} "
                f"max={wheel_velocity_target.max().item()}"
            )
        wheel_position_after = robot.data.joint_pos_target[:, wheel_ids]
        if not torch.allclose(wheel_position_after, wheel_position_before, atol=1e-5):
            raise AssertionError("mixed action wrote a position target to a wheel joint")

        leg_position_target = robot.data.joint_pos_target[:, leg_ids]
        if not torch.isfinite(leg_position_target).all():
            raise AssertionError("leg position targets are non-finite")
        print("M1 mixed action probe: target_checks_done", flush=True)
        print(
            f"M1 mixed action probe passed: num_envs={args.num_envs} steps={args.steps} "
            f"action_dim={action_manager.total_action_dim} wheel_velocity_target=8.0 "
            f"leg_position_target_abs_mean={leg_position_target.abs().mean().item():.6f}",
            flush=True,
        )
        return 0
    finally:
        if env is not None:
            env.close()
        try:
            simulation_app.close()
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise


def _run_cli() -> None:
    try:
        raise SystemExit(main())
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    _run_cli()
