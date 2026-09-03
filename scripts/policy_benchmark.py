"""Single-checkpoint Isaac Lab benchmark runner.

The script deliberately keeps simulator imports lazy so protocol and CLI tests
can run on machines without Isaac Sim.  One process evaluates one checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
SIM_ROOT = PROJECT_ROOT / "Go2Pvcnn"
RSL_RL_ROOT = SIM_ROOT / "rsl_rl"
for _path in (PROJECT_ROOT, SIM_ROOT, RSL_RL_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Evaluate one policy checkpoint on the paper benchmark.")
    parser.add_argument("--experiment-type", required=True, choices=("amp", "distillation", "ppo", "teacher"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--suite", required=True, choices=("complex_mixed", "large_runway", "small_runway"))
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--layout-manifest", type=Path, default=None)
    parser.add_argument("--layout-count", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/policy_benchmark"))
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--smoke-test", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_conditions(args):
    from evaluation.manifest import build_conditions, read_manifest, write_manifest

    if args.layout_manifest is not None and args.layout_manifest.exists():
        return read_manifest(args.layout_manifest)
    conditions = build_conditions(args.suite, args.layout_count, args.seed)
    if args.layout_manifest is not None:
        write_manifest(args.layout_manifest, conditions)
    return conditions


def _finite(*values) -> bool:
    return all(bool(torch.isfinite(torch.as_tensor(value)).all().item()) for value in values)


def _condition_batch(conditions, start: int, count: int, device: torch.device) -> torch.Tensor:
    rows = [conditions[(start + index) % len(conditions)].command for index in range(count)]
    return torch.as_tensor(rows, dtype=torch.float32, device=device)


def _contact_masks(base_env, num_envs: int, device: torch.device):
    """Best-effort collision/fall masks; unavailable semantic sensors stay false."""

    large = torch.zeros(num_envs, dtype=torch.bool, device=device)
    small = torch.zeros_like(large)
    fallen = torch.zeros_like(large)
    termination = getattr(base_env, "termination_manager", None)
    term_dones = getattr(termination, "_term_dones", None)
    if isinstance(term_dones, dict):
        for name, value in term_dones.items():
            mask = torch.as_tensor(value, dtype=torch.bool, device=device).reshape(-1)
            if mask.numel() != num_envs:
                continue
            if any(token in str(name).lower() for token in ("bad_orientation", "base_contact", "fall")):
                fallen |= mask
    # Tracking tasks expose sticky semantic flags in some versions.
    for attr, target in (("episode_had_large_collision", large), ("episode_had_small_collision", small)):
        value = getattr(base_env, attr, None)
        if value is not None:
            target |= torch.as_tensor(value, dtype=torch.bool, device=device).reshape(-1)
    return large, small, fallen


def _run_rollout(args, simulation_app) -> int:
    import gymnasium as gym
    from isaaclab.envs import ManagerBasedRLEnv
    from rsl_rl.env import VecEnv
    from tensordict import TensorDict

    import go2_pvcnn.tasks.register_envs  # noqa: F401
    import tracking.register_envs  # noqa: F401
    from agent import get_train_cfg
    from evaluation.benchmark_suites import build_suite_config
    from evaluation.manifest import read_manifest
    from evaluation.metrics import EpisodeAccumulator
    from evaluation.model_adapters import build_play_mapping, resolve_model_spec
    from evaluation.result_writer import ResultWriter
    from evaluation.trajectory_alignment import ValidWindowBuffer, tracking_mse
    from tracking.managers.parallelism_reference_manager import get_parallelism_reference_manager
    from tracking.amp_env import _frame_from_robot, _reference_frame
    from tracking.env import ParallelismTrackingEnv
    from Go2Pvcnn.scripts.play import _make_env_wrapper

    spec = resolve_model_spec(args.experiment_type, args.checkpoint)
    conditions = _load_conditions(args)
    if not conditions or any(condition.suite != args.suite for condition in conditions):
        raise ValueError("layout manifest contains no conditions for the requested suite")
    cfg_cls, task_id = build_play_mapping()[args.experiment_type]
    env_cfg = cfg_cls()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.sim.device = args.device
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = int(args.seed)
    env_cfg = build_suite_config(args.suite, env_cfg, conditions[0])
    # Reference trajectories are diagnostics only and are not added to actor obs.
    if spec.requires_reference_manager:
        env_cfg.planner_owned_reference_cache = False

    print(f"[benchmark] creating {task_id} with {args.num_envs} envs", flush=True)
    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    if not isinstance(env.unwrapped, ManagerBasedRLEnv):
        raise TypeError(f"unexpected environment type: {type(env.unwrapped)!r}")
    base_env = env.unwrapped
    manager = get_parallelism_reference_manager(base_env)
    wrapped_env = _make_env_wrapper(base_env, gym_module=gym, vec_env_cls=VecEnv, tensor_dict_cls=TensorDict, clip_actions=100.0)
    runner = __import__("rsl_rl.runners", fromlist=["OnPolicyRunner"]).OnPolicyRunner(
        wrapped_env, get_train_cfg(spec.experiment), log_dir=None, device=env_cfg.sim.device
    )
    runner.load(str(spec.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=wrapped_env.device)
    obs, _ = wrapped_env.consume_initial_observations()
    device = wrapped_env.device
    num_envs = int(args.num_envs)
    metrics = EpisodeAccumulator(num_envs, device)
    windows = ValidWindowBuffer(num_envs, horizon=24, device=device)
    logical_plan_ids = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    completed: list[dict[str, object]] = []
    valid_mse: list[dict[str, float]] = []
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else max(int(c.timeout_steps) for c in conditions)
    started = time.perf_counter()
    for step in range(max_steps):
        if args.suite in {"large_runway", "small_runway"} and step % 23 == 0:
            command = _condition_batch(conditions, step // 23 * num_envs, num_envs, device)
            base_env.command_manager.get_command("base_velocity")[:, :3] = command
        # ParallelismTrackingEnv prepares the reference in its own step hook;
        # ManagerBasedRLEnv (pure PPO) needs the explicit call here.
        if not isinstance(base_env, ParallelismTrackingEnv):
            manager.prepare_step_reference()
        actual_start = _frame_from_robot(base_env)
        with torch.inference_mode():
            actions = policy(obs)
        if not _finite(obs, actions):
            raise FloatingPointError(f"non-finite observation/action at step {step}")
        obs, rewards, dones, extras = wrapped_env.step(actions)
        actual_target = _frame_from_robot(base_env)
        reference_target = _reference_frame(manager, start=False)
        valid = torch.as_tensor(manager.step_plan_valid, dtype=torch.bool, device=device)
        manager_plan_ids = torch.as_tensor(manager.plan_count, dtype=torch.long, device=device)
        # A 24-frame window is a rolling buffer across re-plans.  Keep one
        # logical id while rows remain valid; invalid plans and resets start a
        # fresh id.  This matches the planner's continuous root anchoring.
        logical_plan_ids = torch.where(
            valid & (logical_plan_ids >= 0), logical_plan_ids, manager_plan_ids
        )
        plan_ids = logical_plan_ids
        # Seed B0 once, then append each target frame.  A valid window is 24
        # frames and exactly 23 transitions, matching the planner contract.
        if step == 0:
            windows.push(actual_start, actual_start, plan_ids, valid, torch.zeros(num_envs, dtype=torch.bool, device=device), torch.ones(num_envs, dtype=torch.bool, device=device))
        done_mask = torch.as_tensor(dones, dtype=torch.bool, device=device)
        window_mask = windows.push(actual_target, reference_target, plan_ids, valid, done_mask, valid)
        if args.smoke_test and step in (0, 1, 21, 22, 23, 24):
            print(
                f"[benchmark][diag] step={step} valid={int(valid.sum().item())}/{num_envs} "
                f"plan_count={int(plan_ids.min().item())}-{int(plan_ids.max().item())} phase_max={int(manager.phase.max().item())} "
                f"windows={int(window_mask.sum().item())} buffer_count={int(windows.count.max().item())}",
                flush=True,
            )
        if bool(window_mask.any().item()):
            values = tracking_mse(windows.actual, windows.reference, window_mask[:, None].expand(-1, 24), windows.actual[:, :1, 24:27])
            valid_mse.append({key: float(value.detach().cpu().item()) for key, value in values.items()})
        large, small, fallen = _contact_masks(base_env, num_envs, device)
        metrics.update(base_env.scene["robot"].data.root_pos_w, base_env.command_manager.get_command("base_velocity")[:, :3], large, small, fallen, done_mask, valid)
        done_ids = torch.nonzero(done_mask, as_tuple=False).flatten()
        if int(done_ids.numel()):
            for record in metrics.finish(done_ids):
                record["condition_id"] = conditions[(step * num_envs + int(record["env_id"])) % len(conditions)].condition_id
                completed.append(record)
            logical_plan_ids[done_ids] = -1
        if args.smoke_test and step + 1 >= max_steps:
            break
    elapsed = max(time.perf_counter() - started, 1.0e-9)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "model": spec.name, "checkpoint": str(spec.checkpoint), "checkpoint_sha256": _sha256(spec.checkpoint), "suite": args.suite, "is_smoke_test": bool(args.smoke_test), "num_envs": num_envs, "steps": max_steps, "seed": int(args.seed)}
    writer = ResultWriter(args.output_dir / spec.name / args.suite, manifest)
    for record in completed:
        writer.write_episode(record)
    writer.write_progress([str(record["condition_id"]) for record in completed])
    writer.write_summary({"run_manifest": manifest, "episodes": len(completed), "valid_windows": len(valid_mse), "valid_mse": valid_mse, "env_steps_per_second": num_envs * max_steps / elapsed, "elapsed_s": elapsed})
    print(json.dumps({"episodes": len(completed), "valid_windows": len(valid_mse), "env_steps_per_second": num_envs * max_steps / elapsed}, sort_keys=True), flush=True)
    env.close()
    return 0


def main() -> int:
    args = build_arg_parser().parse_args()
    if not args.checkpoint.exists() or not args.checkpoint.is_file():
        raise ValueError("--checkpoint must point to an explicit checkpoint file")
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    try:
        return _run_rollout(args, app_launcher.app)
    finally:
        app_launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
