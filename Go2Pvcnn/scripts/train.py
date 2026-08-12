#!/usr/bin/env python3
"""
Train Go2 robot with Teacher mode (ground truth semantic labels) using RSL-RL-2.01 PPO.

This training script uses:
- Real semantic labels from LiDAR (no PVCNN inference)
- Cost map generation from semantic labels
- rsl-rl-2.01 package (local installation)
- Wrapper from go2_pvcnn.wrapper directory

Usage:
    Single GPU:
        python train.py --num_envs 256 --headless
    
    Multi-GPU (2 GPUs):
        python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \\
            train.py --num_envs 512 --headless --distributed
"""

import argparse
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import MethodType

import numpy as np
import torch


THIS_FILE = Path(__file__).resolve()
GO2PVCNN_ROOT = THIS_FILE.parent.parent
RSL_RL_ROOT = GO2PVCNN_ROOT / "rsl_rl"
for _path in (GO2PVCNN_ROOT, RSL_RL_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def build_run_log_dir(
    *,
    log_root_path: str | os.PathLike[str],
    run_name: str | None = None,
    now: datetime | None = None,
    mkdir: bool = True,
) -> str:
    """Create (optionally) and return an absolute run directory under a log root.

    This intentionally mirrors the `logs/<category>/<name>/<timestamp>` structure used
    by training scripts, but is kept dependency-free so it can be reused by offline
    benchmark entrypoints without importing Isaac Lab.
    """

    root = os.path.abspath(os.fspath(log_root_path))
    if run_name is None:
        stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = stamp
    else:
        run_dir = str(run_name)
    out = os.path.join(root, run_dir)
    if mkdir:
        os.makedirs(out, exist_ok=True)
    return out


def get_git_short_hash(repo_root: str | os.PathLike[str]) -> str:
    """Return the short HEAD hash used to version a training run directory."""

    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(repo_root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Unable to resolve Git HEAD for training output: {repo_root}") from exc
    git_hash = result.stdout.strip()
    if not git_hash:
        raise RuntimeError(f"Git returned an empty HEAD hash for training output: {repo_root}")
    return git_hash


def find_latest_run_dir(log_root_path: str | os.PathLike[str]) -> str | None:
    """Find the newest old-style or date/hash-style run directory."""

    root = os.path.abspath(os.fspath(log_root_path))
    if not os.path.isdir(root):
        return None
    candidates: list[str] = []
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        date_dir = entry.path
        if any(child.name.startswith("model_") and child.name.endswith(".pt") for child in os.scandir(date_dir)):
            candidates.append(date_dir)
            continue
        for hash_entry in os.scandir(date_dir):
            if not hash_entry.is_dir():
                continue
            if any(
                child.name.startswith("model_") and child.name.endswith(".pt")
                for child in os.scandir(hash_entry.path)
            ):
                candidates.append(hash_entry.path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Train Go2 robot with Teacher semantic labels using RSL-RL PPO.")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
    parser.add_argument(
        "--mpc_num_envs",
        type=int,
        default=None,
        help=(
            "Number of environments sampled by the MPC planner per replan. "
            "Defaults to the experiment config value."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
    parser.add_argument("--max_iterations", type=int, default=5000, help="Maximum training iterations.")
    parser.add_argument("--video", action="store_true", default=False, help="Record training videos.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos (steps).")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between recordings (steps).")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from checkpoint.")
    parser.add_argument("--load_run", type=str, default=None, help="Name of run to load when resuming.")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Checkpoint file to load.")
    parser.add_argument(
        "--keep_std",
        action="store_true",
        default=False,
        help="Keep the action std stored in the checkpoint when resuming; default resets it to init_noise_std.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Enable multi-GPU training with PyTorch distributed.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="teacher_elevation_trajectory_mpc_semantic",
        choices=[
            "teacher_elevation_trajectory_mpc_semantic",
            "teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance",
            "parallelism_tracking_flat",
            "parallelism_tracking_small_obstacles",
            "parallelism_tracking_ladder",
            "parallelism_tracking_cross_large_complex",
        ],
        help="Experiment: semantic MPC teacher or flat-small avoidance continuation.",
    )
    parser.add_argument(
        "--verbose-planner",
        action="store_true",
        default=False,
        help="Print compact planner timing diagnostics (quiet by default).",
    )
    parser.add_argument(
        "--planner-backend",
        type=str,
        default="mpc",
        choices=["mpc"],
        help="Trajectory planner backend. Cleanup build supports only mpc.",
    )

    AppLauncher.add_app_launcher_args(parser)
    return parser


def _parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


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


def _prepare_runtime_args(args_cli: argparse.Namespace) -> argparse.Namespace:
    # Critical for Isaac Lab/CUDA startup ordering: the allocator must be configured
    # before AppLauncher constructs the simulation app.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # GPU MAPPING FOR MULTI-GPU (must be before AppLauncher)
    if args_cli.distributed and "GPU_IDS" in os.environ:
        gpu_ids = [int(x.strip()) for x in os.environ["GPU_IDS"].split(",") if x.strip()]
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if local_rank >= len(gpu_ids):
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but GPU_IDS only has {len(gpu_ids)} GPUs: {os.environ['GPU_IDS']}"
            )

        target_gpu_id = gpu_ids[local_rank]
        args_cli.device = f"cuda:{target_gpu_id}"

        print(f"\n[GPU Mapping] LOCAL_RANK={local_rank} -> GPU {target_gpu_id}")
        print(f"[GPU Mapping] Set device to: {args_cli.device}")

    # Distributed training runs one Kit process per rank.
    if args_cli.distributed:
        _ls = getattr(args_cli, "livestream", 0)
        if _ls in (1, 2):
            print(
                "[train.py] Distributed mode: ignoring --livestream "
                f"(was {_ls}); use single-GPU training for WebRTC visualization."
            )
            args_cli.livestream = 0

    # Launch Isaac Sim
    if getattr(args_cli, "livestream", -1) in (1, 2) and not args_cli.enable_cameras:
        args_cli.enable_cameras = True
        print(
            "[INFO][train.py] livestream: enabled AppLauncher --enable_cameras so the simulator "
            "uses a rendering experience (works without X11; WebRTC client on another machine)."
        )

    return args_cli


def _launch_app(args_cli: argparse.Namespace):
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    return app_launcher, app_launcher.app


def _attach_reference_manager_if_enabled(env, env_cfg, experiment_name: str) -> None:
    from extension.trajectory_manager_factory import attach_trajectory_manager_if_enabled

    manager_device = getattr(env, "device", env_cfg.sim.device)
    manager = attach_trajectory_manager_if_enabled(
        env,
        env_cfg,
        experiment_name=experiment_name,
        device=manager_device,
    )
    if manager is not None:
        print(
            f"[Planner] Attached {getattr(manager, 'planner_backend', 'mpc')} trajectory manager "
            f"for {experiment_name}"
        )


class _StepTimingProbe:
    """Compact per-env-step timer for locating slow 4096 rollout stages."""

    def __init__(self, env, *, label: str, max_prints: int = 5, cuda_sync: bool = True):
        self.env = env
        self.label = label
        self.max_prints = max(0, int(max_prints))
        self.cuda_sync = bool(cuda_sync)
        self.step_idx = 0
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self._orig_step = env.step

    def _sync(self) -> None:
        device = torch.device(getattr(self.env, "device", "cpu"))
        if self.cuda_sync and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _now(self) -> float:
        self._sync()
        return time.perf_counter()

    def record(self, name: str, elapsed: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + float(elapsed)
        self.counts[name] = self.counts.get(name, 0) + 1

    def wrap_method(self, obj, method_name: str, label: str) -> None:
        if obj is None or not hasattr(obj, method_name):
            return
        original = getattr(obj, method_name)
        if getattr(original, "_t302g_timed", False):
            return

        def timed(*args, **kwargs):
            t0 = self._now()
            try:
                return original(*args, **kwargs)
            finally:
                self.record(label, self._now() - t0)

        timed._t302g_timed = True  # type: ignore[attr-defined]
        setattr(obj, method_name, timed)

    def wrap_env_step(self) -> None:
        def timed_step(env_self, action):
            self.step_idx += 1
            self.totals = {}
            self.counts = {}
            t0 = self._now()
            try:
                return self._orig_step(action)
            finally:
                total = self._now() - t0
                self.record("env.step.total", total)
                if self.step_idx <= self.max_prints:
                    ordered = [
                        "env.step.total",
                        "action.process",
                        "action.apply",
                        "scene.write",
                        "sim.step",
                        "scene.update",
                        "termination.compute",
                        "reward.compute",
                        "reset_idx",
                        "command.compute",
                        "event.interval",
                        "observation.compute",
                    ]
                    parts = []
                    for key in ordered:
                        if key in self.totals:
                            parts.append(f"{key}={self.totals[key] * 1000.0:.2f}ms/{self.counts[key]}")
                    extra = [key for key in self.totals if key not in ordered]
                    for key in sorted(extra):
                        parts.append(f"{key}={self.totals[key] * 1000.0:.2f}ms/{self.counts[key]}")
                    print(f"[Timing][{self.label}] step={self.step_idx} " + " ".join(parts), flush=True)
                    manager = getattr(env_self, "_trajectory_manager", None)
                    if manager is not None and hasattr(manager, "runtime_counters"):
                        counters = manager.runtime_counters()
                        if counters:
                            formatted = " ".join(f"{key}={value}" for key, value in sorted(counters.items()))
                            print(f"[Timing][{self.label}][mpc_counters] step={self.step_idx} {formatted}", flush=True)

        self.env.step = MethodType(timed_step, self.env)

    def wrap_reward_terms(self) -> None:
        manager = getattr(self.env, "reward_manager", None)
        if manager is None or getattr(manager, "_t302g_reward_terms_timed", False):
            return
        original_compute = manager.compute

        def timed_reward_compute(dt: float):
            manager._reward_buf[:] = 0.0
            term_parts = []
            for name, term_cfg in zip(manager._term_names, manager._term_cfgs):
                if term_cfg.weight == 0.0:
                    continue
                t0 = self._now()
                value = term_cfg.func(manager._env, **term_cfg.params) * term_cfg.weight * dt
                elapsed = self._now() - t0
                self.record(f"reward.term.{name}", elapsed)
                if self.step_idx <= self.max_prints:
                    term_parts.append(f"{name}={elapsed * 1000.0:.2f}ms")
                manager._reward_buf += value
                manager._episode_sums[name] += value
                manager._step_reward[:, manager._term_names.index(name)] = value / dt
            if self.step_idx <= self.max_prints and term_parts:
                print(f"[Timing][{self.label}][reward_terms] step={self.step_idx} " + " ".join(term_parts), flush=True)
            return manager._reward_buf

        timed_reward_compute._t302g_timed = True  # type: ignore[attr-defined]
        manager.compute = timed_reward_compute
        manager._t302g_reward_terms_timed = True
        manager._t302g_original_compute = original_compute


def _attach_step_timing_probe(env, experiment_name: str) -> None:
    enabled = os.environ.get("T302G_STEP_TIMING", "")
    if not enabled and experiment_name != "teacher_elevation_trajectory_mpc_semantic":
        return
    max_prints = int(os.environ.get("T302G_STEP_TIMING_STEPS", "5"))
    cuda_sync = os.environ.get("T302G_STEP_TIMING_CUDA_SYNC", "1") != "0"
    probe = _StepTimingProbe(env, label=experiment_name, max_prints=max_prints, cuda_sync=cuda_sync)
    probe.wrap_method(env.action_manager, "process_action", "action.process")
    probe.wrap_method(env.action_manager, "apply_action", "action.apply")
    probe.wrap_method(env.scene, "write_data_to_sim", "scene.write")
    probe.wrap_method(env.scene, "update", "scene.update")
    probe.wrap_method(env.sim, "step", "sim.step")
    probe.wrap_method(env.termination_manager, "compute", "termination.compute")
    probe.wrap_method(env.reward_manager, "compute", "reward.compute")
    probe.wrap_method(env.observation_manager, "compute", "observation.compute")
    probe.wrap_method(env.command_manager, "compute", "command.compute")
    probe.wrap_method(env.event_manager, "apply", "event.interval")
    probe.wrap_method(env, "_reset_idx", "reset_idx")
    probe.wrap_reward_terms()
    probe.wrap_env_step()
    env._t302g_step_timing_probe = probe
    print(
        f"[Timing] Enabled env.step timing for {experiment_name}: "
        f"max_prints={max_prints}, cuda_sync={cuda_sync}",
        flush=True,
    )


def _livestream_camera_update_interval(livestream: int) -> int:
    return 4 if livestream in (1, 2) else 1


def _compute_follow_camera_pose(robot_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera_direction = np.array([3.0, 0.0, 0.0], dtype=np.float64)
    camera_position = robot_pos - camera_direction + np.array([0.0, 0.0, 1.5], dtype=np.float64)
    return camera_position, robot_pos


class _SingleEnvLivestreamFollowCamera:
    """Follow env0 during single-env WebRTC training without touching multi-env runs."""

    def __init__(self, env, *, enabled: bool, livestream: int, num_envs: int):
        self.env = env
        self.enabled = bool(enabled)
        self.livestream = int(livestream)
        self.num_envs = int(num_envs)
        self.interval = _livestream_camera_update_interval(self.livestream)
        self.step_idx = 0
        self._orig_step = env.step

    def should_update(self) -> bool:
        if not self.enabled:
            return False
        if self.livestream not in (1, 2):
            return False
        if self.num_envs != 1:
            return False
        return self.step_idx % max(1, self.interval) == 0

    def _update_camera(self, env_self) -> None:
        try:
            robot_pos = env_self.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
            camera_position, target_position = _compute_follow_camera_pose(robot_pos)
            env_self.sim.set_camera_view(camera_position, target_position)
        except Exception as exc:
            print(f"[WARN][train.py] livestream follow camera update failed once: {exc}", flush=True)
            self.enabled = False

    def wrap_env_step(self) -> None:
        if not self.enabled:
            return
        if getattr(self.env.step, "_single_env_livestream_follow_camera", False):
            return

        def followed_step(env_self, action):
            result = self._orig_step(action)
            self.step_idx += 1
            if self.should_update():
                self._update_camera(env_self)
            return result

        followed_step._single_env_livestream_follow_camera = True  # type: ignore[attr-defined]
        self.env.step = MethodType(followed_step, self.env)


def _attach_single_env_livestream_follow_camera(env, *, rank: int, livestream: int, num_envs: int) -> None:
    enabled = rank == 0 and livestream in (1, 2) and num_envs == 1
    if not enabled:
        return
    follow_camera = _SingleEnvLivestreamFollowCamera(
        env,
        enabled=enabled,
        livestream=livestream,
        num_envs=num_envs,
    )
    follow_camera.wrap_env_step()
    env._single_env_livestream_follow_camera = follow_camera
    print(
        "[train.py] Single-env livestream follow camera enabled "
        f"(interval={follow_camera.interval} env steps, env0 root).",
        flush=True,
    )


def main() -> int:
    """Main training function."""

    args_cli = _prepare_runtime_args(_parse_args())
    requested_livestream = int(getattr(args_cli, "livestream", 0))
    app_launcher, simulation_app = _launch_app(args_cli)

    import gymnasium as gym

    from agent import get_train_cfg
    from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
        TeacherElevationTrajectoryMpcSemanticEnvCfg,
        TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg,
    )
    from tracking.parallelism_ladder_env_cfg import ParallelismTrackingLadderEnvCfg
    from tracking.parallelism_cross_large_complex_env_cfg import ParallelismTrackingCrossLargeComplexEnvCfg
    from tracking.parallelism_small_obstacles_env_cfg import ParallelismTrackingSmallObstaclesEnvCfg
    from tracking.parallelism_tracking_env_cfg import ParallelismTrackingFlatEnvCfg
    import go2_pvcnn.tasks.register_envs  # noqa: F401 — register Gym tasks
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_yaml
    from rsl_rl.runners import OnPolicyRunner

    # Configure PyTorch only when the script is actually running.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    dist = None

    # ========================================
    # Multi-GPU Setup
    # ========================================
    if args_cli.distributed:
        import torch.distributed as dist

        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            raise RuntimeError(
                "Distributed mode enabled but RANK/WORLD_SIZE not set. "
                "Use: python -m torch.distributed.run --nproc_per_node=N script.py --distributed"
            )

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = app_launcher.local_rank

        print(f"[Multi-GPU] Global Rank: {rank}/{world_size}, Local Rank: {local_rank}")
        print(f"[Multi-GPU] Device: {args_cli.device}")

        # Initialize process group
        dist.init_process_group(backend="nccl", init_method="env://")

        # Set CUDA device
        device_id = app_launcher.device_id
        torch.cuda.set_device(device_id)

        # Divide environments across GPUs
        envs_per_gpu = args_cli.num_envs // world_size
        args_cli.num_envs = envs_per_gpu
        print(f"[Multi-GPU] Adjusted to {envs_per_gpu} envs per GPU ({envs_per_gpu * world_size} total)")
    else:
        rank = 0
        world_size = 1
        device_id = app_launcher.device_id
        torch.cuda.set_device(device_id)
        print(f"[Single-GPU] Using device: cuda:{device_id}")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if rank == 0:
            print(f"\n[CUDA] Available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.2f} GB)")
    
    # ========================================
    # Create Environment Configuration
    # ========================================
    EXPERIMENT_ENV_MAP = {
        "teacher_elevation_trajectory_mpc_semantic": (
            TeacherElevationTrajectoryMpcSemanticEnvCfg,
            "Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0",
        ),
        "teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance": (
            TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg,
            "Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Flat-Small-Avoidance-Go2-v0",
        ),
        "parallelism_tracking_flat": (
            ParallelismTrackingFlatEnvCfg,
            "Isaac-Go2-Parallelism-Tracking-Flat-v0",
        ),
        "parallelism_tracking_small_obstacles": (
            ParallelismTrackingSmallObstaclesEnvCfg,
            "Isaac-Go2-Parallelism-Tracking-Small-Obstacles-v0",
        ),
        "parallelism_tracking_ladder": (
            ParallelismTrackingLadderEnvCfg,
            "Isaac-Go2-Parallelism-Tracking-Ladder-v0",
        ),
        "parallelism_tracking_cross_large_complex": (
            ParallelismTrackingCrossLargeComplexEnvCfg,
            "Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-v0",
        ),
    }
    env_cfg_cls, env_id = EXPERIMENT_ENV_MAP[args_cli.experiment]
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = f"cuda:{app_launcher.device_id}"
    if args_cli.mpc_num_envs is not None:
        if int(args_cli.mpc_num_envs) <= 0:
            raise ValueError(f"--mpc_num_envs must be positive, got {args_cli.mpc_num_envs}.")
        env_cfg.mpc_planner_cfg.runtime.parallel_plan_batch_size = int(args_cli.mpc_num_envs)

    if args_cli.distributed:
        env_cfg.seed = args_cli.seed + app_launcher.local_rank
    else:
        env_cfg.seed = args_cli.seed

    # Planner verbosity is owned by the planner/manager path; the train CLI only toggles it.
    setattr(env_cfg, "verbose_planner", bool(getattr(args_cli, "verbose_planner", False)))
    setattr(env_cfg, "planner_backend", str(args_cli.planner_backend))

    # ========================================
    # Setup Logging Directory
    # ========================================
    experiment_name = args_cli.experiment
    log_root_path = os.path.join("logs", "rsl_rl", experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    if rank == 0:
        date_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        git_hash = get_git_short_hash(GO2PVCNN_ROOT)
        log_dir = os.path.join(log_root_path, date_dir, git_hash)
        os.makedirs(log_dir, exist_ok=True)
        print(f"[Logging] Directory: {log_dir}")
        
        if args_cli.distributed:
            temp_log_path = "/tmp/teacher_mpc_semantic_log_dir.txt"
            with open(temp_log_path, "w") as f:
                f.write(log_dir)
    
    # Synchronize across all processes
    if args_cli.distributed:
        dist.barrier()
        if rank != 0:
            with open("/tmp/teacher_mpc_semantic_log_dir.txt", "r") as f:
                log_dir = f.read().strip()
            print(f"[Rank {rank}] Using shared log dir: {log_dir}")
    
    # ========================================
    # Create Environment
    # ========================================
    print(f"\n[Env] Creating {experiment_name} Environment...")
    print(f"  - num_envs: {env_cfg.scene.num_envs}")
    print(f"  - mpc_num_envs: {env_cfg.mpc_planner_cfg.runtime.parallel_plan_batch_size}")
    print(f"  - device: {env_cfg.sim.device}")
    print(f"  - seed: {env_cfg.seed}")
    
    # Create gym environment
    env = gym.make(env_id, cfg=env_cfg)
    
    # Cast to ManagerBasedRLEnv for type safety
    assert isinstance(env.unwrapped, ManagerBasedRLEnv)
    base_env: ManagerBasedRLEnv = env.unwrapped
    _attach_reference_manager_if_enabled(base_env, env_cfg, experiment_name)
    _attach_step_timing_probe(base_env, experiment_name)
    _attach_single_env_livestream_follow_camera(
        base_env,
        rank=rank,
        livestream=requested_livestream,
        num_envs=int(args_cli.num_envs),
    )
    
    print(f"[Env] Environment created successfully")
    print(f"  - observation_space: {env.observation_space}")
    print(f"  - action_space: {env.action_space}")
    
    # ========================================
    # Wrap Environment for RSL-RL
    # ========================================
    print(
        "\n[Wrapper] Creating RSL-RL environment wrapper... "
        "(next: initial env.reset(); batched reference replans on startup + "
        "cfg.mpc_planner_cfg.runtime.replan_interval_steps)",
        flush=True,
    )
    
    # Note: For teacher mode, we don't need PVCNN wrapper
    # We create a simple wrapper that doesn't require pvcnn_wrapper parameter
    # This is a temporary solution - you might want to create a specific TeacherEnvWrapper
    
    # For now, we'll use the PVCNN wrapper without actual PVCNN (set to None)
    # Or we can create a simpler wrapper. Let me create a simple wrapper class:
    
    from rsl_rl.env import VecEnv
    
    class SimpleRslRlEnvWrapper(VecEnv):
        """Simple wrapper for RSL-RL without PVCNN."""
        
        def __init__(self, env: ManagerBasedRLEnv, clip_actions: float | None = None):
            self.env = env
            self.clip_actions = clip_actions
            self.num_envs = env.num_envs
            self.device = env.device
            self.max_episode_length = env.max_episode_length
            
            if hasattr(env, "action_manager"):
                self.num_actions = env.action_manager.total_action_dim
            else:
                self.num_actions = gym.spaces.flatdim(env.single_action_space)
            
            # Modify action space
            if clip_actions is not None:
                self.env.action_space = gym.spaces.Box(
                    low=-clip_actions, high=clip_actions,
                    shape=(self.num_actions,), dtype=env.action_space.dtype
                )
            
            # Reset environment
            self.env.reset()

        def _flatten_group(self, obs_dict, group_names: list[str]) -> torch.Tensor:
            values = []
            for name in group_names:
                value = obs_dict[name]
                values.append(value.reshape(value.shape[0], -1))
            return torch.cat(values, dim=-1)

        def _format_observations(self, obs_dict) -> tuple[torch.Tensor, dict]:
            policy_obs = self._flatten_group(obs_dict, ["policy_elevation_semantic_map", "policy_state"])
            critic_obs = self._flatten_group(obs_dict, ["critic_elevation_semantic_map", "critic_state"])
            return policy_obs, {"observations": {"critic": critic_obs}}
        
        @property
        def unwrapped(self):
            return self.env.unwrapped
        
        @property
        def cfg(self):
            """Return environment configuration for logger."""
            return self.env.unwrapped.cfg
        
        @property
        def episode_length_buf(self):
            return self.env.unwrapped.episode_length_buf
        
        @episode_length_buf.setter
        def episode_length_buf(self, value):
            self.env.unwrapped.episode_length_buf = value
        
        @property
        def observation_space(self):
            return self.env.observation_space
        
        @property
        def action_space(self):
            return self.env.action_space
        
        def get_observations(self):
            obs_dict = self.env.unwrapped.observation_manager.compute()
            return self._format_observations(obs_dict)
        
        def reset(self):
            obs_dict, _ = self.env.reset()
            return self._format_observations(obs_dict)
        
        def step(self, actions):
            if self.clip_actions is not None:
                actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

            obs_dict, rewards, dones, truncated, extras = self.env.step(actions)


            # Combine dones and truncated
            dones = dones | truncated

            # PPO bootstrap on timeout (for correct value estimation)
            extras["time_outs"] = truncated
            policy_obs, obs_extras = self._format_observations(obs_dict)
            extras.setdefault("observations", {}).update(obs_extras["observations"])

            return policy_obs, rewards, dones, extras
    
    # Create wrapper
    wrapped_env = SimpleRslRlEnvWrapper(base_env, clip_actions=100.0)
    print(f"[Wrapper] Wrapper created")
    
    # ========================================
    # Create Runner Configuration
    # ========================================
    print(f"\n[Runner] Creating RSL-RL runner configuration...")

    # Training configuration from agent module
    train_cfg = get_train_cfg(experiment_name)
    # 4096-env semantic MPC runs with CNN maps can exceed 24GB cards when rollout
    # horizon/minibatch are too large. Keep the CLI unchanged and apply a runtime
    # memory guardrail on the trainer config.
    num_envs = int(env_cfg.scene.num_envs)
    if num_envs >= 4096 and int(train_cfg.get("num_steps_per_env", 0)) > 24:
        old_steps = int(train_cfg["num_steps_per_env"])
        train_cfg["num_steps_per_env"] = 24
        print(
            f"[Runner][MemGuard] {experiment_name} @ 4096 envs: "
            f"num_steps_per_env {old_steps} -> {train_cfg['num_steps_per_env']}"
        )
    algorithm_cfg = train_cfg.get("algorithm", {})
    if num_envs >= 4096 and isinstance(algorithm_cfg, dict):
        total_batch = int(num_envs * int(train_cfg.get("num_steps_per_env", 24)))
        target_mini_batch = 12_288
        min_mini_batches = max(1, math.ceil(total_batch / target_mini_batch))
        old_mini_batches = int(algorithm_cfg.get("num_mini_batches", 4))
        if old_mini_batches < min_mini_batches:
            algorithm_cfg["num_mini_batches"] = min_mini_batches
            print(
                f"[Runner][MemGuard] {experiment_name} @ 4096 envs: "
                f"num_mini_batches {old_mini_batches} -> {min_mini_batches}"
            )

    # Print configuration
    if rank == 0:
        print(f"[Runner] Configuration:")
        print(f"  - num_steps_per_env: {train_cfg['num_steps_per_env']}")
        print(f"  - max_iterations: {args_cli.max_iterations}")
        print(f"  - learning_rate: {train_cfg['algorithm']['learning_rate']}")
        print(f"  - num_learning_epochs: {train_cfg['algorithm']['num_learning_epochs']}")
    
    # ========================================
    # Create Runner
    # ========================================
    print(f"\n[Runner] Creating OnPolicyRunner...")
    
    runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=log_dir, device=env_cfg.sim.device)
    
    print(f"[Runner] Runner created successfully")
    
    # ========================================
    # Resume from Checkpoint (if specified)
    # ========================================
    if args_cli.resume:
        print(f"\n[Resume] Loading checkpoint...")
        
        if args_cli.load_run is not None:
            resume_path = os.path.join(log_root_path, args_cli.load_run)
        else:
            # Find latest run
            resume_path = find_latest_run_dir(log_root_path)
        
        if resume_path is None:
            raise ValueError("No run found to resume from!")
        
        print(f"[Resume] Loading from: {resume_path}")
        
        if args_cli.load_checkpoint is not None:
            checkpoint_file = args_cli.load_checkpoint
        else:
            checkpoint_file = "model_最新.pt"
        
        checkpoint_path = os.path.join(resume_path, checkpoint_file)
        
        if os.path.exists(checkpoint_path):
            runner.load(checkpoint_path, keep_std=args_cli.keep_std)
            print(f"[Resume] Checkpoint loaded: {checkpoint_path}")
            if args_cli.keep_std:
                print("[Resume] Keeping action std from checkpoint")
            else:
                print("[Resume] Resetting action std to init_noise_std")
        else:
            print(f"[Resume] WARNING: Checkpoint not found: {checkpoint_path}")
    
    # ========================================
    # Save Configuration
    # ========================================
    if rank == 0:
        # Save environment config
        env_cfg_dict = env_cfg.to_dict()
        dump_yaml(os.path.join(log_dir, "env_cfg.yaml"), _yaml_safe(env_cfg_dict))
        
        # Save training config
        dump_yaml(os.path.join(log_dir, "train_cfg.yaml"), _yaml_safe(train_cfg))
        
        print(f"\n[Config] Configurations saved to {log_dir}")
    
    # ========================================
    # Start Training
    # ========================================
    print(f"\n{'='*80}")
    print(f"Starting Training - {experiment_name}")
    print(f"{'='*80}\n")
    
    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}\n")

    # ========================================
    # Cleanup
    # ========================================
    env.close()
    if args_cli.distributed and dist is not None and dist.is_initialized():
        dist.destroy_process_group()
    simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
