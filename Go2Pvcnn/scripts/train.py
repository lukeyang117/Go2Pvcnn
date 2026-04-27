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
import os
import sys
from datetime import datetime
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
GO2PVCNN_ROOT = THIS_FILE.parent.parent
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))


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


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Train Go2 robot with Teacher semantic labels using RSL-RL PPO.")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
    parser.add_argument("--max_iterations", type=int, default=5000, help="Maximum training iterations.")
    parser.add_argument("--video", action="store_true", default=False, help="Record training videos.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos (steps).")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between recordings (steps).")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from checkpoint.")
    parser.add_argument("--load_run", type=str, default=None, help="Name of run to load when resuming.")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Checkpoint file to load.")
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Enable multi-GPU training with PyTorch distributed.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="teacher_semantic",
        choices=[
            "teacher_semantic",
            "teacher_without_semantic",
            "teacher_elevation",
            "teacher_elevation_semantic_map",
            "teacher_elevation_trajectory",
        ],
        help="Experiment: teacher_semantic (CNN+state), teacher_without_semantic (state-only), "
        "teacher_elevation (elevation map CNN), teacher_elevation_semantic_map (dual grid CNN), "
        "teacher_elevation_trajectory (high-res elevation + trajectory reward).",
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
        default="together",
        choices=["together", "legacy"],
        help="For teacher_elevation_trajectory: trajectory planner backend.",
    )

    AppLauncher.add_app_launcher_args(parser)
    return parser


def _parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


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
            f"[Planner] Attached {getattr(manager, 'planner_backend', 'legacy')} trajectory manager "
            "for teacher_elevation_trajectory"
        )


def main() -> int:
    """Main training function."""

    args_cli = _prepare_runtime_args(_parse_args())
    app_launcher, simulation_app = _launch_app(args_cli)

    import gymnasium as gym

    from agent import get_train_cfg
    from go2_pvcnn.tasks.teacher_elevation_env_cfg import TeacherElevationEnvCfg
    from go2_pvcnn.tasks.teacher_elevation_semantic_map_env_cfg import TeacherElevationSemanticMapEnvCfg
    from go2_pvcnn.tasks.teacher_elevation_trajectory_env_cfg import TeacherElevationTrajectoryEnvCfg
    from go2_pvcnn.tasks.teacher_semantic_env_cfg import TeacherSemanticEnvCfg
    from go2_pvcnn.tasks.teacher_without_semantic_env_cfg import TeacherWithoutSemanticEnvCfg
    import go2_pvcnn.tasks.register_envs  # noqa: F401 — register Gym tasks
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_yaml
    from rsl_rl_2_01.runners import OnPolicyRunner

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
        "teacher_semantic": (TeacherSemanticEnvCfg, "Isaac-Teacher-Semantic-Go2-v0"),
        "teacher_without_semantic": (TeacherWithoutSemanticEnvCfg, "Isaac-Teacher-Without-Semantic-Go2-v0"),
        "teacher_elevation": (TeacherElevationEnvCfg, "Isaac-Teacher-Elevation-Go2-v0"),
        "teacher_elevation_semantic_map": (
            TeacherElevationSemanticMapEnvCfg,
            "Isaac-Teacher-Elevation-Semantic-Map-Go2-v0",
        ),
        "teacher_elevation_trajectory": (
            TeacherElevationTrajectoryEnvCfg,
            "Isaac-Teacher-Elevation-Trajectory-Go2-v0",
        ),
    }
    env_cfg_cls, env_id = EXPERIMENT_ENV_MAP[args_cli.experiment]
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = f"cuda:{app_launcher.device_id}"

    if args_cli.distributed:
        env_cfg.seed = args_cli.seed + app_launcher.local_rank
    else:
        env_cfg.seed = args_cli.seed

    # Planner verbosity is owned by the planner/manager path; the train CLI only toggles it.
    if args_cli.experiment == "teacher_elevation_trajectory":
        setattr(env_cfg, "verbose_planner", bool(getattr(args_cli, "verbose_planner", False)))
        if getattr(args_cli, "planner_backend", None) is not None:
            setattr(env_cfg, "planner_backend", str(args_cli.planner_backend))

    # ========================================
    # Setup Logging Directory
    # ========================================
    experiment_name = args_cli.experiment
    log_root_path = os.path.join("logs", "rsl_rl", experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    if rank == 0:
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_root_path, log_dir)
        os.makedirs(log_dir, exist_ok=True)
        print(f"[Logging] Directory: {log_dir}")
        
        if args_cli.distributed:
            temp_log_path = "/tmp/teacher_semantic_log_dir.txt"
            with open(temp_log_path, "w") as f:
                f.write(log_dir)
    
    # Synchronize across all processes
    if args_cli.distributed:
        dist.barrier()
        if rank != 0:
            with open("/tmp/teacher_semantic_log_dir.txt", "r") as f:
                log_dir = f.read().strip()
            print(f"[Rank {rank}] Using shared log dir: {log_dir}")
    
    # ========================================
    # Create Environment
    # ========================================
    print(f"\n[Env] Creating {experiment_name} Environment...")
    print(f"  - num_envs: {env_cfg.scene.num_envs}")
    print(f"  - device: {env_cfg.sim.device}")
    print(f"  - seed: {env_cfg.seed}")
    
    # Create gym environment
    env = gym.make(env_id, cfg=env_cfg)
    
    # Cast to ManagerBasedRLEnv for type safety
    assert isinstance(env.unwrapped, ManagerBasedRLEnv)
    base_env: ManagerBasedRLEnv = env.unwrapped
    _attach_reference_manager_if_enabled(base_env, env_cfg, experiment_name)
    
    print(f"[Env] Environment created successfully")
    print(f"  - observation_space: {env.observation_space}")
    print(f"  - action_space: {env.action_space}")
    
    # ========================================
    # Wrap Environment for RSL-RL
    # ========================================
    print(
        "\n[Wrapper] Creating RSL-RL environment wrapper... "
        "(next: initial env.reset(); batched reference replans on startup + "
        "cfg.reference_replan_interval_steps)",
        flush=True,
    )
    
    # Note: For teacher mode, we don't need PVCNN wrapper
    # We create a simple wrapper that doesn't require pvcnn_wrapper parameter
    # This is a temporary solution - you might want to create a specific TeacherEnvWrapper
    
    # For now, we'll use the PVCNN wrapper without actual PVCNN (set to None)
    # Or we can create a simpler wrapper. Let me create a simple wrapper class:
    
    from rsl_rl_2_01.env import VecEnv
    
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
            from tensordict import TensorDict
            obs_dict = self.env.unwrapped.observation_manager.compute()
            if isinstance(obs_dict, dict) and not isinstance(obs_dict, TensorDict):
                td = TensorDict(obs_dict, batch_size=[self.env.unwrapped.num_envs])
                # print(f"[Debug][get_observations] converted to TensorDict, keys: {list(td.keys())}")
                # for k in td.keys():
                #     v = td[k]
                #     shape = tuple(v.shape) if hasattr(v, "shape") else "N/A"
                #     dtype = getattr(v, "dtype", type(v))
                #     print(f"  - td key={k}, shape={shape}, dtype={dtype}")
                return td
            return obs_dict
        
        def reset(self):
            obs_dict, _ = self.env.reset()
            # Return TensorDict directly, not just "policy" key
            from tensordict import TensorDict
            if isinstance(obs_dict, dict) and not isinstance(obs_dict, TensorDict):
                obs_dict = TensorDict(obs_dict, batch_size=[self.env.unwrapped.num_envs])
            return obs_dict, None
        
        def step(self, actions):
            if self.clip_actions is not None:
                actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

            obs_dict, rewards, dones, truncated, extras = self.env.step(actions)


            # Convert to TensorDict if it's a plain dict
            from tensordict import TensorDict
            if isinstance(obs_dict, dict) and not isinstance(obs_dict, TensorDict):
                # print(f"[Debug][step] Converting plain dict to TensorDict...")
                obs_dict = TensorDict(obs_dict, batch_size=[self.env.unwrapped.num_envs])
                # print(f"[Debug][step] AFTER conversion - type: {type(obs_dict)}")
                # print(f"[Debug][step] keys: {list(obs_dict.keys())}")

            # # Debug: verify TensorDict
            # if isinstance(obs_dict, TensorDict):
            #     print(f"[Debug][step] ✅ obs is TensorDict, keys: {list(obs_dict.keys())}")
            #     for k in obs_dict.keys():
            #         v = obs_dict[k]
            #         shape = tuple(v.shape) if hasattr(v, "shape") else "N/A"
            #         dtype = getattr(v, "dtype", type(v))
            #         print(f"  - key={k}, shape={shape}, dtype={dtype}")

            # Combine dones and truncated
            dones = dones | truncated

            # PPO bootstrap on timeout (for correct value estimation)
            extras["time_outs"] = truncated

            return obs_dict, rewards, dones, extras
    
    # Create wrapper
    wrapped_env = SimpleRslRlEnvWrapper(base_env, clip_actions=100.0)
    print(f"[Wrapper] Wrapper created")
    
    # ========================================
    # Create Runner Configuration
    # ========================================
    print(f"\n[Runner] Creating RSL-RL runner configuration...")
    
    from rsl_rl_2_01.runners import OnPolicyRunner
    
    # Training configuration from agent module
    train_cfg = get_train_cfg(experiment_name)
    
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
            runs = [d for d in os.listdir(log_root_path) if os.path.isdir(os.path.join(log_root_path, d))]
            runs.sort()
            resume_path = os.path.join(log_root_path, runs[-1]) if runs else None
        
        if resume_path is None:
            raise ValueError("No run found to resume from!")
        
        print(f"[Resume] Loading from: {resume_path}")
        
        if args_cli.load_checkpoint is not None:
            checkpoint_file = args_cli.load_checkpoint
        else:
            checkpoint_file = "model_最新.pt"
        
        checkpoint_path = os.path.join(resume_path, checkpoint_file)
        
        if os.path.exists(checkpoint_path):
            runner.load(checkpoint_path)
            print(f"[Resume] Checkpoint loaded: {checkpoint_path}")
        else:
            print(f"[Resume] WARNING: Checkpoint not found: {checkpoint_path}")
    
    # ========================================
    # Save Configuration
    # ========================================
    if rank == 0:
        # Save environment config
        env_cfg_dict = env_cfg.to_dict()
        dump_yaml(os.path.join(log_dir, "env_cfg.yaml"), env_cfg_dict)
        
        # Save training config
        dump_yaml(os.path.join(log_dir, "train_cfg.yaml"), train_cfg)
        
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
