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

# Isaac Lab imports - MUST be before AppLauncher
from isaaclab.app import AppLauncher

# Add argparse arguments
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
parser.add_argument("--distributed", action="store_true", default=False,
                    help="Enable multi-GPU training with PyTorch distributed.")

# Append AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ========================================
# GPU MAPPING FOR MULTI-GPU
# ========================================
if args_cli.distributed and "GPU_IDS" in os.environ:
    gpu_ids = [int(x.strip()) for x in os.environ["GPU_IDS"].split(",") if x.strip()]
    original_local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if original_local_rank >= len(gpu_ids):
        raise RuntimeError(
            f"LOCAL_RANK={original_local_rank} but GPU_IDS only has {len(gpu_ids)} GPUs: {os.environ['GPU_IDS']}"
        )
    
    target_gpu_id = gpu_ids[original_local_rank]
    required_offset = target_gpu_id - original_local_rank
    os.environ["GPU_OFFSET"] = str(required_offset)
    
    print(f"\n[GPU Mapping] LOCAL_RANK={original_local_rank} -> GPU {target_gpu_id}")
    print(f"[GPU Mapping] Set GPU_OFFSET={required_offset}")

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows after Isaac Sim is launched."""

import torch
import gymnasium as gym

# Add Go2Pvcnn to Python path
import sys
from pathlib import Path
go2_pvcnn_root = Path(__file__).resolve().parent.parent
if str(go2_pvcnn_root) not in sys.path:
    sys.path.insert(0, str(go2_pvcnn_root))

# Import Teacher environment
from go2_pvcnn.tasks.teacher_semantic_env_cfg import TeacherSemanticEnvCfg

# Import wrapper from local directory
from go2_pvcnn.wrapper.pvcnn_env_wrapper import RslRlPvcnnEnvWrapper

# RSL-RL-2.01 imports (from rsl-rl-2-01 package)
from rsl_rl_2_01.runners import OnPolicyRunner

# Isaac Lab utilities
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

# Configure PyTorch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# CRITICAL: Set memory allocator
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main():
    """Main training function."""
    
    # ========================================
    # Multi-GPU Setup
    # ========================================
    if args_cli.distributed:
        import torch.distributed as dist
        
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            raise RuntimeError("Distributed mode enabled but RANK/WORLD_SIZE not set. "
                             "Use: python -m torch.distributed.run --nproc_per_node=N script.py --distributed")
        
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
    env_cfg = TeacherSemanticEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = f"cuda:{app_launcher.device_id}"
    
    if args_cli.distributed:
        env_cfg.seed = args_cli.seed + app_launcher.local_rank
    else:
        env_cfg.seed = args_cli.seed
    
    # ========================================
    # Setup Logging Directory
    # ========================================
    experiment_name = "teacher_semantic"
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
    print(f"\n[Env] Creating Teacher Semantic Environment...")
    print(f"  - num_envs: {env_cfg.scene.num_envs}")
    print(f"  - device: {env_cfg.sim.device}")
    print(f"  - seed: {env_cfg.seed}")
    
    # Create gym environment
    env = gym.make("Isaac-Teacher-Semantic-Go2-v0", cfg=env_cfg)
    
    # Cast to ManagerBasedRLEnv for type safety
    assert isinstance(env.unwrapped, ManagerBasedRLEnv)
    base_env: ManagerBasedRLEnv = env.unwrapped
    
    print(f"[Env] Environment created successfully")
    print(f"  - observation_space: {env.observation_space}")
    print(f"  - action_space: {env.action_space}")
    
    # ========================================
    # Wrap Environment for RSL-RL
    # ========================================
    print(f"\n[Wrapper] Creating RSL-RL environment wrapper...")
    
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
            # Convert to TensorDict if it's a plain dict
            if isinstance(obs_dict, dict) and not isinstance(obs_dict, TensorDict):
                return TensorDict(obs_dict, batch_size=[self.env.unwrapped.num_envs])
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
            
            # Combine dones and truncated
            dones = dones | truncated
            
            # Return TensorDict directly
            from tensordict import TensorDict
            if isinstance(obs_dict, dict) and not isinstance(obs_dict, TensorDict):
                obs_dict = TensorDict(obs_dict, batch_size=[self.env.unwrapped.num_envs])
            
            return obs_dict, rewards, dones, extras
    
    # Create wrapper
    wrapped_env = SimpleRslRlEnvWrapper(base_env, clip_actions=100.0)
    print(f"[Wrapper] Wrapper created")
    
    # ========================================
    # Create Runner Configuration
    # ========================================
    print(f"\n[Runner] Creating RSL-RL runner configuration...")
    
    from rsl_rl_2_01.runners import OnPolicyRunner
    
    # Training configuration (dict-based for rsl_rl_2_01)
    train_cfg = {
        # Rollout settings
        "num_steps_per_env": 24,
        "save_interval": 100,  # Save checkpoint every 100 iterations
        
        # Algorithm configuration
        "algorithm": {
            "class_name": "PPO",  # PPO algorithm class
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-3,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "rnd_cfg": None,
            "symmetry_cfg": None,
            # Note: multi_gpu_cfg is passed separately by runner
        },
        
        # Policy configuration (use ActorCriticCNN for cost map + state)
        "policy": {
            "class_name": "ActorCriticCNN",  # CNN for cost map, MLP for state
            "init_noise_std": 1.0,  # Initial noise std for exploration
            "noise_std_type": "log",  # Use log std to ensure std > 0
            "state_dependent_std": False,  # Use fixed std, not state-dependent
            # Actor CNN config for processing cost map (15x15 input, 2 channels)
            "actor_cnn_cfg": {
                "output_channels": [32, 64],  # Conv layer output channels: 减少层数适应15x15
                "kernel_size": [3, 3],  # Kernel sizes: 使用较小的kernel
                "stride": [1, 1],  # Strides: stride=1避免过快缩小
                "padding": "zeros",  # 自动padding保持尺寸
                "max_pool": [True, True],  # MaxPool after each conv: 15->7->3
                "activation": "elu",
                "flatten": True,  # Flatten before passing to MLP
            },
            # Critic CNN config (same as actor)
            "critic_cnn_cfg": {
                "output_channels": [32, 64],
                "kernel_size": [3, 3],
                "stride": [1, 1],
                "padding": "zeros",
                "max_pool": [True, True],
                "activation": "elu",
                "flatten": True,
            },
            # MLP config for combined features (CNN output + state)
            "actor_hidden_dims": [256, 128],
            "critic_hidden_dims": [256, 128],
            "activation": "elu",
        },
        
        # Observation groups (dict mapping set_name to list of groups)
        "obs_groups": {
            "policy": ["policy_cost_map", "policy_state"],  # Cost map + state for actor
            "critic": ["critic_cost_map", "critic_state"],  # Cost map + state for critic
        },
    }
    
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
    print(f"Starting Training - Teacher Semantic Mode")
    print(f"{'='*80}\n")
    
    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}\n")
    
    # ========================================
    # Cleanup
    # ========================================
    env.close()
    if args_cli.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Run main training
    main()
    
    # Close simulation app
    simulation_app.close()
