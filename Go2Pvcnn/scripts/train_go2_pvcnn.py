#!/usr/bin/env python3
"""
Train Go2 robot with PVCNN-based perception using RSL-RL PPO.

Multi-GPU support following Isaac Lab official pattern:
- Use `python -m torch.distributed.run --nproc_per_node=2 train_go2_pvcnn.py --distributed`
- Devices are assigned via app_launcher.local_rank instead of CUDA_VISIBLE_DEVICES
- Each rank uses app_launcher.local_rank to get its GPU ID

Usage:
    Single GPU:
        python train_go2_pvcnn.py --num_envs 256 --headless
    
    Multi-GPU (2 GPUs):
        python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \\
            train_go2_pvcnn.py --num_envs 512 --headless --distributed
"""

import argparse
import os
import sys
from datetime import datetime

# Isaac Lab imports - MUST be before AppLauncher
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train Go2 robot with PVCNN using RSL-RL PPO.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
parser.add_argument("--max_iterations", type=int, default=5000, help="Maximum training iterations.")
parser.add_argument("--video", action="store_true", default=False, help="Record training videos.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos (steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between recordings (steps).")
parser.add_argument("--checkpoint_path", type=str, 
                    default="/mnt/mydisk/lhy/testPvcnnWithIsaacsim/pvcnn/runs/s3dis.pvcnn.area5.c1/best.pth.tar",
                    help="Path to PVCNN checkpoint.")
parser.add_argument("--num_points", type=int, default=2046, help="Number of points for PVCNN input.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume training from checkpoint.")
parser.add_argument("--load_run", type=str, default=None, help="Name of run to load when resuming.")
parser.add_argument("--load_checkpoint", type=str, default=None, help="Checkpoint file to load.")
parser.add_argument("--enable_pvcnn_sync", action="store_true", default=False, 
                    help="Enable synchronous PVCNN training (train PVCNN after PPO updates).")
parser.add_argument("--pvcnn_train_interval", type=int, default=10,
                    help="Train PVCNN every N PPO iterations.")
parser.add_argument("--pvcnn_train_epochs", type=int, default=5,
                    help="Number of epochs for each PVCNN training update.")
parser.add_argument("--distributed", action="store_true", default=False,
                    help="Enable multi-GPU training with PyTorch distributed.")

# Append AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ========================================
# ISAAC LAB OFFICIAL MULTI-GPU PATTERN
# ========================================
# Following /mnt/mydisk/lhy/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py
# Read from environment variable (set by launch script) or default to 0
GPU_OFFSET = int(os.environ.get("GPU_OFFSET", 0))

if args_cli.distributed:
    # Get local rank from app_launcher (set by torch.distributed.run via LOCAL_RANK env var)
    # This will be 0, 1, 2, ... for each process
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Apply GPU offset to skip occupied GPUs
    actual_gpu_id = local_rank + GPU_OFFSET
    
    # Set device using actual GPU ID
    # This tells Isaac Sim which GPU to use for this process
    args_cli.device = f"cuda:{actual_gpu_id}"
    
    print(f"\n[Multi-GPU] Rank {local_rank} using physical GPU {actual_gpu_id}: {args_cli.device}")

# Launch Isaac Sim - AppLauncher will use args_cli.device
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows after Isaac Sim is launched."""

import torch
import gymnasium as gym

# Import Go2 PVCNN components
from go2_pvcnn.tasks import Go2PvcnnEnvCfg
from go2_pvcnn.pvcnn_wrapper import create_pvcnn_wrapper
from go2_pvcnn.wrapper import RslRlPvcnnEnvWrapper

# RSL-RL imports
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCriticCNN  # CNN-based policy for cost map processing

# Isaac Lab utilities
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

# Configure PyTorch for stability and performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False  # Disable benchmark for stability

# CRITICAL: Set memory allocator for better cross-GPU stability
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main():
    """Main training function."""
    
    # ========================================
    # Multi-GPU Setup (Isaac Lab Official Pattern)
    # ========================================
    if args_cli.distributed:
        # Initialize PyTorch distributed
        import torch.distributed as dist
        
        # Torchrun sets these environment variables automatically
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            raise RuntimeError("Distributed mode enabled but RANK/WORLD_SIZE not set. "
                             "Use: python -m torch.distributed.run --nproc_per_node=N script.py --distributed")
        
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = app_launcher.local_rank  # Use AppLauncher's local_rank
        
        print(f"[Multi-GPU] Global Rank: {rank}/{world_size}, Local Rank: {local_rank}")
        print(f"[Multi-GPU] Device: {args_cli.device}")
        
        # Initialize process group
        dist.init_process_group(backend="nccl", init_method="env://")
        
        # Set CUDA device for this process (with GPU offset)
        device_id = local_rank + GPU_OFFSET
        torch.cuda.set_device(device_id)
        
        # Divide environments evenly across GPUs
        envs_per_gpu = args_cli.num_envs // world_size
        args_cli.num_envs = envs_per_gpu
        print(f"[Multi-GPU] Adjusted to {envs_per_gpu} envs per GPU ({envs_per_gpu * world_size} total)")
    else:
        # Single GPU mode (also skip GPU 0)
        rank = 0
        world_size = 1
        device_id = GPU_OFFSET  # Use GPU 1 by default
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
    env_cfg = Go2PvcnnEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # ISAAC LAB OFFICIAL PATTERN: Use app_launcher.local_rank for device assignment
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank + GPU_OFFSET}"
        env_cfg.seed = args_cli.seed + app_launcher.local_rank  # Different seed per rank
    else:
        env_cfg.sim.device = f"cuda:{GPU_OFFSET}"  # Single GPU mode also uses GPU 1
        env_cfg.seed = args_cli.seed
    
    
    # ========================================
    # Setup Logging Directory
    # ========================================
    experiment_name = "go2_pvcnn"
    log_root_path = os.path.join("logs", "rsl_rl", experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    if rank == 0:
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_root_path, log_dir)
        os.makedirs(log_dir, exist_ok=True)
        print(f"[Logging] Directory: {log_dir}")
        
        # Share log_dir with other ranks in multi-GPU mode
        if args_cli.distributed:
            temp_log_path = "/tmp/go2_pvcnn_log_dir.txt"
            with open(temp_log_path, "w") as f:
                f.write(log_dir)
    
    # Synchronize across all processes
    if args_cli.distributed:
        dist.barrier()  # Wait for rank 0 to create directory
        if rank != 0:
            temp_log_path = "/tmp/go2_pvcnn_log_dir.txt"
            with open(temp_log_path, "r") as f:
                log_dir = f.read().strip()
    else:
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_root_path, log_dir)
        os.makedirs(log_dir, exist_ok=True)
    
    # ========================================
    # Create PVCNN Wrapper
    # ========================================
    if rank == 0:
        print(f"\n[PVCNN] Initializing with checkpoint: {args_cli.checkpoint_path}")
    # CRITICAL: PVCNN must use SAME device as Isaac Sim
    
    
    pvcnn_wrapper = create_pvcnn_wrapper(
        checkpoint_path=args_cli.checkpoint_path,
        device=env_cfg.sim.device,
        num_points=2046,
        max_batch_size=64
    )
    
    
    
    # ========================================
    # Create Environment
    # ========================================
    if rank == 0:
        print(f"\n[Environment] Creating {args_cli.num_envs} parallel environments...")
    
    env = gym.make("Go2PvcnnEnv", cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Wrap for video recording if requested
    if args_cli.video and rank == 0:  # Only master records video
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        if rank == 0:
            print("[Video] Recording enabled")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Wrap environment for RSL-RL with PVCNN
    env = RslRlPvcnnEnvWrapper(env, pvcnn_wrapper)
    
    # ========================================
    # Create RSL-RL Agent Configuration
    # ========================================
    agent_cfg = {
        "seed": args_cli.seed + rank,  # Different seed per rank
        "device": str(env_cfg.sim.device),
        "num_steps_per_env": 24,
        "max_iterations": args_cli.max_iterations if args_cli.max_iterations is not None else 1000,
        "save_interval": 50,
        "empirical_normalization": False,
        "logger": "tensorboard",
        "log_interval": 1,
        "obs_groups": {
            "policy": ["policy"],
            "critic": ["critic"],
        },
        # Multi-GPU settings
        "distributed": args_cli.distributed,
        "rank": rank,
        "world_size": world_size,
        # PVCNN sync training
        "enable_pvcnn_sync_training": args_cli.enable_pvcnn_sync,
        "pvcnn_train_interval": args_cli.pvcnn_train_interval,
        "pvcnn_train_epochs": args_cli.pvcnn_train_epochs,
        # Policy configuration
        "policy": {
            "class_name": "ActorCriticCNN",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 256, 256],
            "critic_hidden_dims": [256, 256, 256],
            "activation": "elu",
            "use_cost_map": True,
            "cost_map_channels": 1,  # 单通道 cost_map
            "cost_map_size": 16,  # 16×16 网格（匹配 height_scanner）
            "cnn_channels": [32, 64],  # 减少层数以适应小尺寸输入
            "cnn_feature_dim": 128,
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "pvcnn_model": pvcnn_wrapper.model,  # 直接传入PVCNN模型
            # Multi-GPU parameters (passed to PPO.__init__)
            "distributed": args_cli.distributed,
            "rank": rank,
            "world_size": world_size,
        },
    }
    
    if rank == 0:
        print("\n" + "="*80)
        print("Configuration Summary")
        print("="*80)
        print(f"Environments: {env.num_envs}")
        print(f"Actions: {env.num_actions}")
        print(f"Episode length: {env.max_episode_length}")
        print(f"Device: {env.device}")
        print(f"Learning rate: {agent_cfg['algorithm']['learning_rate']}")
        print(f"Max iterations: {agent_cfg['max_iterations']}")
        print(f"PVCNN sync: {agent_cfg['enable_pvcnn_sync_training']}")
        print("="*80 + "\n")
    
    # ========================================
    # Create RSL-RL Runner
    # ========================================
    # PVCNN模型通过agent_cfg["algorithm"]["pvcnn_model"]传入
    # PPO.__init__会自动创建pvcnn_optimizer
    runner = OnPolicyRunner(env, agent_cfg, log_dir=log_dir, device=agent_cfg["device"])
    
    # ========================================
    # 🔧 Multi-GPU Parameter Synchronization
    # ========================================
    if args_cli.distributed:
        if rank == 0:
            print("\n[Multi-GPU] Synchronizing model parameters across GPUs...")
        
        # Synchronize ActorCritic parameters from rank 0 to all ranks
        for param in runner.alg.actor_critic.parameters():
            dist.broadcast(param.data, src=0)
        
        # Synchronize ActorCritic buffers (e.g., BatchNorm running stats)
        for buffer in runner.alg.actor_critic.buffers():
            dist.broadcast(buffer, src=0)
        
        # Synchronize PVCNN parameters from rank 0 to all ranks
        if pvcnn_wrapper.model is not None:
            for param in pvcnn_wrapper.model.parameters():
                dist.broadcast(param.data, src=0)
            
            for buffer in pvcnn_wrapper.model.buffers(): #buffers like running_mean, running_var
                dist.broadcast(buffer, src=0)
        
        # Synchronize observation normalizer if used
        if agent_cfg.get("empirical_normalization", False):
            for param in runner.obs_normalizer.parameters():
                dist.broadcast(param.data, src=0)
            for buffer in runner.obs_normalizer.buffers():
                dist.broadcast(buffer, src=0)
            
            for param in runner.critic_obs_normalizer.parameters():
                dist.broadcast(param.data, src=0)
            for buffer in runner.critic_obs_normalizer.buffers():
                dist.broadcast(buffer, src=0)
        
        # Wait for all processes to complete synchronization
        dist.barrier()
        
        if rank == 0:
            print("[Multi-GPU] ✅ All model parameters synchronized!")
            
        # Verification: Check if parameters match across GPUs
        if rank == 0:
            test_param = next(runner.alg.actor_critic.parameters()).flatten()[:5]
            print(f"[Rank 0] Sample params: {test_param.detach().cpu().numpy()}")
        elif rank == 1:
            test_param = next(runner.alg.actor_critic.parameters()).flatten()[:5]
            print(f"[Rank 1] Sample params: {test_param.detach().cpu().numpy()}")
        
        dist.barrier()
    
    # Save configuration
    if rank == 0:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    
    # Load checkpoint if resuming
    if args_cli.resume and rank == 0:
        if args_cli.load_run is not None:
            resume_path = os.path.join(log_root_path, args_cli.load_run)
            if args_cli.load_checkpoint is not None:
                resume_path = os.path.join(resume_path, args_cli.load_checkpoint)
            else:
                resume_path = os.path.join(resume_path, "model_*.pt")
            print(f"[Resume] Loading checkpoint from: {resume_path}")
            runner.load(resume_path)
        else:
            print("[Resume] WARNING: --resume flag set but no --load_run specified")
    
    # ========================================
    # Start Training
    # ========================================
    if rank == 0:
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        print(f"Log directory: {log_dir}")
        print(f"\n📊 TensorBoard command:")
        print(f"   tensorboard --logdir={log_dir}")
        print(f"\n   Or view all experiments:")
        print(f"   tensorboard --logdir={log_root_path}")
        print("="*80 + "\n")
    
    runner.learn(num_learning_iterations=agent_cfg["max_iterations"], init_at_random_ep_len=True)
    
    # Close environment
    env.close()
    
    if rank == 0:
        print("\n[SUCCESS] Training completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close simulation
        simulation_app.close()
