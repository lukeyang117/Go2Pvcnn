"""Script to play a trained teacher policy.

Usage:
    # teacher_semantic (default)
    python play.py --run_dir 2026-03-15_12-00-00 --checkpoint model_1600.pt

    # teacher_without_semantic (state-only)
    python play.py --experiment teacher_without_semantic --run_dir 2026-03-15_12-00-00
"""

import argparse
import os
import sys
import torch
import numpy as np

# Add Go2Pvcnn to Python path
go2_pvcnn_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if go2_pvcnn_path not in sys.path:
    sys.path.insert(0, go2_pvcnn_path)

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained teacher policy")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (steps)")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (steps)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate")
parser.add_argument("--checkpoint", type=str, default="model_1600.pt", help="Checkpoint file name")
parser.add_argument("--run_dir", type=str, default=None, help="Run directory name (required)")
parser.add_argument(
    "--experiment",
    type=str,
    default="teacher_semantic",
    choices=["teacher_semantic", "teacher_without_semantic"],
    help="Experiment/task: teacher_semantic (CNN+state) or teacher_without_semantic (state-only)",
)
parser.add_argument("--sample", action="store_true", default=False, help="Sample actions instead of using policy")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Validate required arguments
if args_cli.run_dir is None:
    raise ValueError("--run_dir is required. Specify the training run directory to load checkpoint from.")

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from rsl_rl_2_01.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict

# Import env configs and agent
from go2_pvcnn.tasks.teacher_semantic_env_cfg import TeacherSemanticEnvCfg_PLAY
from go2_pvcnn.tasks.teacher_without_semantic_env_cfg import TeacherWithoutSemanticEnvCfg_PLAY
from agent import get_train_cfg

# Import VecEnv for creating wrapper
from rsl_rl_2_01.env import VecEnv
from tensordict import TensorDict


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
        obs_dict = self.env.unwrapped.observation_manager.compute()
        # Convert to TensorDict if it's a plain dict
        if isinstance(obs_dict, dict) and not isinstance(obs_dict, TensorDict):
            return TensorDict(obs_dict, batch_size=[self.env.unwrapped.num_envs])
        return obs_dict
    
    def reset(self):
        obs_dict, _ = self.env.reset()
        # Return TensorDict directly, not just "policy" key
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
        if isinstance(obs_dict, dict) and not isinstance(obs_dict, TensorDict):
            obs_dict = TensorDict(obs_dict, batch_size=[self.env.unwrapped.num_envs])
        
        return obs_dict, rewards, dones, extras


# Experiment -> (env_cfg_cls, gym_task_id)
EXPERIMENT_PLAY_MAP = {
    "teacher_semantic": (TeacherSemanticEnvCfg_PLAY, "Isaac-Teacher-Semantic-Go2-Play-v0"),
    "teacher_without_semantic": (TeacherWithoutSemanticEnvCfg_PLAY, "Isaac-Teacher-Without-Semantic-Go2-Play-v0"),
}


def main():
    """Play with trained policy."""
    experiment_name = args_cli.experiment
    env_cfg_cls, task_id = EXPERIMENT_PLAY_MAP[experiment_name]

    # Setup logging directory and checkpoint path
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
    log_dir = os.path.join(log_root_path, args_cli.run_dir)
    checkpoint_path = os.path.join(log_dir, args_cli.checkpoint)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\n{'='*80}")
    print(f"Playing - {experiment_name}")
    print(f"{'='*80}")
    print(f"Task: {task_id}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"{'='*80}\n")

    # Create environment configuration
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # Enable video recording if requested
    if args_cli.video:
        env_cfg.sim.enable_cameras = True
        print(f"[Video] Recording enabled (length={args_cli.video_length})")

    # Create environment
    env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": f"model_{args_cli.checkpoint.split('_')[-1].split('.')[0]}",
        }
        print("[INFO] Recording video during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Cast to ManagerBasedRLEnv
    assert isinstance(env.unwrapped, ManagerBasedRLEnv)
    base_env: ManagerBasedRLEnv = env.unwrapped
    
    # Wrap environment for RSL-RL
    print(f"\n[Wrapper] Creating RSL-RL environment wrapper...")
    wrapped_env = SimpleRslRlEnvWrapper(base_env, clip_actions=100.0)
    
    # Print environment info
    print(f"\n[Environment] Created successfully")
    print(f"  - Observation space: {wrapped_env.observation_space}")
    print(f"  - Action space: {wrapped_env.action_space}")
    print(f"  - Device: {wrapped_env.device}")
    
    # Training config from agent (must match training experiment)
    train_cfg = get_train_cfg(experiment_name)

    # Create runner
    print(f"\n[Runner] Creating OnPolicyRunner...")
    runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=None, device=env_cfg.sim.device)
    
    # Load checkpoint
    print(f"\n[Checkpoint] Loading model from: {checkpoint_path}")
    runner.load(checkpoint_path, load_optimizer=False)
    print(f"[Policy] Loaded successfully")
    
    # Get inference policy
    if args_cli.sample:
        policy = runner.alg.policy.act
    else:
        policy = runner.get_inference_policy(device=wrapped_env.device)
    
    print(f"[Policy] Using {'sampling' if args_cli.sample else 'inference'} mode")
    
    # Reset environment
    obs, _ = wrapped_env.get_observations(), None
    timestep = 0
    
    print(f"\n{'='*80}")
    print(f"Starting Play Loop")
    print(f"{'='*80}\n")
    
    try:
        while simulation_app.is_running():
            # Run in inference mode
            with torch.inference_mode():
                # Get action from policy
                actions = policy(obs)
                
                # Step environment
                obs, rewards, dones, extras = wrapped_env.step(actions)
            
            timestep += 1
            
            # Update camera to follow robot (only for single env)
            if args_cli.num_envs == 1:
                # Get robot position
                robot_pos = base_env.scene["robot"].data.root_pos_w[0].cpu().numpy()
                
                # Camera offset from robot (behind and above)
                camera_direction = np.array([3.0, 0.0, 0.0])  # behind the robot
                camera_position = robot_pos - camera_direction + np.array([0.0, 0.0, 1.5])  # offset up
                target_position = robot_pos  # look at robot
                
                # Update camera view to follow robot
                base_env.sim.set_camera_view(camera_position, target_position)
            
            # Exit if video recording is complete
            if args_cli.video and timestep == args_cli.video_length:
                break
    
    except KeyboardInterrupt:
        print("\n[Play] Interrupted by user")
    
    finally:
        # Close environment
        wrapped_env.env.close()
        print(f"\n{'='*80}")
        print(f"Play Complete - Timesteps: {timestep}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    # Run play script
    main()
    
    # Close simulation app
    simulation_app.close()
