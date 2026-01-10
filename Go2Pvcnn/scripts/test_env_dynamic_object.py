#!/usr/bin/env python3
"""
Test dynamic object detection in LiDAR point clouds.

This script tests how Isaac Sim detects dynamic obstacles in the LiDAR sensor.
It loads a trained policy to move the robot and visualizes point cloud detection.

Usage:
    python test_env_dynamic_object.py \
        --checkpoint /path/to/model_999.pt \
        --num_envs 1 --num_steps 500 --headless
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test dynamic object detection in LiDAR point clouds.")
parser.add_argument(
    "--checkpoint", 
    type=str, 
    required=True,
    help="Path to trained model checkpoint (e.g., model_999.pt)"
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of test environments.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of steps to run test.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--use_furniture", action="store_true", help="Use furniture objects (Sofa, Armchair, Table) instead of YCB objects.")
parser.add_argument("--save_images", action="store_true", help="Save camera images during testing.")
parser.add_argument("--image_interval", type=int, default=50, help="Save camera image every N steps.")
parser.add_argument("--image_dir", type=str, default="./test_images", help="Directory to save camera images.")

# Append AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after AppLauncher
import gymnasium as gym

# Import custom modules
from go2_pvcnn.tasks import Go2PvcnnTestEnvCfg, Go2PvcnnFurnitureTestEnvCfg
from go2_pvcnn.pvcnn_wrapper import create_pvcnn_wrapper
from go2_pvcnn.wrapper import RslRlPvcnnEnvWrapper

# Import RSL-RL
from rsl_rl.modules import ActorCriticCNN


def main():
    """Main test function."""
    
    # Create test environment - select based on --use_furniture flag
    if args_cli.use_furniture:
        env_cfg = Go2PvcnnFurnitureTestEnvCfg()
        print(f"\n🪑 Using FURNITURE objects: Sofa, Armchair, TableA")
    else:
        env_cfg = Go2PvcnnTestEnvCfg()
        print(f"\n📦 Using YCB objects: CrackerBox, SugarBox, TomatoSoupCan")
    
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    print(f"\n🎯 Testing Dynamic Object Detection in LiDAR")
    print(f"   Device: {env_cfg.sim.device}")
    print(f"   Environments: {args_cli.num_envs}")
    print(f"   Steps: {args_cli.num_steps}")
    
    # Create PVCNN Wrapper
    pvcnn_checkpoint = "/mnt/mydisk/lhy/testPvcnnWithIsaacsim/pvcnn/runs/s3dis.pvcnn.area5.c1/best.pth.tar"
    print(f"\n🔧 Initializing PVCNN wrapper...")
    pvcnn_wrapper = create_pvcnn_wrapper(
        checkpoint_path=pvcnn_checkpoint,
        device=env_cfg.sim.device,
        num_points=2046,
        max_batch_size=64
    )
    
    # Create Environment
    print(f"\n🌍 Creating test environment...")
    env = gym.make("Go2PvcnnEnv-Test", cfg=env_cfg, render_mode=None)
    env = RslRlPvcnnEnvWrapper(env, pvcnn_wrapper)
    
    # Load policy
    print(f"\n🔄 Loading policy from: {args_cli.checkpoint}")
    if not os.path.exists(args_cli.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args_cli.checkpoint}")
    
    checkpoint = torch.load(args_cli.checkpoint, map_location=env.device)
    obs, _ = env.reset()
    
    # Get dimensions
    obs_dim = obs.shape[1] if len(obs.shape) > 1 else obs.shape[0]
    critic_obs_dim = obs_dim
    act_dim = env.unwrapped.action_manager.total_action_dim
    
    # Create policy
    actor_critic = ActorCriticCNN(
        num_actor_obs=obs_dim,
        num_critic_obs=critic_obs_dim,
        num_actions=act_dim,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        use_cost_map=True,
        cost_map_channels=1,
        cost_map_size=16,
        cnn_channels=[32, 64],
        cnn_feature_dim=128,
    ).to(env.device)
    
    policy_state_dict = checkpoint.get("model_state_dict", checkpoint)
    policy_state_dict.pop("std", None)
    actor_critic.load_state_dict(policy_state_dict, strict=False)
    actor_critic.eval()
    
    print(f"✅ Setup complete!")
    
    # Create image directory if saving images
    if args_cli.save_images:
        os.makedirs(args_cli.image_dir, exist_ok=True)
        print(f"📁 Camera images will be saved to: {args_cli.image_dir}")
        print(f"   Saving every {args_cli.image_interval} steps")
    
    # Access LiDAR sensor
    unwrapped_env = env.unwrapped
    lidar_sensor = unwrapped_env.scene.sensors.get("lidar_sensor")
    
    if lidar_sensor is None:
        print("❌ LiDAR sensor not found!")
        return
    
    print(f"\n📡 LiDAR Sensor Info:")
    print(f"   Sensor type: {type(lidar_sensor)}")
    print(f"   Max distance: {lidar_sensor.cfg.max_distance}m")
    print(f"   Min range: {lidar_sensor.cfg.min_range}m")
    
    # Run test and analyze point clouds
    print(f"\n🚀 Starting dynamic object detection test...")
    
    for step in range(args_cli.num_steps):
        # Get action and step
        with torch.no_grad():
            actions = actor_critic.act_inference(obs)
        obs, rewards, dones, extras = env.step(actions)
        
        # Save camera images if enabled
        if args_cli.save_images and step % args_cli.image_interval == 0:
            camera = unwrapped_env.scene.sensors["camera"]
            rgb_data = camera.data.output["rgb"][0].cpu().numpy()  # [H, W, 3], float32
            rgb_uint8 = (np.clip(rgb_data, 0, 1) * 255).astype(np.uint8)
            image = Image.fromarray(rgb_uint8, mode="RGB")
            image_path = os.path.join(args_cli.image_dir, f"step_{step:04d}.png")
            image.save(image_path)
            if step % (args_cli.image_interval * 4) == 0:  # Print every 4th save
                print(f"💾 Saved image: {image_path}")
        
        # Analyze point cloud every 50 steps
        if (step + 1) % 50 == 0:
            # Get point cloud data
            pc_data = lidar_sensor.data.pos_w[0]  # First environment (num_points, 3)
            num_points = (pc_data.norm(dim=-1) > 0).sum().item()
            
            # Get robot position
            robot_pos = unwrapped_env.scene["robot"].data.root_pos_w[0]
            
            # Calculate point cloud statistics
            if num_points > 0:
                valid_points = pc_data[pc_data.norm(dim=-1) > 0]
                distances = (valid_points - robot_pos).norm(dim=-1)
                
                print(f"\n  Step {step+1}: Point Cloud Analysis")
                print(f"    Valid points: {num_points}")
                print(f"    Distance range: {distances.min():.2f}m - {distances.max():.2f}m")
                print(f"    Average distance: {distances.mean():.2f}m")
    
    print(f"\n✅ Test completed!")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
