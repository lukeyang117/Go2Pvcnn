#!/usr/bin/env python3
"""
Test trained Go2 PVCNN policy with collision detection and statistics.

This script:
1. Loads a trained model checkpoint (PPO policy + PVCNN)
2. Creates PVCNN wrapper for semantic segmentation
3. Runs the policy in test environments with mixed terrain (flat + stairs)
4. Uses PVCNN to generate cost maps for navigation
5. Detects and counts collisions between robot and USD objects
6. Provides statistics on collision rates, success rates, and terrain performance

Usage:
    python test_go2_pvcnn_collision.py \
        --checkpoint /path/to/model_999.pt \
        --pvcnn_checkpoint /path/to/pvcnn/best.pth.tar \
        --num_envs 4 --num_steps 500 --headless
"""

import argparse
import os
import sys
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test Go2 PVCNN policy with collision detection.")
parser.add_argument(
    "--checkpoint", 
    type=str, 
    required=True,
    help="Path to trained model checkpoint (e.g., model_999.pt)"
)
parser.add_argument(
    "--pvcnn_checkpoint",
    type=str,
    default="/mnt/mydisk/lhy/testPvcnnWithIsaacsim/pvcnn/runs/s3dis.pvcnn.area5.c1/best.pth.tar",
    help="Path to PVCNN checkpoint for semantic segmentation"
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of test environments.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of steps to run test.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--save_images", action="store_true", help="Save camera images during testing.")
parser.add_argument("--image_interval", type=int, default=50, help="Save camera image every N steps.")
parser.add_argument("--image_dir", type=str, default="./test_images", help="Directory to save camera images.")

# Append AppLauncher arguments (this includes --headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator with parsed arguments (includes headless mode)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after AppLauncher
import isaaclab.envs.mdp as isaac_mdp
import yaml
import json

# Import custom modules
from go2_pvcnn.tasks import Go2PvcnnTestEnvCfg
from go2_pvcnn.pvcnn_wrapper import create_pvcnn_wrapper
from go2_pvcnn.wrapper import RslRlPvcnnEnvWrapper

# Import RSL-RL
from rsl_rl.modules import ActorCritic, ActorCriticCNN

# Import gym for wrapper
import gymnasium as gym


class CollisionDetector:
    """Tracks and reports collisions between robot and USD objects."""
    
    def __init__(self, env):
        # Support both wrapped and unwrapped environments
        self.env = env.unwrapped if hasattr(env, 'unwrapped') else env
        self.num_envs = self.env.num_envs
        
        # Collision counters
        self.total_steps = 0
        self.collision_counts = defaultdict(lambda: defaultdict(int))  # {env_id: {object_name: count}}
        self.terrain_steps = defaultdict(int)  # {terrain_type: steps}
        self.terrain_collisions = defaultdict(int)  # {terrain_type: collisions}
        
        # Per-step collision tracking
        self.current_collisions = set()  # Set of (env_id, object_name) tuples
        
        # Object sensors (8 objects per environment)
        self.object_sensors = {
            "cracker_box_0": "object_0_contact",
            "sugar_box_1": "object_1_contact",
            "tomato_soup_can_2": "object_2_contact",
            "cracker_box_3": "object_3_contact",
            "sugar_box_4": "object_4_contact",
            "tomato_soup_can_5": "object_5_contact",
            "cracker_box_6": "object_6_contact",
            "sugar_box_7": "object_7_contact",
        }
        
        # Body-ground collision sensor
        self.body_ground_sensor = "body_ground_contact"
        self.body_ground_collisions = 0
        
    def detect_collisions(self):
        """Detect collisions at current timestep."""
        new_collisions = []
        
        # Detect object collisions
        for obj_name, sensor_name in self.object_sensors.items():
            if sensor_name not in self.env.scene.sensors:
                continue
                
            sensor = self.env.scene.sensors[sensor_name]
            
            # Get contact forces - Shape: (num_envs, num_bodies, 3)
            contact_forces = sensor.data.net_forces_w
            
            # Check if any body has contact force above threshold
            # Sum over XYZ dimensions and check if magnitude > threshold
            force_magnitudes = torch.norm(contact_forces, dim=-1)  # (num_envs, num_bodies)
            has_collision = (force_magnitudes > 0.1).any(dim=-1)  # (num_envs,)
            
            # Record collisions
            env_ids_with_collision = torch.where(has_collision)[0]
            for env_id in env_ids_with_collision:
                env_id_int = env_id.item()
                collision_key = (env_id_int, obj_name)
                
                # Only count new collisions (not continuous contact)
                if collision_key not in self.current_collisions:
                    self.collision_counts[env_id_int][obj_name] += 1
                    new_collisions.append((env_id_int, obj_name))
                    self.current_collisions.add(collision_key)
        
        # Detect body-ground collisions
        if self.body_ground_sensor in self.env.scene.sensors:
            sensor = self.env.scene.sensors[self.body_ground_sensor]
            contact_forces = sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
            force_magnitudes = torch.norm(contact_forces, dim=-1)
            has_body_collision = (force_magnitudes > 1.0).any(dim=-1)  # Higher threshold for ground
            
            body_collision_count = has_body_collision.sum().item()
            if body_collision_count > 0:
                self.body_ground_collisions += body_collision_count
                env_ids = torch.where(has_body_collision)[0]
                for env_id in env_ids:
                    new_collisions.append((env_id.item(), "body_ground"))
        
        # Remove ended collisions
        self.current_collisions = {
            (env_id, obj_name) 
            for env_id, obj_name in self.current_collisions
            if self._check_ongoing_collision(env_id, obj_name)
        }
        
        return new_collisions
    
    def _check_ongoing_collision(self, env_id: int, obj_name: str) -> bool:
        """Check if collision is still ongoing."""
        # Handle body-ground collisions
        if obj_name == "body_ground":
            if self.body_ground_sensor not in self.env.scene.sensors:
                return False
            sensor = self.env.scene.sensors[self.body_ground_sensor]
            contact_forces = sensor.data.net_forces_w[env_id]
            force_magnitudes = torch.norm(contact_forces, dim=-1)
            return (force_magnitudes > 1.0).any().item()
        
        # Handle object collisions
        sensor_name = self.object_sensors.get(obj_name)
        if not sensor_name or sensor_name not in self.env.scene.sensors:
            return False
            
        sensor = self.env.scene.sensors[sensor_name]
        contact_forces = sensor.data.net_forces_w[env_id]  # (num_bodies, 3)
        force_magnitudes = torch.norm(contact_forces, dim=-1)
        return (force_magnitudes > 0.1).any().item()
    
    def update_terrain_stats(self):
        """Update terrain-based statistics."""
        # For testing, we don't track terrain-based stats since we removed curriculum
        # Just count collisions per step
        for env_id in range(self.num_envs):
            # Check if this env has collision this step  
            for obj_name in self.object_sensors.keys():
                if (env_id, obj_name) in self.current_collisions:
                    # Record as general collision (no terrain differentiation)
                    self.terrain_collisions["all"] += 1
                    break  # Count once per env per step
    
    def get_statistics(self):
        """Get collision statistics."""
        total_collisions = sum(
            sum(obj_counts.values()) 
            for obj_counts in self.collision_counts.values()
        )
        
        # Collision rates per object
        object_collision_rates = {}
        for obj_name in self.object_sensors.keys():
            obj_total = sum(
                self.collision_counts[env_id].get(obj_name, 0)
                for env_id in range(self.num_envs)
            )
            rate = obj_total / max(self.total_steps, 1) * 100  # Percentage
            object_collision_rates[obj_name] = {
                "count": obj_total,
                "rate_percent": rate
            }
        
        # Terrain-based statistics
        terrain_stats = {}
        for terrain_type in ["flat", "stairs"]:
            steps = self.terrain_steps.get(terrain_type, 0)
            collisions = self.terrain_collisions.get(terrain_type, 0)
            rate = (collisions / max(steps, 1)) * 100
            terrain_stats[terrain_type] = {
                "steps": steps,
                "collisions": collisions,
                "collision_rate_percent": rate
            }
        
        # Overall statistics
        overall_collision_rate = (total_collisions / max(self.total_steps * self.num_envs, 1)) * 100
        
        return {
            "total_steps": self.total_steps,
            "total_collisions": total_collisions,
            "overall_collision_rate_percent": overall_collision_rate,
            "per_object": object_collision_rates,
            "per_terrain": terrain_stats,
            "collisions_per_env": {
                env_id: sum(obj_counts.values())
                for env_id, obj_counts in self.collision_counts.items()
            }
        }
    
    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*80)
        print("COLLISION DETECTION TEST RESULTS")
        print("="*80)
        
        print(f"\n📊 Overall Statistics:")
        print(f"  Total Steps: {stats['total_steps']}")
        print(f"  Total Environments: {self.num_envs}")
        print(f"  Total Collisions: {stats['total_collisions']}")
        print(f"  Overall Collision Rate: {stats['overall_collision_rate_percent']:.2f}%")
        print(f"  Body-Ground Collisions: {self.body_ground_collisions} ({self.body_ground_collisions / max(stats['total_steps'], 1) * 100:.2f}%)")
        
        print(f"\n🎯 Per-Object Collision Rates:")
        for obj_name, obj_stats in stats['per_object'].items():
            print(f"  {obj_name:25s}: {obj_stats['count']:4d} collisions ({obj_stats['rate_percent']:.2f}%)")
        
        print(f"\n🏞️  Per-Terrain Performance:")
        for terrain_type, terrain_stats in stats['per_terrain'].items():
            print(f"  {terrain_type.capitalize():10s}: {terrain_stats['collisions']:4d}/{terrain_stats['steps']:6d} steps ({terrain_stats['collision_rate_percent']:.2f}% collision rate)")
        
        print(f"\n🤖 Top 5 Environments with Most Collisions:")
        sorted_envs = sorted(
            stats['collisions_per_env'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for env_id, count in sorted_envs:
            print(f"  Env {env_id:3d}: {count:4d} collisions")
        
        print("\n" + "="*80 + "\n")


def main():
    """Main test function."""
    
    # Create test environment
    env_cfg = Go2PvcnnTestEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # ========================================
    # Create PVCNN Wrapper (same as training)
    # ========================================
    print(f"\n🎯 Initializing PVCNN wrapper...")
    print(f"   PVCNN checkpoint: {args_cli.pvcnn_checkpoint}")
    print(f"   Device: {env_cfg.sim.device}")
    
    pvcnn_wrapper = create_pvcnn_wrapper(
        checkpoint_path=args_cli.pvcnn_checkpoint,
        device=env_cfg.sim.device,
        num_points=2046,
        max_batch_size=64
    )
    print(f"✅ PVCNN wrapper initialized successfully!")
    
    # ========================================
    # Create Environment with PVCNN Wrapper
    # ========================================
    print(f"\n🌍 Creating test environment...")
    env = gym.make("Go2PvcnnEnv-Test", cfg=env_cfg, render_mode=None)
    
    # Wrap environment for RSL-RL with PVCNN (same as training)
    env = RslRlPvcnnEnvWrapper(env, pvcnn_wrapper)
    print(f"✅ Environment wrapped with PVCNN successfully!")
    
    # Debug: Check what's in the scene
    unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    print(f"\n🔍 Scene entities:")
    for key in dir(unwrapped_env.scene):
        if not key.startswith('_'):
            print(f"    - {key}")
    
    # Check sensors
    if hasattr(unwrapped_env.scene, 'sensors'):
        print(f"\n🔍 Sensors in scene:")
        for sensor_name in unwrapped_env.scene.sensors.keys():
            print(f"    - {sensor_name}")
    
    # Load policy
    print(f"\n🔄 Loading policy from: {args_cli.checkpoint}")
    
    if not os.path.exists(args_cli.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args_cli.checkpoint}")
    
    # Load checkpoint
    checkpoint = torch.load(args_cli.checkpoint, map_location=env.device)
    
    # Reset environment to get initial observation
    obs, _ = env.reset()  # RslRlPvcnnEnvWrapper returns obs directly, not dict
    
    # Get dimensions from observation
    obs_dim = obs.shape[1] if len(obs.shape) > 1 else obs.shape[0]
    
    # Get critic observation dimension (usually same as policy for this task)
    critic_obs_dim = obs_dim
    
    # Get action dimension from unwrapped environment
    act_dim = env.unwrapped.action_manager.total_action_dim
    
    print(f"📐 Environment dimensions: obs={obs_dim}, critic_obs={critic_obs_dim}, act={act_dim}")
    
    # Create actor-critic CNN (matching training configuration)
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
    
    # Load weights
    policy_state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Remove 'std' key if it exists (it will be reinitialized)
    policy_state_dict.pop("std", None)
    actor_critic.load_state_dict(policy_state_dict, strict=False)
    actor_critic.eval()
    
    print(f"✅ Policy loaded successfully!")
    
    # Initialize collision detector
    collision_detector = CollisionDetector(env)
    
    # Make robot invisible in camera view using Isaac Lab/USD API
    from pxr import Usd, UsdGeom
    unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    stage = unwrapped_env.sim.stage
    for env_idx in range(unwrapped_env.num_envs):
        robot_prim_path = f"/World/envs/env_{env_idx}/Robot"
        # Set visibility to invisible for the entire robot subtree
        robot_prim = stage.GetPrimAtPath(robot_prim_path)
        if robot_prim and robot_prim.IsValid():
            imageable = UsdGeom.Imageable(robot_prim)
            if imageable:
                imageable.MakeInvisible()
            # Also set all children invisible recursively
            for child in Usd.PrimRange(robot_prim):
                child_imageable = UsdGeom.Imageable(child)
                if child_imageable:
                    child_imageable.MakeInvisible()
    print(f"🙈 Robot set to invisible in camera view")
    
    # Create image directory if saving images
    if args_cli.save_images:
        os.makedirs(args_cli.image_dir, exist_ok=True)
        print(f"📁 Camera images will be saved to: {args_cli.image_dir}")
        print(f"   Saving every {args_cli.image_interval} steps")
    
    # print(f"\n🚀 Starting test run for {args_cli.num_steps} steps...")
    # print(f"   Environments: {args_cli.num_envs}")
    # print(f"   Terrain: Mixed (50% flat, 50% stairs)")
    # print(f"   Objects per env: 8")
    # print(f"   Goal Navigation: Enabled")
    
    # Set goal positions for each environment (RELATIVE to env_origins)
    # Goal: Navigate 5 meters forward in X direction from env_origin
    unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    env_origins = unwrapped_env.scene.env_origins[:, :2]  # (num_envs, 2) - XY coordinates
    
    # Goal offset relative to origin
    goal_offset = torch.zeros((args_cli.num_envs, 2), device=env.device)
    goal_offset[:, 0] = 5.0  # 5 meters forward in X
    goal_offset[:, 1] = 0.0  # Stay centered in Y
    
    # Calculate absolute goal positions in world frame
    goal_positions = env_origins + goal_offset  # (num_envs, 2)
    
    goal_reached = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=env.device)
    goal_threshold = 0.5  # 0.5 meter threshold to consider goal reached
    
    print(f"   Goal Offset (relative): ({goal_offset[0, 0]:.1f}, {goal_offset[0, 1]:.1f})")
    print(f"   Goal Threshold: {goal_threshold} meters")
    print(f"   Env Origins (first 2): {env_origins[:2].cpu().numpy()}")
    print(f"   Goal Positions (first 2): {goal_positions[:2].cpu().numpy()}")
    
    # Run test
    for step in range(args_cli.num_steps):
        # Get current robot positions (x, y) - access through unwrapped env
        unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        robot_positions = unwrapped_env.scene["robot"].data.root_pos_w[:, :2]  # (num_envs, 2)
        robot_full_pos = unwrapped_env.scene["robot"].data.root_pos_w  # (num_envs, 3) with z
        robot_velocities = unwrapped_env.scene["robot"].data.root_lin_vel_w  # (num_envs, 3)
        robot_angular_vel = unwrapped_env.scene["robot"].data.root_ang_vel_w  # (num_envs, 3)
        
        # Calculate distance to goal
        distance_to_goal = torch.norm(robot_positions - goal_positions, dim=1)  # (num_envs,)
        
        # Check if goal reached
        newly_reached = (distance_to_goal < goal_threshold) & (~goal_reached)
        # if newly_reached.any():
        #     for env_id in torch.where(newly_reached)[0]:
        #         print(f"    ✅ Env {env_id.item()} reached goal at step {step}!")
        #         goal_reached[env_id] = True
        
        # Get action from policy
        with torch.no_grad():
            actions = actor_critic.act_inference(obs)
        
        # Step environment (RslRlPvcnnEnvWrapper returns 4 values: obs, rew, dones, extras)
        obs, rewards, dones, extras = env.step(actions)
        
        # Print velocity commands every 100 steps for debugging
        if (step + 1) % 100 == 0 and hasattr(unwrapped_env, 'command_manager'):
            vel_commands = unwrapped_env.command_manager.get_command("base_velocity")
            # if vel_commands is not None:
            #     print(f"    📋 Velocity Commands (Env0): LinVel=({vel_commands[0, 0]:.2f}, {vel_commands[0, 1]:.2f}), AngVel={vel_commands[0, 2]:.2f}")
        
        # Detect collisions
        new_collisions = collision_detector.detect_collisions()
        
        # Update terrain statistics
        collision_detector.update_terrain_stats()
        
        # Update step counter
        collision_detector.total_steps += 1
        
        # Save camera images if enabled
        if args_cli.save_images and (step + 1) % args_cli.image_interval == 0:
            print(f"    🔍 Attempting to save image at step {step+1}...")
            try:
                # Access camera sensor from sensors dictionary
                if hasattr(unwrapped_env.scene, "sensors") and "camera" in unwrapped_env.scene.sensors:
                    camera = unwrapped_env.scene.sensors["camera"]
                    print(f"    🔍 Camera found in sensors, updating...")
                    camera.update(dt=0.0)  # Force update
                    
                    # Get RGB image for first environment
                    print(f"    🔍 Getting RGB data from camera.data.output...")
                    rgb_data = camera.data.output["rgb"][0].cpu().numpy()  # (H, W, 3 or 4)
                    print(f"    🔍 RGB data shape: {rgb_data.shape}, dtype: {rgb_data.dtype}")
                    
                    # Convert to uint8 if needed
                    if rgb_data.dtype == np.float32:
                        rgb_data = (rgb_data * 255).astype(np.uint8)
                    
                    # Handle RGBA -> RGB
                    if rgb_data.shape[2] == 4:
                        rgb_data = rgb_data[:, :, :3]
                    
                    # Save image
                    from PIL import Image
                    img = Image.fromarray(rgb_data)
                    image_filename = os.path.join(args_cli.image_dir, f"step_{step+1:05d}.png")
                    img.save(image_filename)
                    print(f"    📸 Saved camera image: {image_filename}")
                else:
                    print(f"    ❌ Camera not found in scene.sensors!")
                    if hasattr(unwrapped_env.scene, "sensors"):
                        print(f"    Available sensors: {list(unwrapped_env.scene.sensors.keys())}")
            except Exception as e:
                import traceback
                print(f"    ❌ Failed to save camera image at step {step+1}: {e}")
                print(f"    Traceback: {traceback.format_exc()}")
        
        # Print progress with detailed motion info
        if (step + 1) % 100 == 0:
            stats = collision_detector.get_statistics()
            # Calculate motion statistics
            avg_speed = torch.norm(robot_velocities[:, :2], dim=1).mean().item()  # XY plane speed
            avg_height = robot_full_pos[:, 2].mean().item()
            avg_dist_to_goal = distance_to_goal.mean().item()
            
            # print(f"  Step {step+1}/{args_cli.num_steps} | "
            #       f"Collisions: {stats['total_collisions']} ({stats['overall_collision_rate_percent']:.2f}%) | "
            #       f"Speed: {avg_speed:.2f}m/s | Height: {avg_height:.2f}m | Dist2Goal: {avg_dist_to_goal:.2f}m")
        
        # Print detailed motion info every 50 steps
        if (step + 1) % 50 == 0:
            for env_id in range(min(2, args_cli.num_envs)):  # Print first 2 envs only
                pos = robot_full_pos[env_id]
                vel = robot_velocities[env_id]
                ang_vel = robot_angular_vel[env_id]
                speed = torch.norm(vel[:2]).item()
                dist = distance_to_goal[env_id].item()
                
                # print(f"    🤖 Env{env_id}: Pos({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | "
                #       f"Vel({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}) | "
                #       f"Speed:{speed:.2f}m/s | AngVel:{torch.norm(ang_vel).item():.2f}rad/s | "
                #       f"Dist2Goal:{dist:.2f}m")
        
        # Print new collisions in real-time
        for env_id, obj_name in new_collisions:
            if obj_name == "body_ground":
                print(f"    ⚠️  Body-Ground Collision: Env {env_id}")
            else:
                print(f"    ⚠️  Object Collision: Env {env_id} - {obj_name}")
    
    # Print final statistics
    collision_detector.print_statistics()
    
    # Print goal navigation results
   # print(f"\n🎯 Goal Navigation Results:")
    num_reached = goal_reached.sum().item()
    # print(f"  Environments reached goal: {num_reached}/{args_cli.num_envs} ({num_reached/args_cli.num_envs*100:.1f}%)")
    unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    final_distances = torch.norm(unwrapped_env.scene["robot"].data.root_pos_w[:, :2] - goal_positions, dim=1)
    # print(f"  Average final distance to goal: {final_distances.mean().item():.2f} meters")
    # print(f"  Min distance: {final_distances.min().item():.2f} m, Max distance: {final_distances.max().item():.2f} m")
    
    # Save statistics to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = f"collision_test_results_{timestamp}.yaml"
    stats = collision_detector.get_statistics()
    stats['body_ground_collisions'] = collision_detector.body_ground_collisions
    stats['goal_navigation'] = {
        'num_reached': num_reached,
        'total_envs': args_cli.num_envs,
        'success_rate': num_reached / args_cli.num_envs,
        'avg_final_distance': final_distances.mean().item(),
        'min_distance': final_distances.min().item(),
        'max_distance': final_distances.max().item(),
    }
    
    with open(stats_file, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    
    print(f"\n📁 Statistics saved to: {stats_file}")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    # Run test
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close simulation
        simulation_app.close()
