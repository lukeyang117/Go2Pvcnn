#!/usr/bin/env python3
"""
Test LiDAR sensor integration with Go2 robot in Isaac Sim.

This script creates a simple environment with a Go2 robot equipped with a LiDAR sensor
and tests the LiDAR functionality without the PVCNN semantic understanding part.

Usage:
    python test_lidar_config.py --num_envs 4 --headless
"""

import argparse
import sys
from pathlib import Path

# Isaac Lab imports - MUST be before AppLauncher
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test LiDAR sensor with Go2 robot.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
parser.add_argument("--num_steps", type=int, default=200, help="Number of simulation steps to run.")

# Append AppLauncher arguments (includes --device automatically)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print(f"Device ID: {app_launcher.device_id}")

"""Rest of the script follows after Isaac Sim is launched."""

import torch
import gymnasium as gym
import numpy as np

from isaaclab.envs import ManagerBasedRLEnv

# Add Go2Pvcnn to path
sys.path.insert(0, "/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn")

from go2_pvcnn.sensor.lidar import LidarCfg, LivoxPatternCfg
from go2_pvcnn.tasks.go2_lidar_env_cfg import Go2LidarEnvCfg





def main():
    """Main test function."""
    
    # Create test environment
    env_cfg = Go2LidarEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.device = args_cli.device
    
    # Create the environment
    print("Creating environment...")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"✓ Environment created with {env.num_envs} parallel environments")
    
    # Get observations
    print("\nInitializing observations...")
    obs, _ = env.reset()
    # obs is a dict with keys like "policy", "critic"
    if isinstance(obs, dict):
        policy_obs = obs.get("policy", None)
        if policy_obs is not None:
            print(f"✓ Policy observation shape: {policy_obs.shape}")
        for key, val in obs.items():
            if isinstance(val, torch.Tensor):
                print(f"  - {key}: {val.shape}")
    else:
        print(f"✓ Observation shape: {obs.shape}")
    
    # Debug: Check if LiDAR sensor exists
    print("\n" + "=" * 70)
    print("LiDAR Sensor Check")
    print("=" * 70)
    
    # Try different ways to access the sensor
    lidar_sensor = None
    
    if hasattr(env.scene, 'sensors') and isinstance(env.scene.sensors, dict):
        print(f"  Available sensors: {list(env.scene.sensors.keys())}", flush=True)
        if 'lidar' in env.scene.sensors:
            print(f"✓ Found via env.scene.sensors['lidar']", flush=True)
            lidar_sensor = env.scene.sensors['lidar']
    
    print("=" * 70 + "\n")
    
    # Run simulation steps
    print(f"\nRunning {args_cli.num_steps} simulation steps...")
    print("-" * 70)
    
    distances_history = []
    pointcloud_samples = []
    
    for step in range(args_cli.num_steps):
        # Get random actions
        actions = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        
        # Step environment
        obs, _, terminated, truncated, info = env.step(actions)
        
        # Access LiDAR data through environment
        try:
            # Get LiDAR sensor from scene.sensors dict
            if hasattr(env.scene, 'sensors') and isinstance(env.scene.sensors, dict):
                if 'lidar' in env.scene.sensors:
                    lidar_sensor = env.scene.sensors['lidar']
                    
                    # Get LiDAR data
                    try:
                        lidar_data = lidar_sensor.data
                        
                        if hasattr(lidar_data, 'ray_hits_w'):
                            ray_hits = lidar_data.ray_hits_w  # Shape: (num_envs, num_rays, 3)
                            if ray_hits is not None and ray_hits.numel() > 0:
                                # Calculate distances from ray start positions
                                ray_origins = ray_hits[..., 0:1, :]  # First hit as origin
                                ray_displacements = ray_hits - ray_origins
                                distances = torch.norm(ray_displacements, dim=-1)  # (num_envs, num_rays)
                                
                                # Record statistics
                                distances_history.append({
                                    'step': step,
                                    'mean': distances.mean().item(),
                                    'min': distances.min().item(),
                                    'max': distances.max().item(),
                                    'std': distances.std().item(),
                                    'num_rays': ray_hits.shape[1],
                                    'num_envs': ray_hits.shape[0],
                                })
                    except Exception as inner_e:
                        if step == 0:
                            print(f"[ERROR] Failed to access LiDAR data: {inner_e}", flush=True)
        except Exception as e:
            if step == 0:
                print(f"[ERROR] Step {step}: {e}", flush=True)
        
        # Progress reporting with flush
        report_interval = max(1, args_cli.num_steps // 10)  # Report 10 times total
        if (step + 1) % report_interval == 0 or step == 0:
            print(f"  Step {step + 1}/{args_cli.num_steps}", flush=True)
            if len(distances_history) > 0:
                last_stats = distances_history[-1]
                print(f"    Ray hits - Mean dist: {last_stats['mean']:.3f}m, "
                      f"Range: [{last_stats['min']:.3f}, {last_stats['max']:.3f}]m, "
                      f"Std: {last_stats['std']:.3f}m", flush=True)
            else:
                if (step + 1) % report_interval == 0:
                    print(f"    [INFO] distances_history: {len(distances_history)} records", flush=True)
    
    
    print("-" * 70)
    print("✓ Simulation completed successfully")
    
    # Print statistics
    if distances_history:
        print("\n" + "=" * 70)
        print("LiDAR Data Statistics")
        print("=" * 70)
        
        mean_distances = np.array([d['mean'] for d in distances_history])
        min_distances = np.array([d['min'] for d in distances_history])
        max_distances = np.array([d['max'] for d in distances_history])
        std_distances = np.array([d['std'] for d in distances_history])
        
        print(f"\nRay Casting Results ({len(distances_history)} steps):")
        print(f"  Rays per environment:    {distances_history[0]['num_rays']}")
        print(f"  Parallel environments:   {distances_history[0]['num_envs']}")
        print(f"  Total rays per step:     {distances_history[0]['num_rays'] * distances_history[0]['num_envs']:,}")
        
        print(f"\nDistance Statistics (Ray Hit Measurements):")
        print(f"  Mean distance:      {mean_distances.mean():.3f} ± {mean_distances.std():.3f} m")
        print(f"  Min range (avg):    {min_distances.mean():.3f} ± {min_distances.std():.3f} m")
        print(f"  Max range (avg):    {max_distances.mean():.3f} ± {max_distances.std():.3f} m")
        print(f"  Std dev (avg):      {std_distances.mean():.3f} ± {std_distances.std():.3f} m")
        
        
        
       
    
   
    env.close()
    simulation_app.close()
    


if __name__ == "__main__":
    main()
