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
    
    for step in range(args_cli.num_steps):
        # Get random actions
        actions = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        
        # Step environment
        obs, _, terminated, truncated, info = env.step(actions)
        
        # Access LiDAR data through environment
        if hasattr(env.scene, 'sensors') and isinstance(env.scene.sensors, dict):
            if 'lidar' in env.scene.sensors:
                lidar_sensor = env.scene.sensors['lidar']
                
                # Get LiDAR data
                lidar_data = lidar_sensor.data
                print(f"\nStep {step}: LiDAR Data")
                print(f"  Distances shape: {lidar_data.distances.shape}")
                print(f"  Distances range: [{lidar_data.distances.min():.3f}, {lidar_data.distances.max():.3f}]")
                
                # 获取语义分类数据
                if hasattr(lidar_data, 'semantic_labels') and lidar_data.semantic_labels is not None:
                    semantic_labels = lidar_data.semantic_labels
                    print(f"  Semantic labels shape: {semantic_labels.shape}")
                    print(f"  Unique classes: {torch.unique(semantic_labels).cpu().tolist()}")
                
                # 获取语义分类统计
                if hasattr(lidar_data, 'semantic_class_counts') and lidar_data.semantic_class_counts is not None:
                    class_counts = lidar_data.semantic_class_counts
                    print(f"  Semantic class counts shape: {class_counts.shape}")
                    
                    # 打印每个环境的分类统计
                    print(f"\n  Semantic Classification Statistics:")
                    print(f"  {'Env':<5} {'No-hit (0)':<12} {'Terrain (1)':<12} {'Obstacle (2)':<12} {'Valuable (3)':<12}")
                    print(f"  {'-'*60}")
                    for env_id in range(min(5, env.num_envs)):  # 只显示前5个环境
                        counts = class_counts[env_id].cpu().numpy()
                        print(f"  {env_id:<5} {int(counts[0]):<12} {int(counts[1]):<12} {int(counts[2]):<12} {int(counts[3]):<12}")
                    
                    # 打印平均统计
                    avg_counts = class_counts.float().mean(dim=0).cpu().numpy()
                    print(f"  {'AVG':<5} {avg_counts[0]:<12.1f} {avg_counts[1]:<12.1f} {avg_counts[2]:<12.1f} {avg_counts[3]:<12.1f}")
                    print()
                
                # 获取高程图数据
                if hasattr(lidar_data, 'height_map') and lidar_data.height_map is not None:
                    height_map = lidar_data.height_map
                    print(f"  Height map shape: {height_map.shape}")
                    # 统计每个环境的高程图
                    for env_id in range(min(3, env.num_envs)):  # 只显示前3个环境
                        env_height_map = height_map[env_id]
                        # 统计有效(非NaN)的网格单元
                        valid_cells = torch.isfinite(env_height_map).sum().item()
                        total_cells = env_height_map.numel()
                        if valid_cells > 0:
                            valid_heights = env_height_map[torch.isfinite(env_height_map)]
                            print(f"    Env {env_id}: {valid_cells}/{total_cells} cells ({100*valid_cells/total_cells:.1f}%) "
                                  f"Z range=[{valid_heights.min().item():.3f}, {valid_heights.max().item():.3f}]m")
                        else:
                            print(f"    Env {env_id}: No valid height data")
    
    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)
    
    env.close()
    simulation_app.close()
    


if __name__ == "__main__":
    main()
