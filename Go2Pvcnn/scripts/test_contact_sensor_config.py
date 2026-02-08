#!/usr/bin/env python3
"""
Test Contact Sensor integration with Go2 robot in Isaac Sim.

This script creates a simple environment with a Go2 robot equipped with contact sensors
and tests the contact sensor functionality with categorized object detection (ground, small objects, furniture).

Usage:
    python test_contact_sensor_config.py --num_envs 4 --headless
"""

import argparse
import sys
from pathlib import Path

# Isaac Lab imports - MUST be before AppLauncher
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test Contact Sensors with Go2 robot.")
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

from go2_pvcnn.tasks.go2_contact_global_env_cfg import Go2LidarEnvCfg





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
    
    # Debug: Check contact sensors
    print("\n" + "=" * 80)
    print("Contact Sensor Check")
    print("=" * 80)
    
    # Check available sensors
    if hasattr(env.scene, 'sensors') and isinstance(env.scene.sensors, dict):
        print(f"  Available sensors: {list(env.scene.sensors.keys())}")
        
        # Check each contact sensor
        contact_sensor_names = [
            'contact_forces_ground',
            'contact_forces_small_objects', 
            'contact_forces_furniture',
            'contact_forces'
        ]
        
        for sensor_name in contact_sensor_names:
            if sensor_name in env.scene.sensors:
                sensor = env.scene.sensors[sensor_name]
                print(f"\n  ✓ Found sensor: {sensor_name}")
                print(f"    - Prim path: {sensor.cfg.prim_path}")
                print(f"    - Update period: {sensor.cfg.update_period}")
                print(f"    - History length: {sensor.cfg.history_length}")
                print(f"    - Track air time: {sensor.cfg.track_air_time}")
                if sensor.cfg.filter_prim_paths_expr:
                    print(f"    - Filter paths: {len(sensor.cfg.filter_prim_paths_expr)} paths")
                    for path in sensor.cfg.filter_prim_paths_expr[:3]:  # Show first 3
                        print(f"        {path}")
                    if len(sensor.cfg.filter_prim_paths_expr) > 3:
                        print(f"        ... and {len(sensor.cfg.filter_prim_paths_expr) - 3} more")
            else:
                print(f"\n  ✗ Sensor not found: {sensor_name}")
    
    print("=" * 80 + "\n")
    
    # Run simulation steps
    print(f"\nRunning {args_cli.num_steps} simulation steps...")
    print("-" * 80)
    
    for step in range(args_cli.num_steps):
        # Get random actions
        actions = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        
        # Step environment
        obs, _, terminated, truncated, info = env.step(actions)
        
        # Print contact sensor data every 20 steps
        if step % 1 == 0:
            print(f"\n{'='*80}")
            print(f"Step {step}: Contact Sensor Data")
            print(f"{'='*80}")
            
            if hasattr(env.scene, 'sensors') and isinstance(env.scene.sensors, dict):
                
                # 1. Ground contact sensor
                if 'contact_forces_ground' in env.scene.sensors:
                    sensor = env.scene.sensors['contact_forces_ground']
                    data = sensor.data
                    print(f"\n[1] GROUND Contact Sensor:")
                    print(f"  - Net forces shape: {data.net_forces_w.shape}")
                    # No force_matrix for ground sensor (no filter used)
                    print(f"  - Using filter: No (monitors all contacts)")
                    
                    # Check dimensions: shape is [num_envs, num_bodies, 3]
                    assert data.net_forces_w.shape[0] == env.num_envs, f"Env dimension mismatch: {data.net_forces_w.shape[0]} != {env.num_envs}"
                    assert data.net_forces_w.shape[2] == 3, f"Force dimension should be 3, got {data.net_forces_w.shape[2]}"
                    num_bodies = data.net_forces_w.shape[1]
                    print(f"  - Monitoring {num_bodies} bodies (4 feet) across {env.num_envs} environments")
                    
                    # Statistics
                    force_norms = torch.norm(data.net_forces_w, dim=-1)  # [num_envs, num_bodies]
                    in_contact = (force_norms > sensor.cfg.force_threshold).sum().item()
                    total_bodies = force_norms.numel()
                    print(f"  - Bodies in contact: {in_contact}/{total_bodies} (threshold={sensor.cfg.force_threshold}N)")
                    print(f"  - Force magnitude range: [{force_norms.min():.3f}, {force_norms.max():.3f}] N")
                    
                    # Mean force across all bodies and envs
                    mean_force = data.net_forces_w.mean(dim=(0, 1))  # Average over envs and bodies
                    print(f"  - Mean force: [x={mean_force[0]:.3f}, y={mean_force[1]:.3f}, z={mean_force[2]:.3f}] N")
                    
                    # Per-foot statistics (show first env)
                    print(f"  - Env 0 foot forces (FL, FR, RL, RR):")
                    for foot_idx in range(num_bodies):
                        foot_force = torch.norm(data.net_forces_w[0, foot_idx])
                        print(f"      Foot {foot_idx}: {foot_force:.3f} N")
                    
                    if data.current_air_time is not None:
                        print(f"  - Air time shape: {data.current_air_time.shape}")
                        print(f"  - Air time range: [{data.current_air_time.min():.3f}, {data.current_air_time.max():.3f}] s")
                
                # 2. Small objects contact sensor
                if 'contact_forces_small_objects' in env.scene.sensors:
                    sensor = env.scene.sensors['contact_forces_small_objects']
                    data = sensor.data
                    print(f"\n[2] SMALL OBJECTS Contact Sensor:")
                    print(f"  - Net forces shape: {data.net_forces_w.shape}")
                    print(f"  - Force matrix shape: {data.force_matrix_w.shape}")
                    
                    # Check dimensions: shape is [num_envs, 1, 3] for single body
                    # force_matrix shape is [num_envs, 1, num_objects, 3]
                    assert data.net_forces_w.shape[0] == env.num_envs, f"Env dimension mismatch"
                    num_bodies = data.net_forces_w.shape[1]
                    print(f"  - Monitoring {num_bodies} body (base) across {env.num_envs} environments")
                    print(f"  - Filtering {data.force_matrix_w.shape[2]} objects per environment")
                    
                    # Check each object's contact
                    for env_id in range(min(2, env.num_envs)):  # Show first 2 envs
                        env_forces = data.force_matrix_w[env_id, 0]  # [num_objects, 3]
                        force_norms = torch.norm(env_forces, dim=-1)  # [num_objects]
                        in_contact = (force_norms > sensor.cfg.force_threshold).sum().item()
                        if in_contact > 0:
                            print(f"  - Env {env_id}: {in_contact} objects in contact")
                            for obj_id in range(len(force_norms)):
                                if force_norms[obj_id] > sensor.cfg.force_threshold:
                                    print(f"      Object {obj_id}: {force_norms[obj_id]:.3f} N")
                        else:
                            print(f"  - Env {env_id}: No contacts with small objects")
                
                # 3. Furniture contact sensor
                if 'contact_forces_furniture' in env.scene.sensors:
                    sensor = env.scene.sensors['contact_forces_furniture']
                    data = sensor.data
                    print(f"\n[3] FURNITURE Contact Sensor:")
                    print(f"  - Net forces shape: {data.net_forces_w.shape}")
                    print(f"  - Force matrix shape: {data.force_matrix_w.shape}")
                    
                    # Check dimensions: shape is [num_envs, 1, 3] for single body
                    # force_matrix shape is [num_envs, 1, num_furniture, 3]
                    assert data.net_forces_w.shape[0] == env.num_envs, f"Env dimension mismatch"
                    num_bodies = data.net_forces_w.shape[1]
                    print(f"  - Monitoring {num_bodies} body (base) across {env.num_envs} environments")
                    print(f"  - Filtering {data.force_matrix_w.shape[2]} furniture items")
                    
                    # Check each furniture's contact
                    any_contact = False
                    for env_id in range(env.num_envs):
                        env_forces = data.force_matrix_w[env_id, 0]  # [num_furniture, 3]
                        force_norms = torch.norm(env_forces, dim=-1)  # [num_furniture]
                        in_contact = (force_norms > sensor.cfg.force_threshold).sum().item()
                        if in_contact > 0:
                            any_contact = True
                            print(f"  - ⚠️  Env {env_id}: Colliding with {in_contact} furniture items!")
                            for furn_id in range(len(force_norms)):
                                if force_norms[furn_id] > sensor.cfg.force_threshold:
                                    print(f"      Furniture {furn_id}: {force_norms[furn_id]:.3f} N")
                    
                    if not any_contact:
                        print(f"  - ✓ No furniture collisions detected (Good!)")
                
                print(f"\n{'='*80}")
    
    print("\n" + "=" * 80)
    print("Contact Sensor Test Completed Successfully!")
    print("All dimension checks passed ✓")
    print("=" * 80)
    
    env.close()
    simulation_app.close()
    


if __name__ == "__main__":
    main()
