#!/usr/bin/env python3
"""
Test ABS policy in Isaac Lab environment with collision and success rate statistics.

ABS (Adaptive Behavior System) uses:
- Proprioceptive observations: contact, angular velocity, gravity, velocity commands, 
  timer, joint positions, joint velocities, actions
- Ray2D sensor: 11-ray 2D scanning (-45° to 45°) for obstacle detection
- Input: 61-dim (contact:4 + ang_vel:3 + gravity:3 + cmd:3 + timer:1 + 
         joint_pos:12 + joint_vel:12 + actions:12 + ray2d:11)
- Output: 12-dim joint commands
- Test environment has dynamic objects to avoid

Statistics:
- Collision rate: base-ground contact + object collisions
- Success rate: reaching goal position
- Episode metrics: rewards, steps, terminations

Usage:
    bash Go2Pvcnn/scripts/test_go2_abs.sh Go2Pvcnn/scripts/test_go2_abs_collision.py \
        --checkpoint /mnt/mydisk/lhy/testPvcnnWithIsaacsim/other_model/ABS/model_4000.pt \
        --num_envs 4 --num_steps 1000 --headless --device cuda:1
"""

import argparse
import torch
import numpy as np
from datetime import datetime
import yaml

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Test ABS policy with collision detection")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to ABS policy.pt")
parser.add_argument("--num_envs", type=int, default=4, help="Number of test environments")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of test steps")
parser.add_argument("--goal_distance", type=float, default=5.0, help="Goal distance in meters")
parser.add_argument("--goal_threshold", type=float, default=0.5, help="Success threshold in meters")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--save_results", action="store_true", help="Save results to YAML")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after AppLauncher
import gymnasium as gym
from go2_pvcnn.tasks.go2_abs_test_cfg import Go2AbsTestEnvCfg_PLAY

# Configure PyTorch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class ABSActor(torch.nn.Module):
    """ABS Actor network from legged_gym."""
    def __init__(self, num_obs=61, num_actions=12, hidden_dims=[512, 256, 128]):
        super().__init__()
        layers = []
        layers.append(torch.nn.Linear(num_obs, hidden_dims[0]))
        layers.append(torch.nn.ELU())
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(torch.nn.Linear(hidden_dims[i], num_actions))
            else:
                layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(torch.nn.ELU())
        self.actor = torch.nn.Sequential(*layers)
        self.std = None  # Will be loaded from checkpoint
        
    def forward(self, obs):
        return self.actor(obs)


def main():
    """Main test loop with collision and success rate tracking."""
    
    print("="*80)
    print("ABS Policy Test - Collision & Success Rate Analysis")
    print("="*80)
    print(f"Checkpoint: {args_cli.checkpoint}")
    print(f"Num Envs: {args_cli.num_envs}")
    print(f"Num Steps: {args_cli.num_steps}")
    print(f"Goal Distance: {args_cli.goal_distance}m")
    print(f"Goal Threshold: {args_cli.goal_threshold}m")
    print(f"Seed: {args_cli.seed}")
    print("="*80 + "\n")
    
    # Set seed
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    
    # Device setup
    device = f"cuda:{app_launcher.device_id}"
    print(f"Using device: {device}\n")
    
    # ========================================
    # Load ABS Model
    # ========================================
    print("[1/4] Loading ABS model...")
    try:
        # Load checkpoint (state_dict format from legged_gym)
        checkpoint = torch.load(args_cli.checkpoint, map_location=device)
        
        # Create actor network (61 obs -> 12 actions, hidden: 512, 256, 128)
        policy = ABSActor(num_obs=61, num_actions=12, hidden_dims=[512, 256, 128])
        
        # Load actor weights
        actor_state_dict = {k.replace('actor.', ''): v for k, v in checkpoint['model_state_dict'].items() 
                           if k.startswith('actor.')}
        policy.actor.load_state_dict(actor_state_dict)
        
        # Load std (action noise)
        if 'std' in checkpoint['model_state_dict']:
            policy.std = checkpoint['model_state_dict']['std']
        
        policy.to(device)
        policy.eval()
        
        print(f"✓ Model loaded (iteration: {checkpoint.get('iter', 'unknown')})")
        print(f"  Type: ABSActor")
        print(f"  Expected input: (batch, 61) - proprioceptive + ray2d")
        print(f"    - contact: 4, ang_vel: 3, gravity: 3, cmd: 3")
        print(f"    - timer: 1, joint_pos: 12, joint_vel: 12, actions: 12")
        print(f"    - ray2d: 11 (obstacle distances)")
        print(f"  Expected output: (batch, 12) - joint commands")
        print(f"  Action std: {policy.std.cpu().numpy() if policy.std is not None else 'N/A'}\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return
    
    # ========================================
    # Create Test Environment
    # ========================================
    print("[2/4] Creating test environment...")
    
    # Configure environment
    env_cfg = Go2AbsTestEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.seed
    
    print(f"  Environments: {env_cfg.scene.num_envs}")
    print(f"  Device: {env_cfg.sim.device}")
    print(f"  Seed: {env_cfg.seed}")
    
    # Create environment
    env = gym.make("Go2AbsEnv-Test", cfg=env_cfg)
    print(f"✓ Environment created\n")
    
    # ========================================
    # Initialize Tracking Variables
    # ========================================
    print("[3/4] Initializing trackers...")
    
    # Statistics trackers
    episode_stats = {
        'total_episodes': 0,
        'successful_episodes': 0,
        'collision_ground': 0,
        'collision_objects': 0,
        'total_steps': 0,
        'total_reward': 0.0,
    }
    
    # Per-environment trackers
    env_collided_ground = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
    env_collided_objects = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
    env_steps = torch.zeros(args_cli.num_envs, dtype=torch.int, device=device)
    
    # Movement tracking
    prev_robot_pos = torch.zeros(args_cli.num_envs, 3, device=device)
    total_distance_moved = torch.zeros(args_cli.num_envs, device=device)
    
    print(f"✓ Statistics tracking initialized")
    print(f"✓ Movement tracking initialized\n")
    
    # ========================================
    # Run Test
    # ========================================
    print(f"[4/4] Running {args_cli.num_steps} steps...")
    print("-" * 80)
    
    # Reset environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]  # Shape: (num_envs, 61) - ABS直接使用61维观测
    
    # Initialize previous position
    prev_robot_pos[:] = env.unwrapped.scene["robot"].data.root_pos_w[:, :3]
    
    step = 0
    
    # Test loop
    while step < args_cli.num_steps:
        # Debug: Check observation for NaN
        if torch.isnan(obs).any():
            print(f"\n❌ NaN detected in observation at step {step}!")
            print(f"Observation shape: {obs.shape}")
            print(f"NaN count: {torch.isnan(obs).sum().item()}")
            nan_indices = torch.where(torch.isnan(obs))[0]
            print(f"First few NaN indices: {nan_indices[:10].tolist()}")
            break
        
        # Get action from policy
        with torch.no_grad():
            actions = policy(obs)
        
        # Debug: Check action for NaN
        if torch.isnan(actions).any():
            print(f"\n❌ NaN detected in action at step {step}!")
            print(f"Action shape: {actions.shape}")
            print(f"NaN count: {torch.isnan(actions).sum().item()}")
            break
        
        # Step environment
        obs_dict, rewards, terminated, truncated, info = env.step(actions)
        obs = obs_dict["policy"]
        
        # ========================================
        # Collision Detection
        # ========================================
        contact_forces = env.unwrapped.scene["contact_forces"]
        base_contact = contact_forces.data.net_forces_w[:, 0, :]
        base_contact_magnitude = torch.norm(base_contact, dim=1)
        ground_collision = base_contact_magnitude > 1.0
        
        # Check object collisions
        object_collision = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
        for obj_idx in range(3):
            obj_contact = env.unwrapped.scene[f"object_{obj_idx}_contact"]
            obj_forces = obj_contact.data.net_forces_w[:, 0, :]
            obj_contact_magnitude = torch.norm(obj_forces, dim=1)
            object_collision |= (obj_contact_magnitude > 0.1)
        
        env_collided_ground |= ground_collision
        env_collided_objects |= object_collision
        
        # ========================================
        # Goal Tracking & Movement
        # ========================================
        robot_pos_3d = env.unwrapped.scene["robot"].data.root_pos_w[:, :3]
        robot_pos = robot_pos_3d[:, :2]
        
        # Calculate movement
        movement_delta = torch.norm(robot_pos_3d - prev_robot_pos, dim=1)
        total_distance_moved += movement_delta
        prev_robot_pos[:] = robot_pos_3d
        
        # Check goal reached
        if hasattr(env.unwrapped, 'goal_positions'):
            goal_positions = env.unwrapped.goal_positions
            distance_to_goal = torch.norm(goal_positions - robot_pos, dim=1)
            reached_goal = distance_to_goal < args_cli.goal_threshold
        else:
            reached_goal = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
        
        # Update statistics
        episode_stats['total_reward'] += rewards.sum().item()
        episode_stats['total_steps'] += args_cli.num_envs
        env_steps += 1
        
        # ========================================
        # Handle Episode Terminations
        # ========================================
        dones = terminated | truncated | reached_goal
        
        if dones.any():
            done_ids = torch.where(dones)[0]
            
            for env_id in done_ids:
                episode_stats['total_episodes'] += 1
                
                # Success: reached goal without collision
                if reached_goal[env_id] and not env_collided_ground[env_id] and not env_collided_objects[env_id]:
                    episode_stats['successful_episodes'] += 1
                
                # Record collisions
                if env_collided_ground[env_id]:
                    episode_stats['collision_ground'] += 1
                if env_collided_objects[env_id]:
                    episode_stats['collision_objects'] += 1
            
            # Reset trackers
            env_collided_ground[done_ids] = False
            env_collided_objects[done_ids] = False
            env_steps[done_ids] = 0
        
        step += 1
        
        # Progress report
        if step % 200 == 0 or step == 1:
            # 计算统计指标
            avg_reward = episode_stats['total_reward'] / episode_stats['total_steps'] if episode_stats['total_steps'] > 0 else 0
            success_rate = episode_stats['successful_episodes'] / episode_stats['total_episodes'] * 100 if episode_stats['total_episodes'] > 0 else 0
            collision_rate = (episode_stats['collision_ground'] + episode_stats['collision_objects']) / episode_stats['total_episodes'] * 100 if episode_stats['total_episodes'] > 0 else 0
            
            # 移动统计
            avg_movement = total_distance_moved.mean().item()
            current_pos = env.unwrapped.scene["robot"].data.root_pos_w[0, :3].cpu().numpy()
            
            # 打印进度信息
            print(f"  Step {step:4d}/{args_cli.num_steps} | "
                  f"Episodes: {episode_stats['total_episodes']:3d} | "
                  f"Success: {success_rate:5.1f}% | "
                  f"Collision: {collision_rate:5.1f}% | "
                  f"Avg Reward: {avg_reward:6.3f}")
            print(f"    Env[0] Pos: [{current_pos[0]:6.2f}, {current_pos[1]:6.2f}, {current_pos[2]:6.2f}] | "
                  f"Avg Moved: {avg_movement:6.3f}m")
    
    # ========================================
    # Final Statistics
    # ========================================
    print("-" * 80)
    print("\n" + "="*80)
    print("Test Completed - Final Results")
    print("="*80)
    
    total_episodes = episode_stats['total_episodes']
    if total_episodes > 0:
        # 计算各项统计率
        success_rate = episode_stats['successful_episodes'] / total_episodes * 100
        ground_collision_rate = episode_stats['collision_ground'] / total_episodes * 100
        object_collision_rate = episode_stats['collision_objects'] / total_episodes * 100
        total_collision_rate = ground_collision_rate + object_collision_rate
        avg_reward = episode_stats['total_reward'] / episode_stats['total_steps']
    else:
        success_rate = 0
        ground_collision_rate = 0
        object_collision_rate = 0
        total_collision_rate = 0
        avg_reward = 0
    
    print(f"\nEpisode Statistics:")
    print(f"  Total Episodes:        {total_episodes}")
    print(f"  Successful Episodes:   {episode_stats['successful_episodes']} ({success_rate:.2f}%)")
    print(f"\nCollision Statistics:")
    print(f"  Ground Collisions:     {episode_stats['collision_ground']} ({ground_collision_rate:.2f}%)")
    print(f"  Object Collisions:     {episode_stats['collision_objects']} ({object_collision_rate:.2f}%)")
    print(f"  Total Collision Rate:  {total_collision_rate:.2f}%")
    print(f"\nMovement Statistics:")
    print(f"  Avg Distance Moved:    {total_distance_moved.mean().item():.3f}m")
    print(f"  Max Distance Moved:    {total_distance_moved.max().item():.3f}m")
    print(f"  Min Distance Moved:    {total_distance_moved.min().item():.3f}m")
    print(f"\nPerformance Metrics:")
    print(f"  Total Steps:           {episode_stats['total_steps']:,}")
    print(f"  Average Reward:        {avg_reward:.4f}")
    print(f"  Goal Distance:         {args_cli.goal_distance:.1f}m")
    print(f"  Goal Threshold:        {args_cli.goal_threshold:.1f}m")
    print("="*80 + "\n")
    
    # ========================================
    # Save Results
    # ========================================
    if args_cli.save_results:
        results = {
            'checkpoint': args_cli.checkpoint,
            'test_config': {
                'num_envs': args_cli.num_envs,
                'num_steps': args_cli.num_steps,
                'goal_distance': args_cli.goal_distance,
                'goal_threshold': args_cli.goal_threshold,
                'seed': args_cli.seed,
            },
            'statistics': {
                'total_episodes': int(total_episodes),
                'successful_episodes': int(episode_stats['successful_episodes']),
                'success_rate': float(success_rate),
                'collision_ground': int(episode_stats['collision_ground']),
                'collision_objects': int(episode_stats['collision_objects']),
                'ground_collision_rate': float(ground_collision_rate),
                'object_collision_rate': float(object_collision_rate),
                'total_collision_rate': float(total_collision_rate),
                'total_steps': int(episode_stats['total_steps']),
                'average_reward': float(avg_reward),
                'avg_distance_moved': float(total_distance_moved.mean().item()),
                'max_distance_moved': float(total_distance_moved.max().item()),
                'min_distance_moved': float(total_distance_moved.min().item()),
            },
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"abs_test_results_{timestamp}.yaml"
        
        with open(filename, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        print(f"Results saved to: {filename}")
    
    # Close
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
