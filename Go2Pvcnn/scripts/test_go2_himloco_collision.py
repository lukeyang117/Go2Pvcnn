#!/usr/bin/env python3
"""
Test HIMLoco policy in Isaac Lab environment with collision and success rate statistics.

HIMLoco uses proprioceptive observations only (no PVCNN/LiDAR):
- Input: 45-dim obs × 6 history = 270-dim
- Output: 12-dim joint commands
- Test environment has dynamic objects to avoid

Statistics:
- Collision rate: base-ground contact + object collisions
- Success rate: reaching goal position
- Episode metrics: rewards, steps, terminations

Usage:
    python test_himloco_collision.py --checkpoint /path/to/policy.pt --num_envs 4 --headless
"""

import argparse
import torch
import numpy as np
from datetime import datetime
import yaml

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Test HIMLoco policy with collision detection")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to HIMLoco policy.pt")
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
from go2_pvcnn.tasks.go2_himloco_test_cfg import Go2HimlocoTestEnvCfg_PLAY

# Configure PyTorch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    """Main test loop with collision and success rate tracking."""
    
    print("="*80)
    print("HIMLoco Policy Test - Collision & Success Rate Analysis")
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
    # Load HIMLoco Model
    # ========================================
    print("[1/4] Loading HIMLoco model...")
    try:
        policy = torch.jit.load(args_cli.checkpoint, map_location=device)
        policy.eval()
        print(f"✓ Model loaded")
        print(f"  Type: {type(policy)}")
        print(f"  Expected input: (batch, 270) - 45 obs × 6 history")
        print(f"  Expected output: (batch, 12) - joint commands\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        simulation_app.close()
        return
    
    # ========================================
    # Create Test Environment
    # ========================================
    print("[2/4] Creating test environment...")
    
    # Configure environment
    env_cfg = Go2HimlocoTestEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.seed
    
    print(f"  Environments: {env_cfg.scene.num_envs}")
    print(f"  Device: {env_cfg.sim.device}")
    print(f"  Seed: {env_cfg.seed}")
    
    # Create environment
    env = gym.make("Go2HimlocoEnv-Test", cfg=env_cfg)
    print(f"✓ Environment created\n")
    
    # ========================================
    # Initialize Tracking Variables
    # ========================================
    print("[3/4] Initializing trackers...")
    
    # Observation history
    obs_dim = 45
    history_len = 6
    obs_history = torch.zeros(
        args_cli.num_envs, 
        history_len * obs_dim, 
        dtype=torch.float32,
        device=device
    )
    
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
    prev_robot_pos = torch.zeros(args_cli.num_envs, 3, device=device)  # [x, y, z]
    total_distance_moved = torch.zeros(args_cli.num_envs, device=device)
    
    print(f"✓ History buffer: {obs_history.shape}")
    print(f"✓ Goal tracking initialized")
    print(f"✓ Movement tracking initialized\n")
    
    # ========================================
    # Run Test
    # ========================================
    print(f"[4/4] Running {args_cli.num_steps} steps...")
    print("-" * 80)
    
    # Reset environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    # Handle observation dimension
    if obs.shape[1] > obs_dim:
        obs = obs[:, :obs_dim]
    
    # Initialize history
    for _ in range(history_len):
        obs_history[:, :-obs_dim] = obs_history[:, obs_dim:].clone()
        obs_history[:, -obs_dim:] = obs
    
    # Initialize previous position for movement tracking
    prev_robot_pos[:] = env.unwrapped.scene["robot"].data.root_pos_w[:, :3]
    
    step = 0
    
    # Test loop
    while step < args_cli.num_steps:
        # Get action from policy
        with torch.no_grad():
            actions = policy(obs_history)
        
        # Step environment
        obs_dict, rewards, terminated, truncated, info = env.step(actions) #truncated 是不是因为达到最大步数而终止的
        obs = obs_dict["policy"]
        
        # Handle observation dimension
        if obs.shape[1] > obs_dim:
            obs = obs[:, :obs_dim]
        
        # Update history
        obs_history[:, :-obs_dim] = obs_history[:, obs_dim:].clone()
        obs_history[:, -obs_dim:] = obs
        
        # ========================================
        # Collision Detection
        # ========================================
        # Check base-ground contact
        contact_forces = env.unwrapped.scene["contact_forces"]
        base_contact = contact_forces.data.net_forces_w[:, 0, :]  # [num_envs, 3] - base link
        base_contact_magnitude = torch.norm(base_contact, dim=1)
        ground_collision = base_contact_magnitude > 1.0  # threshold
        
        # Check object collisions (3 objects per env)
        object_collision = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
        for obj_idx in range(3):
            obj_contact = env.unwrapped.scene[f"object_{obj_idx}_contact"]
            obj_forces = obj_contact.data.net_forces_w[:, 0, :]  # [num_envs, 3]
            obj_contact_magnitude = torch.norm(obj_forces, dim=1)
            object_collision |= (obj_contact_magnitude > 0.1)  # any contact
        
        # Mark collisions for this step
        env_collided_ground |= ground_collision
        env_collided_objects |= object_collision
        
        # ========================================
        # Goal Tracking
        # ========================================
        robot_pos_3d = env.unwrapped.scene["robot"].data.root_pos_w[:, :3]  # [num_envs, 3]
        robot_pos = robot_pos_3d[:, :2]  # [num_envs, 2] for goal checking
        
        # Calculate movement distance
        movement_delta = torch.norm(robot_pos_3d - prev_robot_pos, dim=1)
        total_distance_moved += movement_delta
        prev_robot_pos[:] = robot_pos_3d
        
        # Get goal positions from observation function (stored in env.unwrapped)
        if hasattr(env.unwrapped, 'goal_positions'):
            goal_positions = env.unwrapped.goal_positions
            distance_to_goal = torch.norm(goal_positions - robot_pos, dim=1)
            reached_goal = distance_to_goal < args_cli.goal_threshold
        else:
            # Fallback: no goal tracking if not initialized
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
                
                # Check if successful (reached goal without collision)
                # 成功条件：到达目标 且 无地面碰撞 且 无物体碰撞
                if reached_goal[env_id] and not env_collided_ground[env_id] and not env_collided_objects[env_id]:
                    episode_stats['successful_episodes'] += 1
                
                # Record collisions
                # 注意：一个episode可能同时发生地面碰撞和物体碰撞
                # 所以 ground_collision + object_collision 的总数可能 > total_episodes
                # 例如：162个episode中，126个碰撞地面(77.78%)，89个碰撞物体(54.94%)
                #      有些episode既碰撞地面又碰撞物体，所以百分比加起来可能>100%
                if env_collided_ground[env_id]:
                    episode_stats['collision_ground'] += 1  # 该episode发生过地面碰撞
                if env_collided_objects[env_id]:
                    episode_stats['collision_objects'] += 1  # 该episode发生过物体碰撞
            
            # Reset trackers for done environments
            env_collided_ground[done_ids] = False
            env_collided_objects[done_ids] = False
            env_steps[done_ids] = 0
            
            # Reset history for done environments
            for _ in range(history_len):
                obs_history[done_ids, :-obs_dim] = obs_history[done_ids, obs_dim:].clone()
                obs_history[done_ids, -obs_dim:] = obs[done_ids]
        
        step += 1
        
        # Progress report
        if step % 200 == 0 or step == 1:
            # 计算统计指标
            avg_reward = episode_stats['total_reward'] / episode_stats['total_steps'] if episode_stats['total_steps'] > 0 else 0  # 平均奖励
            success_rate = episode_stats['successful_episodes'] / episode_stats['total_episodes'] * 100 if episode_stats['total_episodes'] > 0 else 0  # 成功率(%)
            collision_rate = (episode_stats['collision_ground'] + episode_stats['collision_objects']) / episode_stats['total_episodes'] * 100 if episode_stats['total_episodes'] > 0 else 0  # 碰撞率(%)
            
            # 移动统计
            avg_movement = total_distance_moved.mean().item()  # 所有环境的平均累积移动距离(米)
            current_pos = env.unwrapped.scene["robot"].data.root_pos_w[0, :3].cpu().numpy()  # 第0个环境的机器人当前位置[x, y, z]
            
            # 打印进度信息
            # step: 当前步数 / 总步数
            # Episodes: 已完成的episode总数
            # Success: 成功到达目标且无碰撞的episode占比
            # Collision: 发生碰撞(地面+物体)的episode占比
            # Avg Reward: 平均每步奖励
            print(f"  Step {step:4d}/{args_cli.num_steps} | "
                  f"Episodes: {episode_stats['total_episodes']:3d} | "
                  f"Success: {success_rate:5.1f}% | "
                  f"Collision: {collision_rate:5.1f}% | "
                  f"Avg Reward: {avg_reward:6.3f}")
            # Env[0] Pos: 第0个环境机器人的世界坐标[x, y, z](米)
            # Avg Moved: 所有环境机器人的平均累积移动距离(米)
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
        # 成功率 = 成功episode数 / 总episode数
        success_rate = episode_stats['successful_episodes'] / total_episodes * 100
        
        # 地面碰撞率 = 发生地面碰撞的episode数 / 总episode数
        # 例如：126 / 162 = 77.78%，表示77.78%的episode中机器人base碰到了地面
        ground_collision_rate = episode_stats['collision_ground'] / total_episodes * 100
        
        # 物体碰撞率 = 发生物体碰撞的episode数 / 总episode数  
        # 例如：89 / 162 = 54.94%，表示54.94%的episode中机器人碰到了动态物体
        object_collision_rate = episode_stats['collision_objects'] / total_episodes * 100
        
        # 注意：一个episode可能既碰地面又碰物体，所以两个碰撞率相加可能>100%
        # 总碰撞率只是简单相加，不代表"至少发生一次碰撞"的episode占比
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
    print(f"  Ground Collisions:     {episode_stats['collision_ground']} ({ground_collision_rate:.2f}%)")  # 碰撞地面的episode数及占比
    print(f"  Object Collisions:     {episode_stats['collision_objects']} ({object_collision_rate:.2f}%)")  # 碰撞物体的episode数及占比
    print(f"  Total Collision Rate:  {total_collision_rate:.2f}%")  # 两者相加（可能>100%因为有重叠）
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
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        }
        
        filename = f"himloco_test_results_{results['timestamp']}.yaml"
        with open(filename, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        print(f"Results saved to: {filename}\n")
    
    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        simulation_app.close()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
