"""
Test ActionCoordinatorEnv V4 - Car-Like Control

Test a trained V4 model (2DOF: forward/backward + yaw only).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simulation/gym-pybullet-drones'))

import argparse
import numpy as np
from stable_baselines3 import PPO

from src.envs.action_coordinator_env_v4 import ActionCoordinatorEnvV4


def test_v4_model(model_path, num_episodes=3, gui=True):
    """
    Test V4 trained model.
    
    Args:
        model_path: Path to model .zip file
        num_episodes: Number of episodes to run
        gui: Show PyBullet GUI
    """
    print("="*80)
    print("TESTING V4 MODEL - CAR-LIKE CONTROL")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Action space: 2D [vx, yaw_delta] (no lateral movement)\n")
    
    # Load model
    print("Loading model...")
    model = PPO.load(model_path)
    print("✓ Model loaded\n")
    
    # Create environment
    env = ActionCoordinatorEnvV4(
        gui=gui,
        enable_obstacles=True,
        max_episode_steps=1000
    )
    
    # Disable debug GUI panels if GUI enabled
    if gui:
        import pybullet as p
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env.CLIENT)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=env.CLIENT)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=env.CLIENT)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=env.CLIENT)
        
    
    # Test episodes
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        obs, info = env.reset()
        episode_reward = 0
        step = 0
        
        print(f"Waypoints: {len(env.waypoints)}")
        print(f"Starting position: {info['position'][:2]}")
        print(f"First waypoint: {env.waypoints[0][:2]}\n")
        
        while True:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Print progress every second
            if step % 48 == 0:
                print(f"[{step/48:.1f}s] "
                      f"WP: {info['waypoint_idx']}/{len(env.waypoints)}, "
                      f"Dist: {info['waypoint_distance']:.2f}m, "
                      f"Action: [{action[0]:+.2f}, {action[1]:+.2f}], "
                      f"Reward: {episode_reward:.1f}")
            
            # Check termination
            if terminated or truncated:
                print(f"\n{'─'*60}")
                print(f"Episode ended at step {step} ({step/48:.1f}s)")
                print(f"{'─'*60}")
                
                if terminated:
                    if info.get('crash_type'):
                        print(f"❌ Crashed: {info['crash_type']}")
                    elif env.current_waypoint_idx >= len(env.waypoints):
                        print(f"✓ All waypoints reached!")
                
                if truncated:
                    print(f"⏱ Time limit reached")
                
                print(f"\nStatistics:")
                print(f"  Waypoints reached: {info['waypoints_reached']}/{len(env.waypoints)}")
                print(f"  Total reward: {episode_reward:.2f}")
                print(f"  Average reward: {episode_reward/step:.2f}")
                
                break
    
    env.close()
    print(f"\n{'='*80}")
    print("Testing complete!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test V4 car-like control model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model .zip file')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to run')
    parser.add_argument('--no-gui', action='store_true',
                        help='Disable PyBullet GUI')
    
    args = parser.parse_args()
    
    test_v4_model(
        model_path=args.model,
        num_episodes=args.episodes,
        gui=not args.no_gui
    )
