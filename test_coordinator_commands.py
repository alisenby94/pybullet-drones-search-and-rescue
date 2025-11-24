"""Test script to see what velocity commands the coordinator generates for different waypoints."""

import numpy as np
from src.envs.action_coordinator_env import ActionCoordinatorEnv
from stable_baselines3 import PPO
import time

def test_coordinator_commands():
    """Test coordinator with various waypoint positions."""
    
    # Load the trained coordinator model
    try:
        model = PPO.load("models/bodyframe_v1_coordinator_final")
        print("✓ Loaded final coordinator model")
    except:
        try:
            # Try to find the latest checkpoint
            import glob
            checkpoints = sorted(glob.glob("models/bodyframe_v1_coordinator_*"))
            if checkpoints:
                model = PPO.load(checkpoints[-1])
                print(f"✓ Loaded coordinator checkpoint: {checkpoints[-1]}")
            else:
                print("✗ No coordinator model found. Train first with: python -m src.train")
                return
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return
    
    # Create environment with GUI for visualization
    env = ActionCoordinatorEnv(gui=True, motor_model_path="models/bodyframe_v1_motor_final")
    
    # Test waypoints (world coordinates relative to start position [0, 0, 1])
    test_waypoints = [
        ([2.0, 0.0, 1.0], "2m forward"),
        ([0.0, 2.0, 1.0], "2m right"),
        ([-2.0, 0.0, 1.0], "2m backward"),
        ([0.0, -2.0, 1.0], "2m left"),
        ([2.0, 2.0, 1.0], "2m forward-right diagonal"),
        ([0.0, 0.0, 2.0], "1m up"),
        ([3.0, 0.0, 0.5], "3m forward, 0.5m down"),
        ([1.0, 1.0, 1.5], "1m forward-right, 0.5m up"),
    ]
    
    print("\n" + "="*80)
    print("COORDINATOR VELOCITY COMMAND TEST")
    print("="*80)
    
    for waypoint_world, description in test_waypoints:
        print(f"\n--- Testing waypoint: {description} ---")
        print(f"World coordinates: {waypoint_world}")
        
        # Reset environment and set the target waypoint
        obs, info = env.reset()
        env.waypoints = [np.array(waypoint_world)]
        env.current_waypoint_idx = 0
        
        # Get drone's starting position and orientation
        pos = env._getDroneStateVector(0)[:3]
        rpy = env._getDroneStateVector(0)[7:10]
        print(f"Drone position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        print(f"Drone orientation (RPY): [{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}]")
        
        # Calculate distance and direction
        direction_world = np.array(waypoint_world) - pos
        distance = np.linalg.norm(direction_world)
        print(f"Distance to waypoint: {distance:.3f} m")
        print(f"Direction (world frame): [{direction_world[0]:.3f}, {direction_world[1]:.3f}, {direction_world[2]:.3f}]")
        
        # Transform to body frame for reference
        yaw = rpy[2]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        direction_body = np.array([
            cos_yaw * direction_world[0] + sin_yaw * direction_world[1],
            -sin_yaw * direction_world[0] + cos_yaw * direction_world[1],
            direction_world[2]
        ])
        print(f"Direction (body frame): [{direction_body[0]:.3f}, {direction_body[1]:.3f}, {direction_body[2]:.3f}]")
        
        # Get velocity command from coordinator
        action, _ = model.predict(obs, deterministic=True)
        
        # Denormalize action to actual velocity command
        vel_cmd = action.copy()
        vel_cmd[0:3] = vel_cmd[0:3] * 1.0  # Linear velocity in m/s
        vel_cmd[3] = vel_cmd[3] * (np.pi / 6)  # Angular velocity in rad/s
        
        print(f"\n*** Coordinator Output ***")
        print(f"Normalized action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}]")
        print(f"Velocity command (body frame):")
        print(f"  vx: {vel_cmd[0]:+.3f} m/s (forward/backward)")
        print(f"  vy: {vel_cmd[1]:+.3f} m/s (right/left)")
        print(f"  vz: {vel_cmd[2]:+.3f} m/s (up/down)")
        print(f"  wz: {vel_cmd[3]:+.3f} rad/s (yaw rate)")
        
        # Execute a few steps to see if it starts moving correctly
        print("\nExecuting 5 steps to observe initial movement...")
        for step in range(5):
            obs, reward, terminated, truncated, info = env.step(action)
            new_pos = env._getDroneStateVector(0)[:3]
            displacement = new_pos - pos
            print(f"  Step {step+1}: pos=[{new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f}], "
                  f"moved [{displacement[0]:+.3f}, {displacement[1]:+.3f}, {displacement[2]:+.3f}], "
                  f"reward={reward:.2f}")
            pos = new_pos
            time.sleep(0.1)
            
            # Get new action based on updated observation
            action, _ = model.predict(obs, deterministic=True)
            
            if terminated or truncated:
                print(f"  Episode ended: terminated={terminated}, truncated={truncated}")
                break
        
        time.sleep(1.0)  # Pause between tests
    
    env.close()
    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)

if __name__ == "__main__":
    test_coordinator_commands()
