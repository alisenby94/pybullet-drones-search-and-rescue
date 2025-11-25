"""
Test the coordinator agent with PID motor control.

Tests waypoint navigation with the trained coordinator agent.
The coordinator issues velocity commands which are executed by a PID controller.
"""

import numpy as np
import argparse
from src.envs.action_coordinator_env import ActionCoordinatorEnv
from stable_baselines3 import PPO
try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT = True
except ImportError:
    HAS_RECURRENT = False
    print("Warning: sb3-contrib not installed, cannot load RecurrentPPO models")
import time
import pybullet as p
from pathlib import Path

def test_hierarchical_system(model_path=None):
    """Test the trained coordinator agent."""
    print("\n" + "="*80)
    print("COORDINATOR AGENT TEST: Waypoint Navigation with Direct RPM Control")
    print("="*80)
    
    # Load the trained coordinator model
    if model_path:
        # Use specified model path
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"\nError: Specified model not found: {model_path}")
            return
        print(f"\nLoading specified model: {model_path}")
        coord_model = RecurrentPPO.load(model_path)
    else:
        # Try to load the latest experiment
        model_dir = Path("models/action_coordinator")
        latest_model = None
        for exp_num in range(20, 0, -1):  # Try v20 down to v1
            model_path = model_dir / f"experiment_v{exp_num}" / "best_model.zip"
            if model_path.exists():
                latest_model = model_path
                print(f"\nLoading model: {model_path}")
                break
        
        if latest_model is None:
            print("\nNo trained model found!")
            return
        
        coord_model = RecurrentPPO.load(latest_model)
    
    # Create test environment with GUI and vision
    env = ActionCoordinatorEnv(
        gui=True,
        enable_vision=True,
        enable_streaming=True,
        num_obstacles=10,
        enable_obstacles=True
    )
    
    print(f"Environment: {env.observation_space.shape[0]}D obs ({10 + (512 if env.enable_vision else 0)}D)")
    print(f"Action space: Direct RPM commands [rpm0, rpm1, rpm2, rpm3]")
    print(f"RPM range: Hover ± 5% ({int(env.HOVER_RPM * 0.95)} - {int(env.HOVER_RPM * 1.05)} RPM)")
    print("="*80)
    
    # Waypoint visualization markers
    waypoint_markers = []
    
    def draw_waypoint_markers(waypoints, current_idx):
        """Draw visual markers for waypoints in PyBullet GUI."""
        nonlocal waypoint_markers
        
        # Remove old markers
        for marker_id in waypoint_markers:
            try:
                p.removeBody(marker_id)
            except:
                pass
        waypoint_markers.clear()
        
        # Draw new markers
        for i, wp in enumerate(waypoints):
            if i == current_idx:
                # Current waypoint: Green sphere
                color = [0, 1, 0, 0.8]  # Green, semi-transparent
                radius = 0.3
            elif i < current_idx:
                # Reached waypoints: Gray sphere
                color = [0.5, 0.5, 0.5, 0.3]  # Gray, very transparent
                radius = 0.2
            else:
                # Future waypoints: Blue sphere
                color = [0, 0.5, 1, 0.5]  # Blue, semi-transparent
                radius = 0.25
            
            # Create visual shape
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=radius,
                rgbaColor=color,
                physicsClientId=env.CLIENT
            )
            
            # Create marker body (no collision)
            marker_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape,
                basePosition=wp,
                physicsClientId=env.CLIENT
            )
            waypoint_markers.append(marker_id)
    
    # Run test scenarios
    test_scenarios = [
        {
            'name': 'Random Waypoints with Obstacles (Training-like)',
            'num_runs': 3
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*80}\n")
        
        for run_idx in range(scenario['num_runs']):
            print(f"{'='*80}")
            print(f"Test run: {run_idx + 1}/{scenario['num_runs']}")
            print(f"{'='*80}")
            
            # Reset environment (generates random waypoints)
            obs, info = env.reset()
            
            # Draw waypoint markers
            draw_waypoint_markers(env.waypoints, env.current_waypoint_idx)
            
            # Print waypoints
            print(f"Waypoints: {len(env.waypoints)}")
            for i, wp in enumerate(env.waypoints):
                print(f"  WP{i}: [{wp[0]:+.1f}, {wp[1]:+.1f}, {wp[2]:+.1f}]")
            print(f"{'='*80}\n")
            
            # Initialize LSTM states
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)
            
            step_count = 0
            total_reward = 0
            waypoints_reached = 0
            
            while step_count < 1000:  # Max 1000 steps per test
                # Get action from coordinator model
                action, lstm_states = coord_model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True
                )
                episode_start = np.zeros((1,), dtype=bool)
                
                # Execute action
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Update waypoint markers if waypoint changed
                if info.get('waypoint_reached', False):
                    draw_waypoint_markers(env.waypoints, env.current_waypoint_idx)
                    waypoints_reached += 1
                
                total_reward += reward
                step_count += 1
                
                # Print progress
                pos = env._getDroneStateVector(0)[0:3]
                current_wp = env.waypoints[env.current_waypoint_idx]
                dist = np.linalg.norm(pos - current_wp)
                
                # Update camera to follow drone
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.0,
                    cameraYaw=50,
                    cameraPitch=-35,
                    cameraTargetPosition=pos,
                    physicsClientId=env.CLIENT
                )
                
                #disabled detailed step printout to reduce console spam
                # print(f"Step {step_count:4d} | "
                #       f"Pos: [{pos[0]:+5.2f}, {pos[1]:+5.2f}, {pos[2]:+5.2f}] | "
                #       f"WP{env.current_waypoint_idx}: [{current_wp[0]:+5.2f}, {current_wp[1]:+5.2f}, {current_wp[2]:+5.2f}] | "
                #       f"Dist: {dist:5.2f}m | "
                #       f"Reward: {reward:+7.2f} | "
                #       f"Total: {total_reward:+8.2f}")
                
                if terminated or truncated:
                    break
                
                time.sleep(0.1)  # Slow down for visualization
            
            # Print summary
            print(f"\n{'='*80}")
            print(f"Test Run {run_idx + 1} Summary:")
            print(f"  Steps: {step_count}")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Waypoints Reached: {waypoints_reached}/{len(env.waypoints)}")
            print(f"  Terminated: {terminated}")
            print(f"  Truncated: {truncated}")
            if 'crashed' in info:
                print(f"  Crashed: {info['crashed']}")
            print(f"{'='*80}\n")
            
            time.sleep(2.0)  # Pause between runs
    
    # Cleanup
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test hierarchical system with trained models")
    parser.add_argument("--model", type=str, default=None, 
                        help="Path to specific model file (e.g., models/action_coordinator/experiment_v13/best_model.zip)")
    args = parser.parse_args()
    
    test_hierarchical_system(model_path=args.model)
