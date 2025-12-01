"""
Test the V2 coordinator agent with PID motor control and heading control.

Tests waypoint navigation with the trained V2 coordinator agent.
The coordinator issues velocity commands which are executed by VelocityAviary's PID controller
with heading control for stereovision.
"""

import numpy as np
import argparse
from src.envs.action_coordinator_env_v2 import ActionCoordinatorEnvV2
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
import cv2

def test_v2_system(model_path=None):
    """Test the trained V2 coordinator agent."""
    print("\n" + "="*80)
    print("V2 COORDINATOR AGENT TEST: Waypoint Navigation with Heading Control")
    print("="*80)
    
    # Load the trained coordinator model
    if model_path:
        # Use specified model path
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"\nError: Specified model not found: {model_path}")
            return
        print(f"\nLoading specified model: {model_path}")
        
        # Try loading as RecurrentPPO first, then PPO
        try:
            coord_model = RecurrentPPO.load(model_path)
            print(f"  ✓ Loaded as RecurrentPPO")
        except:
            coord_model = PPO.load(model_path)
            print(f"  ✓ Loaded as PPO")
    else:
        # Try to load the latest v2 model
        model_dir = Path("models")
        latest_model = None
        
        # Look for v2_* models
        for model_file in sorted(model_dir.glob("v2_*_final.zip"), reverse=True):
            latest_model = model_file
            print(f"\nLoading model: {model_file}")
            break
        
        if latest_model is None:
            print("\nNo trained V2 model found!")
            print("Train a model first with: python train_v2.py --timesteps 100000 --name v2_baseline")
            return
        
        try:
            coord_model = RecurrentPPO.load(latest_model)
            print(f"  ✓ Loaded as RecurrentPPO")
        except:
            coord_model = PPO.load(latest_model)
            print(f"  ✓ Loaded as PPO")
    
    # Create test environment with GUI and HTTP streaming enabled
    env = ActionCoordinatorEnvV2(gui=True, enable_streaming=True)
    
    print(f"\nEnvironment Configuration:")
    print(f"  Observation space: {env.observation_space.shape[0]}D")
    print(f"    - Velocity (3D)")
    print(f"    - Yaw (1D)")
    print(f"    - Vector to waypoint (3D)")
    print(f"    - Stereovision depth features (512D)")
    print(f"  Action space: [vx, vy, vz] in [-1, 1]")
    print(f"    - vx/vy/vz: velocity in m/s (mapped to ±{env.SPEED_LIMIT} m/s)")
    print(f"    - yaw: AUTO-ALIGNED by PID to face velocity direction")
    print(f"  Speed limit: {env.SPEED_LIMIT} m/s (magnitude clipped)")
    print(f"  Control frequency: {env.CTRL_FREQ} Hz")
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
            'name': 'Random Waypoint Navigation',
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
            
            # Initialize LSTM states if using RecurrentPPO
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)
            use_lstm = hasattr(coord_model, 'policy') and hasattr(coord_model.policy, 'lstm_actor')
            
            step_count = 0
            total_reward = 0
            waypoints_reached = 0
            terminated = False
            truncated = False
            
            # Track timing for framerate calculation
            start_time = time.time()

            while step_count < 10000:  # Max 10000 steps per test
                # Get action from coordinator model
                if use_lstm:
                    action, lstm_states = coord_model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=episode_start,
                        deterministic=True
                    )
                    episode_start = np.zeros((1,), dtype=bool)
                else:
                    action, _ = coord_model.predict(obs, deterministic=True)
                
                print(f"\nPredicted action: {action}")

                # Execute action
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Update waypoint markers if waypoint changed
                current_wp_count = info['waypoints_reached']
                if current_wp_count > waypoints_reached:
                    draw_waypoint_markers(env.waypoints, env.current_waypoint_idx)
                    waypoints_reached = current_wp_count
                
                total_reward += reward
                step_count += 1
                
                # Get current state for visualization
                state = env._getDroneStateVector(0)
                pos = state[0:3]
                vel = state[10:13]
                rpy = state[7:10]
                yaw = rpy[2]
                quat = state[3:7]
                
                # Current waypoint info
                current_wp = env.waypoints[env.current_waypoint_idx]
                dist = info['waypoint_distance']
                
                # Update camera to follow drone from behind
                # Convert yaw from radians to degrees and add 270 to view from behind
                camera_yaw = np.degrees(yaw) + 270
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.0,  # Distance behind drone
                    cameraYaw=camera_yaw,  # Follow drone's heading
                    cameraPitch=-20,  # Slightly above looking down
                    cameraTargetPosition=pos,
                    physicsClientId=env.CLIENT
                )
                
                # Format status bar
                # Action is [vx, vy, vz] all in [-1, 1] (yaw auto-aligned by PID)
                action_flat = action.flatten() if action.ndim > 1 else action
                
                status = (
                    f"\r[Step {step_count:4d}] "
                    f"Pos:[{pos[0]:+5.2f},{pos[1]:+5.2f},{pos[2]:+5.2f}] "
                    f"Vel:[{vel[0]:+5.2f},{vel[1]:+5.2f},{vel[2]:+5.2f}] "
                    f"Yaw:{np.degrees(yaw):+6.1f}° "
                    f"ToWP:[{(current_wp-pos)[0]:+5.2f},{(current_wp-pos)[1]:+5.2f},{(current_wp-pos)[2]:+5.2f}] "
                    f"Dist:{dist:5.2f}m "
                    f"Action:[vx={action_flat[0]:+.2f},vy={action_flat[1]:+.2f},vz={action_flat[2]:+.2f}] "
                    f"WP:{waypoints_reached}/{len(env.waypoints)} "
                    f"R:{reward:+7.2f} "
                    f"TR:{total_reward:+8.2f}"
                )
                print(status, end='', flush=True)
                
                if terminated or truncated:
                    break
                
                time.sleep(0.05)  # Slow down for visualization
            
            # Print newline after status bar before summary
            print()  # Move to next line
            
            # Calculate timing metrics
            end_time = time.time()
            elapsed_time = end_time - start_time
            avg_fps = step_count / elapsed_time if elapsed_time > 0 else 0
            avg_step_time = (elapsed_time / step_count * 1000) if step_count > 0 else 0
            
            # Print summary
            print(f"\n{'='*80}")
            print(f"Test Run {run_idx + 1} Summary:")
            print(f"  Steps: {step_count}")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Average Reward: {total_reward/step_count:.2f}")
            print(f"  Waypoints Reached: {waypoints_reached}/{len(env.waypoints)}")
            print(f"  Success Rate: {waypoints_reached/len(env.waypoints)*100:.1f}%")
            print(f"  Terminated: {terminated}")
            print(f"  Truncated: {truncated}")
            
            # Timing and framerate stats
            print(f"\n  Performance:")
            print(f"    Elapsed time: {elapsed_time:.2f} seconds")
            print(f"    Average FPS: {avg_fps:.1f} steps/sec")
            print(f"    Average step time: {avg_step_time:.1f} ms/step")
            print(f"    Expected FPS (48 Hz): 48.0 steps/sec")
            print(f"    Time dilation: {avg_fps/48.0:.2f}x" if avg_fps > 0 else "    Time dilation: N/A")
            
            # Velocity tracking stats
            print(f"\n  Velocity Tracking:")
            print(f"    Target velocity: {info['target_velocity_mag']:.3f} m/s")
            print(f"    Actual velocity: {info['actual_velocity_mag']:.3f} m/s")
            print(f"    Tracking error: {info['velocity_tracking_error']:.3f} m/s")
            
            print(f"{'='*80}\n")
            
            time.sleep(2.0)  # Pause between runs
    
    # Cleanup
    env.close()
    
    print("\n" + "="*80)
    print("✅ V2 COORDINATOR TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test V2 coordinator with trained models")
    parser.add_argument("--model", type=str, default=None, 
                        help="Path to specific model file (e.g., models/v2_test_training_final.zip)")
    args = parser.parse_args()
    
    test_v2_system(model_path=args.model)
