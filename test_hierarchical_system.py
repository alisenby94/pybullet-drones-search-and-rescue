"""
Test the coordinator agent with PID motor control.

Tests waypoint navigation with the trained coordinator agent.
The coordinator issues velocity commands which are executed by a PID controller.
"""

import numpy as np
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

def test_hierarchical_system():
    """Test the coordinator agent with PID motor control."""
    
    # Load coordinator model - no motor agent needed (uses PID controller)
    try:
        if HAS_RECURRENT:
            from sb3_contrib import RecurrentPPO as RPPO
            
            # Try to load the most recent coordinator model
            try:
                coord_model = RPPO.load("models/experiment_v4_final")
                print("✓ Loaded experiment_v4 coordinator model (RecurrentPPO with GRU)")
                print("  Motor control: PID controller (no trained model)\n")
            except FileNotFoundError:
                try:
                    coord_model = RPPO.load("models/coordinator_v1_final")
                    print("✓ Loaded coordinator_v1 coordinator model (RecurrentPPO with GRU)")
                    print("  Motor control: PID controller (no trained model)\n")
                except FileNotFoundError:
                    print("✗ No coordinator model found!")
                    print("  Train a model first: python -m src.train --timesteps 200000 --name experiment_v1")
                    return
        else:
            # No recurrent support
            try:
                coord_model = PPO.load("models/experiment_v1_final")
                print("✓ Loaded experiment_v1 coordinator model (PPO)")
                print("  Motor control: PID controller (no trained model)\n")
            except FileNotFoundError:
                print("✗ No coordinator model found!")
                print("  Train a model first: python -m src.train --timesteps 200000 --name experiment_v1")
                return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Create coordinator environment (no motor_model_path needed - uses PID)
    env = ActionCoordinatorEnv(
        gui=True,
        enable_obstacles=True,  # Show obstacles
        num_obstacles=10
    )
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Simple Forward',
            'waypoints': np.array([
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [4.0, 0.0, 1.0],
            ]),
            'duration': 1000
        },
        {
            'name': 'L-Shape Path',
            'waypoints': np.array([
                [0.0, 0.0, 1.0],
                [3.0, 0.0, 1.0],
                [3.0, 3.0, 1.0],
            ]),
            'duration': 1000
        },
        {
            'name': 'Square Path',
            'waypoints': np.array([
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [2.0, 2.0, 1.0],
                [0.0, 2.0, 1.0],
                [0.0, 0.0, 1.0],
            ]),
            'duration': 1000
        },
        {
            'name': 'Ascending Path',
            'waypoints': np.array([
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 1.5],
                [4.0, 0.0, 2.0],
                [6.0, 0.0, 2.5],
            ]),
            'duration': 1000
        },
    ]
    
    print("="*80)
    print("COORDINATOR AGENT TEST: Waypoint Navigation with PID Control")
    print("="*80)
    
    overall_results = []
    
    for scenario in test_scenarios:
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario['name']}")
        print(f"Waypoints: {len(scenario['waypoints'])}")
        for i, wp in enumerate(scenario['waypoints']):
            print(f"  WP{i}: [{wp[0]:+.1f}, {wp[1]:+.1f}, {wp[2]:+.1f}]")
        print(f"{'='*80}")
        
        # Reset with custom waypoints
        env.waypoints = scenario['waypoints']
        obs, info = env.reset()
        
        # Initialize LSTM state for RecurrentPPO models
        # For RecurrentPPO, we need to track episode_start and lstm_states
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        
        waypoints_reached = 0
        total_distance_to_waypoints = 0
        position_history = []
        velocity_commands = []
        step_count = 0
        terminated = False  # Initialize to avoid unbound variable
        truncated = False
        
        for step in range(scenario['duration']):
            # Get current state
            state = env._getDroneStateVector(0)
            pos = state[:3]
            vel = state[10:13]
            position_history.append(pos.copy())
            
            # Update camera to follow drone
            camera_distance = 1.0
            camera_yaw = 45
            camera_pitch = -30
            p.resetDebugVisualizerCamera(
                cameraDistance=camera_distance,
                cameraYaw=camera_yaw,
                cameraPitch=camera_pitch,
                cameraTargetPosition=pos.tolist()
            )
            
            # Get velocity command from coordinator
            # RecurrentPPO needs episode_start and lstm_states
            is_recurrent = HAS_RECURRENT and hasattr(coord_model, 'policy') and hasattr(coord_model.policy, 'lstm_actor')
            if is_recurrent:
                action, lstm_states = coord_model.predict(
                    obs, 
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True
                )
                episode_start = np.zeros((1,), dtype=bool)  # Only True on first step
            else:
                action, _ = coord_model.predict(obs, deterministic=True)
            velocity_commands.append(action.copy())
            
            # Execute action (coordinator commands motor controller internally)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Track waypoints reached
            current_wp = scenario['waypoints'][env.current_waypoint_idx]
            dist_to_wp = np.linalg.norm(pos - current_wp)
            total_distance_to_waypoints += dist_to_wp
            
            if env.waypoints_reached > waypoints_reached:
                waypoints_reached = env.waypoints_reached
                print(f"  ✓ Waypoint {waypoints_reached} reached at step {step}")
            
            # Print progress
            if step % 50 == 0:
                print(f"  Step {step:3d}: "
                      f"pos=[{pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}], "
                      f"vel_cmd=[{action[0]:+.2f}, {action[1]:+.2f}, {action[2]:+.2f}], "
                      f"dist_to_wp={dist_to_wp:.2f}m, "
                      f"reward={reward:+.2f}")
            
            time.sleep(0.02)  # Slow down for visualization
            
            if terminated:
                print(f"\n  Episode terminated at step {step}")
                print(f"  Reason: Crashed or out of bounds")
                break
            
            if truncated:
                print(f"\n  Episode truncated at step {step}")
                break
        
        # Calculate statistics
        completion_rate = waypoints_reached / len(scenario['waypoints']) * 100
        avg_dist_to_wp = total_distance_to_waypoints / step_count if step_count > 0 else 0
        
        # Calculate path smoothness (acceleration changes)
        velocity_commands = np.array(velocity_commands)
        if len(velocity_commands) > 1:
            vel_changes = np.diff(velocity_commands, axis=0)
            avg_vel_change = np.mean(np.linalg.norm(vel_changes, axis=1))
        else:
            avg_vel_change = 0
        
        # Calculate total distance traveled
        position_history = np.array(position_history)
        if len(position_history) > 1:
            distances = np.linalg.norm(np.diff(position_history, axis=0), axis=1)
            total_distance = np.sum(distances)
        else:
            total_distance = 0
        
        final_pos = position_history[-1] if len(position_history) > 0 else np.zeros(3)
        
        print(f"\n  --- Results ---")
        print(f"  Waypoints reached: {waypoints_reached}/{len(scenario['waypoints'])} ({completion_rate:.1f}%)")
        print(f"  Steps completed: {step_count}/{scenario['duration']}")
        print(f"  Average distance to waypoint: {avg_dist_to_wp:.2f}m")
        print(f"  Total distance traveled: {total_distance:.2f}m")
        print(f"  Average velocity change: {avg_vel_change:.3f} m/s/step")
        print(f"  Final position: [{final_pos[0]:+.2f}, {final_pos[1]:+.2f}, {final_pos[2]:+.2f}]")
        print(f"  Crashed: {'YES' if terminated else 'NO'}")
        
        # Evaluate performance
        if completion_rate >= 80 and not terminated:
            status = "✅ EXCELLENT"
        elif completion_rate >= 50 and not terminated:
            status = "✓ GOOD"
        elif completion_rate >= 30:
            status = "⚠ FAIR"
        else:
            status = "✗ POOR"
        print(f"  Performance: {status}")
        
        overall_results.append({
            'name': scenario['name'],
            'completion': completion_rate,
            'crashed': terminated,
            'avg_dist': avg_dist_to_wp
        })
        
        time.sleep(2.0)  # Pause between scenarios
    
    env.close()
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL COORDINATOR PERFORMANCE")
    print(f"{'='*80}")
    
    avg_completion = np.mean([r['completion'] for r in overall_results])
    crash_count = sum([1 for r in overall_results if r['crashed']])
    
    print(f"Average waypoint completion: {avg_completion:.1f}%")
    print(f"Scenarios crashed: {crash_count}/{len(overall_results)}")
    
    for result in overall_results:
        crash_marker = "💥" if result['crashed'] else "✓"
        print(f"  {crash_marker} {result['name']:20s}: {result['completion']:5.1f}% complete, "
              f"avg dist {result['avg_dist']:.2f}m")
    
    if avg_completion >= 70 and crash_count == 0:
        print("\n✅ EXCELLENT: Coordinator navigates waypoints successfully!")
    elif avg_completion >= 50 and crash_count <= 1:
        print("\n✓ GOOD: Coordinator shows solid navigation performance")
    elif avg_completion >= 30:
        print("\n⚠ FAIR: Coordinator needs improvement")
    else:
        print("\n✗ POOR: Coordinator struggles with waypoint navigation")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    test_hierarchical_system()
