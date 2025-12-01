"""
Test Stereovision System

Usage: python test_stereo_vision.py [--gui] [--stream]
"""
import argparse
import numpy as np
import time
import pybullet as p
from src.envs.action_coordinator_env import ActionCoordinatorEnv


def test_stereo_vision(gui=False, enable_streaming=False):
    print("="*60)
    print("STEREOVISION SYSTEM TEST")
    print("="*60)
    
    env = ActionCoordinatorEnv(gui=gui, enable_vision=True, enable_streaming=enable_streaming)
    obs, info = env.reset()
    
    # Add a red cube obstacle 1 meter in front of the drone (straight ahead in +X direction)
    # Since drone starts at yaw=0, +X is forward
    drone_pos = np.array([0.0, 0.0, 1.75])  # Known initial position
    obstacle_x = drone_pos[0] + 1.0  # 1m in front (drone's +X direction)
    obstacle_y = drone_pos[1]
    obstacle_z = drone_pos[2]  # Same height as drone
    
    # Create cube (0.3m x 0.3m x 0.3m) - using p.connect default (assumes single physics server)
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.15])
    visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.15, 0.15, 0.15],
        rgbaColor=[1.0, 0.0, 0.0, 1.0]  # Bright red
    )
    obstacle_id = p.createMultiBody(
        baseMass=0,  # Static obstacle
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[obstacle_x, obstacle_y, obstacle_z]
    )
    
    print(f"\n✅ Added red cube obstacle:")
    print(f"   Position: ({obstacle_x:.2f}, {obstacle_y:.2f}, {obstacle_z:.2f})")
    print(f"   Distance from drone: 1.0m (straight ahead)")
    print(f"   Size: 0.3m × 0.3m × 0.3m\n")
    
    # Let physics settle
    for _ in range(20):
        p.stepSimulation()
    time.sleep(0.1)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Expected: (2058,) = 10D sensors + 2048D vision")
    
    if obs.shape[0] == 2058:
        print("✅ Observation shape correct!")
    else:
        print(f"❌ Wrong shape! Got {obs.shape}")
        return False
    
    for step in range(50):
        action = np.zeros(4)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            vision_obs = obs[10:]
            print(f"Step {step}: reward={reward:.2f}, vision_mean={np.mean(vision_obs):.3f}m")
        
        if terminated or truncated:
            break
    
    env.close()
    print("✅ Test complete!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    test_stereo_vision(gui=args.gui, enable_streaming=args.stream)
