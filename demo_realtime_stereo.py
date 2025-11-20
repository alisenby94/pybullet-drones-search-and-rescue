#!/usr/bin/env python3
"""
Real-time stereo vision demonstration with drone flight
Shows FPS, depth updates, and obstacle detection during flight
"""
import sys
sys.path.insert(0, '/home/illicit/Repos/pybullet-drones-search-and-rescue/simulation/gym-pybullet-drones')

import numpy as np
import pybullet as p
import time
from stereo_vision import StereoVisionSystem, StereoConfig

def main():
    print("="*70)
    print("REAL-TIME STEREO VISION DEMO")
    print("="*70)
    
    # Start PyBullet with GUI
    client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)
    
    # Create environment
    print("\nCreating environment...")
    plane_shape = p.createCollisionShape(p.GEOM_PLANE)
    plane_id = p.createMultiBody(0, plane_shape)
    
    # Add obstacles at various distances
    box_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
    obstacles = []
    positions = [
        ([3, 0, 0.5], [1, 0, 0, 1]),
        ([5, 2, 0.5], [0, 1, 0, 1]),
        ([7, -2, 0.5], [0, 0, 1, 1]),
        ([9, 1, 1.0], [1, 1, 0, 1]),
        ([11, -1, 0.5], [1, 0, 1, 1]),
    ]
    
    for pos, color in positions:
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=color)
        obj = p.createMultiBody(1, box_shape, vis, pos)
        obstacles.append(obj)
    
    print(f"✓ Created {len(obstacles)} obstacles")
    
    # Initialize stereo vision system
    # Test different resolutions to see performance impact
    configs = {
        'low': StereoConfig(img_width=64, img_height=64, baseline=0.08),
        'medium': StereoConfig(img_width=128, img_height=128, baseline=0.08),
        'high': StereoConfig(img_width=256, img_height=256, baseline=0.08)
    }
    
    # Start with medium resolution
    resolution = 'medium'
    config = configs[resolution]
    stereo = StereoVisionSystem(config)
    
    print(f"\n✓ Stereo system initialized:")
    print(f"  Resolution: {config.img_width}x{config.img_height}")
    print(f"  Baseline: {config.baseline}m")
    print(f"  FOV: {config.fov}°")
    
    # Drone visualization (simple sphere to represent drone)
    drone_visual = p.createVisualShape(
        p.GEOM_SPHERE, 
        radius=0.1, 
        rgbaColor=[0, 1, 0, 0.8]
    )
    drone_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
    drone_id = p.createMultiBody(0.1, drone_collision, drone_visual, [0, 0, 2])
    
    # Camera indicators (small cubes for left/right cameras)
    cam_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[1, 0, 0, 1])
    left_cam_id = p.createMultiBody(0, -1, cam_shape)
    right_cam_id = p.createMultiBody(0, -1, cam_shape)
    
    print("\n✓ Drone created")
    print("\nControls:")
    print("  Press 1, 2, 3 to change resolution (64, 128, 256)")
    print("  Press Q to quit")
    print("  Drone will fly forward automatically")
    print("\n" + "="*70)
    
    # Flight parameters
    drone_pos = np.array([0.0, 0.0, 2.0])
    drone_quat = np.array([0, 0, 0, 1])  # Facing +X
    velocity = 1.0  # m/s forward
    
    # Performance tracking
    frame_times = []
    depth_times = []
    max_history = 100
    
    # Flight path
    flight_time = 0
    dt = 1./240.  # PyBullet timestep
    
    print("\nStarting real-time flight...")
    print("Frame | FPS  | Depth(ms) | Min Dist | Obstacles | Status")
    print("-" * 70)
    
    frame_count = 0
    last_print = time.time()
    
    try:
        while drone_pos[0] < 12.0:  # Fly until 12m forward
            frame_start = time.time()
            
            # Update drone position
            drone_pos[0] += velocity * dt
            p.resetBasePositionAndOrientation(drone_id, drone_pos, drone_quat)
            
            # Update camera visualizations
            rot_mat = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)
            half_baseline = config.baseline / 2.0
            left_offset = rot_mat @ np.array([0, -half_baseline, 0])
            right_offset = rot_mat @ np.array([0, half_baseline, 0])
            
            p.resetBasePositionAndOrientation(left_cam_id, drone_pos + left_offset, [0, 0, 0, 1])
            p.resetBasePositionAndOrientation(right_cam_id, drone_pos + right_offset, [0, 0, 0, 1])
            
            # Capture stereo vision data (this is the real-time part!)
            depth_start = time.time()
            depth_map, images = stereo.get_depth_map(
                drone_pos, drone_quat, client, method='pybullet'
            )
            depth_time = (time.time() - depth_start) * 1000  # ms
            
            # Analyze depth data for obstacles
            h, w = depth_map.shape
            center_region = depth_map[h//3:2*h//3, w//3:2*w//3]
            min_depth = np.min(center_region)
            avg_depth = np.mean(center_region)
            
            # Count obstacles in view (depth < 15m)
            close_pixels = np.sum(depth_map < 15.0)
            obstacles_in_view = int(close_pixels / (h * w / 10))  # Rough estimate
            
            # Determine status
            if min_depth < 2.0:
                status = "🚨 DANGER"
            elif min_depth < 4.0:
                status = "⚠️  CAUTION"
            else:
                status = "✓ CLEAR"
            
            # Step simulation
            p.stepSimulation()
            
            # Calculate frame time
            frame_time = (time.time() - frame_start) * 1000  # ms
            frame_times.append(frame_time)
            depth_times.append(depth_time)
            
            if len(frame_times) > max_history:
                frame_times.pop(0)
                depth_times.pop(0)
            
            # Print stats every 0.5 seconds
            frame_count += 1
            if time.time() - last_print > 0.5:
                avg_frame_time = np.mean(frame_times)
                fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0
                avg_depth_time = np.mean(depth_times)
                
                print(f"{frame_count:5d} | {fps:4.1f} | {avg_depth_time:9.2f} | "
                      f"{min_depth:8.2f} | {obstacles_in_view:9d} | {status}")
                
                last_print = time.time()
            
            # Check for keyboard input
            keys = p.getKeyboardEvents()
            if ord('q') in keys or ord('Q') in keys:
                print("\nUser quit")
                break
            elif ord('1') in keys:
                resolution = 'low'
                config = configs[resolution]
                stereo = StereoVisionSystem(config)
                print(f"\n→ Switched to LOW resolution (64x64)")
            elif ord('2') in keys:
                resolution = 'medium'
                config = configs[resolution]
                stereo = StereoVisionSystem(config)
                print(f"\n→ Switched to MEDIUM resolution (128x128)")
            elif ord('3') in keys:
                resolution = 'high'
                config = configs[resolution]
                stereo = StereoVisionSystem(config)
                print(f"\n→ Switched to HIGH resolution (256x256)")
            
            flight_time += dt
            
            # Slow down for visualization
            time.sleep(dt / 2)  # Run at 2x realtime for better viewing
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    # Final statistics
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    
    if len(frame_times) > 0:
        print(f"\nTotal frames: {frame_count}")
        print(f"Flight time: {flight_time:.2f}s")
        print(f"Distance traveled: {drone_pos[0]:.2f}m")
        
        print(f"\nFrame timing:")
        print(f"  Total frame time: {np.mean(frame_times):.2f}ms (avg)")
        print(f"  Min/Max: {np.min(frame_times):.2f}ms / {np.max(frame_times):.2f}ms")
        print(f"  Average FPS: {1000.0/np.mean(frame_times):.1f}")
        
        print(f"\nDepth computation timing:")
        print(f"  Depth time: {np.mean(depth_times):.2f}ms (avg)")
        print(f"  Min/Max: {np.min(depth_times):.2f}ms / {np.max(depth_times):.2f}ms")
        print(f"  % of frame time: {100*np.mean(depth_times)/np.mean(frame_times):.1f}%")
        
        print(f"\nResolution: {config.img_width}x{config.img_height}")
        print(f"Baseline: {config.baseline}m")
        
        # Calculate if real-time capable
        max_acceptable_time = 33.33  # 30 FPS threshold
        if np.mean(frame_times) < max_acceptable_time:
            print(f"\n✓ REAL-TIME CAPABLE: {1000.0/np.mean(frame_times):.1f} FPS (>30 required)")
        else:
            print(f"\n⚠️  NOT REAL-TIME: {1000.0/np.mean(frame_times):.1f} FPS (<30 required)")
        
        print("\nConclusion:")
        print(f"  Stereo vision adds ~{np.mean(depth_times):.2f}ms per frame")
        print(f"  Suitable for RL training: {'YES ✓' if np.mean(depth_times) < 10 else 'MARGINAL' if np.mean(depth_times) < 20 else 'NO ✗'}")
        
    print("\n" + "="*70)
    
    p.disconnect()

if __name__ == "__main__":
    main()
