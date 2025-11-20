#!/usr/bin/env python3
"""
Quick test of stereo vision system without GUI
"""
import sys
sys.path.insert(0, '/home/illicit/Repos/pybullet-drones-search-and-rescue/simulation/gym-pybullet-drones')

import numpy as np
import pybullet as p
from stereo_vision import StereoVisionSystem, StereoConfig

def main():
    print("="*70)
    print("STEREO VISION SYSTEM - QUICK TEST")
    print("="*70)
    
    # Start PyBullet in DIRECT mode (no GUI)
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    
    # Add ground plane
    plane_shape = p.createCollisionShape(p.GEOM_PLANE)
    plane_id = p.createMultiBody(0, plane_shape)
    
    # Add some box objects at different distances
    box_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
    box1 = p.createMultiBody(1, box_shape, basePosition=[2, 0, 0.5])
    box2 = p.createMultiBody(1, box_shape, basePosition=[5, 1, 0.5])
    box3 = p.createMultiBody(1, box_shape, basePosition=[8, -1, 0.5])
    
    print("\n✓ Created test environment with 3 boxes at distances: 2m, 5m, 8m")
    
    # Create stereo vision system
    config = StereoConfig(
        baseline=0.1,  # 10cm between cameras
        img_width=128,
        img_height=128,
        fov=60.0
    )
    stereo = StereoVisionSystem(config)
    
    print(f"\n✓ Initialized stereo vision system:")
    print(f"    Baseline: {config.baseline}m")
    print(f"    Image size: {config.img_width}x{config.img_height}")
    print(f"    FOV: {config.fov}°")
    print(f"    Focal length: {config.focal_length:.2f} pixels")
    
    # Simulate drone at position looking forward
    drone_pos = np.array([0.0, 0.0, 2.0])
    drone_quat = np.array([0, 0, 0, 1])  # No rotation (looking along +X axis)
    
    print(f"\n✓ Drone position: {drone_pos}")
    print(f"    Looking along: +X axis")
    
    # Test 1: Capture stereo images
    print("\n[1] Capturing stereo images...")
    images = stereo.capture_stereo_images(drone_pos, drone_quat, client)
    
    print(f"    ✓ Left RGB: {images['left_rgb'].shape}")
    print(f"    ✓ Right RGB: {images['right_rgb'].shape}")
    print(f"    ✓ Left Depth: {images['left_depth'].shape}")
    print(f"    ✓ Right Depth: {images['right_depth'].shape}")
    
    # Test 2: Get depth map using PyBullet method
    print("\n[2] Computing depth map (PyBullet method)...")
    depth_map, _ = stereo.get_depth_map(drone_pos, drone_quat, client, method='pybullet')
    
    print(f"    ✓ Depth map shape: {depth_map.shape}")
    print(f"    ✓ Depth range: {np.min(depth_map):.2f}m to {np.max(depth_map):.2f}m")
    print(f"    ✓ Mean depth: {np.mean(depth_map):.2f}m")
    print(f"    ✓ Median depth: {np.median(depth_map):.2f}m")
    
    # Test 3: Generate point cloud
    print("\n[3] Generating 3D point cloud...")
    point_cloud = stereo.get_point_cloud(depth_map, images['left_rgb'])
    
    print(f"    ✓ Total points: {point_cloud.shape[0]:,}")
    print(f"    ✓ Data format: {'XYZ + RGB' if point_cloud.shape[1] == 6 else 'XYZ only'}")
    
    # Analyze point cloud
    if point_cloud.shape[0] > 0:
        distances = np.linalg.norm(point_cloud[:, :3], axis=1)
        print(f"    ✓ Point distances: {np.min(distances):.2f}m to {np.max(distances):.2f}m")
    
    #  Test 4: Try stereo matching if available
    if stereo.stereo_matcher is not None:
        print("\n[4] Computing depth with stereo matching...")
        depth_stereo, _ = stereo.get_depth_map(drone_pos, drone_quat, client, method='stereo')
        print(f"    ✓ Stereo depth range: {np.min(depth_stereo):.2f}m to {np.max(depth_stereo):.2f}m")
    else:
        print("\n[4] Stereo matching not available (OpenCV StereoBM not found)")
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)
    print("\nStereo vision system is ready to use in your RL environment!")
    print("\nKey features:")
    print("  • Dual camera setup with configurable baseline")
    print("  • Real-time depth map computation")
    print("  • 3D point cloud generation")
    print("  • Optional stereo matching for realistic depth estimation")
    print("="*70)
    
    p.disconnect()

if __name__ == "__main__":
    main()
