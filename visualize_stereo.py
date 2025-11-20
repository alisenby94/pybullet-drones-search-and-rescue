#!/usr/bin/env python3
"""
Visualize stereo vision depth data with matplotlib
"""
import sys
sys.path.insert(0, '/home/illicit/Repos/pybullet-drones-search-and-rescue/simulation/gym-pybullet-drones')

import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from stereo_vision import StereoVisionSystem, StereoConfig

def main():
    print("="*70)
    print("STEREO VISION VISUALIZATION")
    print("="*70)
    
    # Start PyBullet
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    
    # Create environment
    plane_shape = p.createCollisionShape(p.GEOM_PLANE)
    plane_id = p.createMultiBody(0, plane_shape)
    
    # Add objects at different distances
    box_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
    box1 = p.createMultiBody(1, box_shape, basePosition=[2, 0, 0.5])
    box2 = p.createMultiBody(1, box_shape, basePosition=[4, 1, 0.5])
    box3 = p.createMultiBody(1, box_shape, basePosition=[6, -1, 0.5])
    box4 = p.createMultiBody(1, box_shape, basePosition=[8, 0, 0.5])
    
    print("\n✓ Created environment with 4 boxes")
    
    # Initialize stereo system
    config = StereoConfig(
        baseline=0.1,
        img_width=256,
        img_height=256,
        fov=60.0
    )
    stereo = StereoVisionSystem(config)
    
    print(f"✓ Stereo system: {config.img_width}x{config.img_height}, baseline={config.baseline}m")
    
    # Simulate drone
    drone_pos = np.array([0.0, 0.0, 2.0])
    drone_quat = np.array([0, 0, 0, 1])
    
    print(f"✓ Drone position: {drone_pos}")
    print("\nCapturing images...")
    
    # Get depth map and images
    depth_map, images = stereo.get_depth_map(drone_pos, drone_quat, client, method='pybullet')
    
    print(f"✓ Captured data:")
    print(f"  Depth range: {np.min(depth_map):.2f}m to {np.max(depth_map):.2f}m")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Stereo Vision System Output', fontsize=16, fontweight='bold')
    
    # Left RGB
    ax = axes[0, 0]
    ax.imshow(images['left_rgb'])
    ax.set_title('Left Camera RGB')
    ax.axis('off')
    
    # Right RGB
    ax = axes[0, 1]
    ax.imshow(images['right_rgb'])
    ax.set_title('Right Camera RGB')
    ax.axis('off')
    
    # Depth map (colorized)
    ax = axes[0, 2]
    depth_display = np.clip(depth_map, 0, 20)  # Clip for better visualization
    im = ax.imshow(depth_display, cmap='jet', vmin=0, vmax=20)
    ax.set_title('Depth Map (0-20m)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Distance (m)')
    
    # Left segmentation
    ax = axes[1, 0]
    seg_display = images['left_seg'].astype(np.float32)
    seg_display[seg_display < 0] = 0  # Remove negative IDs
    ax.imshow(seg_display, cmap='tab20')
    ax.set_title('Left Segmentation')
    ax.axis('off')
    
    # Depth histogram
    ax = axes[1, 1]
    depth_flat = depth_map[depth_map < 50].flatten()  # Remove far clip values
    ax.hist(depth_flat, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Pixel Count')
    ax.set_title('Depth Distribution')
    ax.grid(True, alpha=0.3)
    
    # Depth cross-section (center row)
    ax = axes[1, 2]
    center_row = depth_map[config.img_height // 2, :]
    ax.plot(center_row, linewidth=2, color='red')
    ax.set_xlabel('Pixel Column')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Depth Cross-Section (Center Row)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 20])
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'stereo_vision_output.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_file}")
    
    # Additional analysis
    print("\n" + "="*70)
    print("DEPTH ANALYSIS")
    print("="*70)
    
    # Obstacle detection (anything closer than 5m in center region)
    h, w = depth_map.shape
    center_region = depth_map[h//3:2*h//3, w//3:2*w//3]
    min_center_depth = np.min(center_region)
    
    print(f"\nCenter region (33-66%):")
    print(f"  Closest object: {min_center_depth:.2f}m")
    print(f"  Average depth: {np.mean(center_region):.2f}m")
    
    if min_center_depth < 5.0:
        print(f"  ⚠️  WARNING: Obstacle detected at {min_center_depth:.2f}m!")
    else:
        print(f"  ✓ Clear path (>{min_center_depth:.2f}m)")
    
    # Generate point cloud stats
    point_cloud = stereo.get_point_cloud(depth_map, images['left_rgb'])
    print(f"\nPoint Cloud:")
    print(f"  Total points: {point_cloud.shape[0]:,}")
    print(f"  Data: {point_cloud.shape[1]} dimensions (XYZ + RGB)")
    
    distances = np.linalg.norm(point_cloud[:, :3], axis=1)
    print(f"  Distance range: {np.min(distances):.2f}m to {np.max(distances):.2f}m")
    
    # Count objects at different distances
    close = np.sum(distances < 3)
    medium = np.sum((distances >= 3) & (distances < 7))
    far = np.sum(distances >= 7)
    
    print(f"\nDistance distribution:")
    print(f"  Close (<3m): {close:,} points ({100*close/len(distances):.1f}%)")
    print(f"  Medium (3-7m): {medium:,} points ({100*medium/len(distances):.1f}%)")
    print(f"  Far (>7m): {far:,} points ({100*far/len(distances):.1f}%)")
    
    print("\n" + "="*70)
    print("✓ Visualization complete!")
    print(f"  Open '{output_file}' to see results")
    print("="*70)
    
    p.disconnect()
    plt.show()

if __name__ == "__main__":
    main()
