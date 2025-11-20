#!/usr/bin/env python3
"""
Real-time stereo vision with stable rendering and matplotlib display
Camera controls: Hold Ctrl+Alt and use mouse to move PyBullet viewport
"""
import sys
sys.path.insert(0, '/home/illicit/Repos/pybullet-drones-search-and-rescue/simulation/gym-pybullet-drones')

import numpy as np
import pybullet as p
import time
import matplotlib.pyplot as plt
from stereo_vision import StereoVisionSystem, StereoConfig

def setup_debug_camera(client_id):
    """
    Setup debug camera with better default position
    PyBullet camera controls: Hold Ctrl+Alt then:
    - Mouse Left: Rotate view
    - Mouse Middle: Pan view
    - Mouse Right: Zoom
    - Mouse Wheel: Zoom in/out
    """
    p.resetDebugVisualizerCamera(
        cameraDistance=15,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[5, 0, 1],
        physicsClientId=client_id
    )
    
    # Disable features that cause "vibration"
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=client_id)  # Disable shadows
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=client_id)

def add_debug_text(client_id, text_lines, position=[0, 0, 3]):
    """Add debug text overlay"""
    text_ids = []
    for i, line in enumerate(text_lines):
        text_id = p.addUserDebugText(
            text=line,
            textPosition=[position[0], position[1], position[2] - i*0.3],
            textColorRGB=[1, 1, 1],
            textSize=1.5,
            lifeTime=0.5,
            physicsClientId=client_id
        )
        text_ids.append(text_id)
    return text_ids

def main():
    print("="*70)
    print("REAL-TIME STEREO VISION - WITH IMAGE WINDOWS")
    print("="*70)
    
    # Start PyBullet with GUI
    client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1./240.)
    p.setRealTimeSimulation(0)
    
    setup_debug_camera(client)
    
    print("\n✓ PyBullet initialized")
    print("\nPyBullet Camera Controls:")
    print("  Hold Ctrl+Alt, then:")
    print("    Left Mouse:   Rotate view")
    print("    Middle Mouse: Pan view")
    print("    Right Mouse:  Zoom")
    print("    Mouse Wheel:  Zoom in/out")
    print("\nKeyboard Controls:")
    print("  1, 2, 3: Change resolution (64, 128, 256)")
    print("  SPACE:   Pause/Resume")
    print("  Q:       Quit")
    
    # Create environment
    print("\nCreating environment...")
    
    plane_shape = p.createCollisionShape(p.GEOM_PLANE)
    plane_visual = p.createVisualShape(p.GEOM_PLANE, rgbaColor=[0.3, 0.5, 0.3, 1])
    plane_id = p.createMultiBody(0, plane_shape, plane_visual)
    
    # Add obstacles
    obstacle_positions = [
        ([3, 0, 0.5], [1, 0, 0, 1]),
        ([5, 2, 0.5], [0, 1, 0, 1]),
        ([7, -2, 0.5], [0, 0, 1, 1]),
        ([9, 1, 1.0], [1, 1, 0, 1]),
        ([11, -1, 0.5], [1, 0, 1, 1]),
    ]
    
    for pos, color in obstacle_positions:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=color)
        p.createMultiBody(1, col, vis, pos)
    
    print(f"✓ Created {len(obstacle_positions)} obstacles")
    
    # Resolution configs
    configs = {
        '64': StereoConfig(img_width=64, img_height=64, baseline=0.08),
        '128': StereoConfig(img_width=128, img_height=128, baseline=0.08),
        '256': StereoConfig(img_width=256, img_height=256, baseline=0.08)
    }
    current_res = '128'
    stereo = StereoVisionSystem(configs[current_res])
    
    print(f"✓ Stereo system: {configs[current_res].img_width}x{configs[current_res].img_height}")
    
    # Create drone visualization
    drone_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.15, rgbaColor=[0, 1, 0, 0.9])
    drone_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.15)
    drone_id = p.createMultiBody(0.1, drone_collision, drone_visual, [0, 0, 2])
    
    # Camera visualization
    cam_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.03, length=0.1, rgbaColor=[1, 0, 0, 0.8])
    left_cam_id = p.createMultiBody(0, -1, cam_vis)
    right_cam_id = p.createMultiBody(0, -1, cam_vis)
    
    print("✓ Drone and cameras created")
    
    # Setup matplotlib windows
    print("\nSetting up matplotlib windows...")
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Stereo Vision - Live Feed', fontsize=14, fontweight='bold')
    
    # Initialize with empty images
    config = configs[current_res]
    empty_img = np.zeros((config.img_height, config.img_width, 3), dtype=np.uint8)
    empty_depth = np.zeros((config.img_height, config.img_width))
    
    im_left = axes[0, 0].imshow(empty_img)
    axes[0, 0].set_title('Left Camera RGB')
    axes[0, 0].axis('off')
    
    im_right = axes[0, 1].imshow(empty_img)
    axes[0, 1].set_title('Right Camera RGB')
    axes[0, 1].axis('off')
    
    im_depth = axes[1, 0].imshow(empty_depth, cmap='plasma', vmin=0, vmax=20)
    axes[1, 0].set_title('Depth Map')
    axes[1, 0].axis('off')
    cbar = plt.colorbar(im_depth, ax=axes[1, 0], label='Distance (m)', fraction=0.046)
    
    # Stats display
    axes[1, 1].axis('off')
    stats_text = axes[1, 1].text(0.1, 0.5, 'Initializing...', fontsize=10, 
                                  family='monospace', verticalalignment='center')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    
    print("✓ Matplotlib windows opened")
    print("\n" + "="*70)
    print("Starting simulation...\n")
    
    # Flight parameters
    drone_pos = np.array([0.0, 0.0, 2.0])
    drone_quat = np.array([0, 0, 0, 1])
    velocity = 1.5
    dt = 1./240.
    
    # State
    paused = False
    flight_time = 0
    frame_count = 0
    frame_times = []
    depth_times = []
    last_stats_print = time.time()
    last_viz_update = time.time()
    
    try:
        while drone_pos[0] < 12.0:
            frame_start = time.time()
            
            # Handle keyboard input
            keys = p.getKeyboardEvents()
            
            if ord('q') in keys or ord('Q') in keys:
                if keys[ord('q')] & p.KEY_WAS_TRIGGERED or keys[ord('Q')] & p.KEY_WAS_TRIGGERED:
                    print("\n\nQuitting...")
                    break
            
            if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
                paused = not paused
                print(f"\n{'⏸️  PAUSED' if paused else '▶️  RESUMED'}")
            
            # Resolution changes
            if ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED:
                current_res = '64'
                stereo = StereoVisionSystem(configs[current_res])
                config = configs[current_res]
                print(f"\n→ Resolution: 64x64")
            elif ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
                current_res = '128'
                stereo = StereoVisionSystem(configs[current_res])
                config = configs[current_res]
                print(f"\n→ Resolution: 128x128")
            elif ord('3') in keys and keys[ord('3')] & p.KEY_WAS_TRIGGERED:
                current_res = '256'
                stereo = StereoVisionSystem(configs[current_res])
                config = configs[current_res]
                print(f"\n→ Resolution: 256x256")
            
            if not paused:
                drone_pos[0] += velocity * dt
                flight_time += dt
            
            # Update drone position
            p.resetBasePositionAndOrientation(drone_id, drone_pos, drone_quat)
            
            # Update camera visualizations
            rot_mat = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)
            half_baseline = config.baseline / 2.0
            
            left_offset = rot_mat @ np.array([0.05, -half_baseline, 0])
            right_offset = rot_mat @ np.array([0.05, half_baseline, 0])
            cam_quat = p.getQuaternionFromEuler([0, np.pi/2, 0])
            
            p.resetBasePositionAndOrientation(left_cam_id, drone_pos + left_offset, cam_quat)
            p.resetBasePositionAndOrientation(right_cam_id, drone_pos + right_offset, cam_quat)
            
            # Capture stereo vision
            depth_start = time.time()
            depth_map, images = stereo.get_depth_map(drone_pos, drone_quat, client, method='pybullet')
            depth_time = (time.time() - depth_start) * 1000
            
            # Analyze depth
            h, w = depth_map.shape
            center_region = depth_map[h//3:2*h//3, w//3:2*w//3]
            min_depth = np.min(center_region)
            avg_depth = np.mean(center_region)
            
            # Determine status
            if min_depth < 2.0:
                status = "🚨 DANGER"
            elif min_depth < 4.0:
                status = "⚠️  CAUTION"
            else:
                status = "✓ CLEAR"
            
            # Update matplotlib windows every 100ms
            if time.time() - last_viz_update > 0.1:
                # Update images
                im_left.set_data(images['left_rgb'][:, :, :3])
                im_right.set_data(images['right_rgb'][:, :, :3])
                im_depth.set_data(depth_map)
                im_depth.set_clim(vmin=0, vmax=min(20, depth_map.max()))
                
                # Update stats
                avg_fps = 1000.0 / np.mean(frame_times) if frame_times else 0
                stats_str = f"""
Frame: {frame_count}
FPS: {avg_fps:.1f}
Resolution: {current_res}x{current_res}

Position:
  X: {drone_pos[0]:.2f}m
  Y: {drone_pos[1]:.2f}m
  Z: {drone_pos[2]:.2f}m

Depth:
  Min: {min_depth:.2f}m
  Avg: {avg_depth:.2f}m
  Status: {status}

Timing:
  Frame: {np.mean(frame_times) if frame_times else 0:.1f}ms
  Depth: {depth_time:.1f}ms

{'⏸️  PAUSED' if paused else '▶️  Running'}
"""
                stats_text.set_text(stats_str)
                
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                last_viz_update = time.time()
            
            # Add debug text overlay in PyBullet
            debug_lines = [
                f"Pos: ({drone_pos[0]:.1f}, {drone_pos[1]:.1f}, {drone_pos[2]:.1f})",
                f"Min Depth: {min_depth:.2f}m | {status}",
                f"FPS: {1000.0/np.mean(frame_times) if frame_times else 0:.1f}",
                "" if not paused else "⏸️  PAUSED"
            ]
            add_debug_text(client, debug_lines, position=[drone_pos[0], drone_pos[1]+2, drone_pos[2]+1])
            
            # Step simulation
            p.stepSimulation()
            
            # Track performance
            frame_time = (time.time() - frame_start) * 1000
            frame_times.append(frame_time)
            depth_times.append(depth_time)
            
            if len(frame_times) > 100:
                frame_times.pop(0)
                depth_times.pop(0)
            
            frame_count += 1
            
            # Print stats periodically
            if time.time() - last_stats_print > 2.0:
                avg_fps = 1000.0 / np.mean(frame_times) if frame_times else 0
                print(f"Frame {frame_count:4d} | FPS: {avg_fps:5.1f} | "
                      f"Depth: {np.mean(depth_times):5.2f}ms | "
                      f"Min: {min_depth:5.2f}m | {status}")
                last_stats_print = time.time()
            
            # Control frame rate
            time.sleep(max(0, dt - (time.time() - frame_start)))
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    # Final stats
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    
    if frame_times:
        print(f"\nFrames rendered: {frame_count}")
        print(f"Flight distance: {drone_pos[0]:.2f}m")
        print(f"Flight time: {flight_time:.2f}s")
        print(f"\nTiming (resolution {current_res}x{current_res}):")
        print(f"  Avg frame time: {np.mean(frame_times):.2f}ms")
        print(f"  Avg FPS: {1000.0/np.mean(frame_times):.1f}")
        print(f"  Depth computation: {np.mean(depth_times):.2f}ms")
    
    print("="*70)
    
    plt.close('all')
    p.disconnect()
    print("\n✓ Closed successfully")

if __name__ == "__main__":
    main()
