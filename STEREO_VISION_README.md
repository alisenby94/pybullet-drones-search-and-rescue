# Stereo Vision System for Drones

Provides depth perception and 3D sensing for drones using a stereo camera setup.

## Features

- **Dual Camera System**: Left and right cameras with configurable baseline (distance between cameras)
- **Depth Estimation**: Two methods available:
  - **PyBullet Depth Buffer** (fast, accurate, GPU-based)
  - **Stereo Matching** (realistic, uses OpenCV matching algorithms)
- **3D Point Cloud Generation**: Convert depth maps to colored 3D point clouds
- **Flexible Configuration**: Customize FOV, image resolution, baseline, depth range

## Quick Start

```python
from stereo_vision import StereoVisionSystem, StereoConfig
import numpy as np
import pybullet as p

# Create stereo system with default config
stereo = StereoVisionSystem()

# Or customize configuration
config = StereoConfig(
    baseline=0.1,      # 10cm between cameras
    fov=60.0,          # 60° field of view
    img_width=256,     # 256x256 image resolution
    img_height=256,
    near_clip=0.1,     # Minimum depth: 0.1m
    far_clip=50.0      # Maximum depth: 50m
)
stereo = StereoVisionSystem(config)

# In your RL environment step/reset:
drone_pos = np.array([x, y, z])  # Drone position
drone_quat = np.array([qx, qy, qz, qw])  # Drone orientation
client_id = self.CLIENT  # PyBullet client ID

# Get depth map
depth_map, images = stereo.get_depth_map(
    drone_pos, drone_quat, client_id, method='pybullet'
)
# depth_map shape: (H, W) with depth in meters
# images dict contains: left_rgb, right_rgb, left_depth, right_depth, left_seg, right_seg

# Generate 3D point cloud (optional)
point_cloud = stereo.get_point_cloud(depth_map, images['left_rgb'])
# point_cloud shape: (N, 6) with [x, y, z, r, g, b]
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `baseline` | float | 0.1 | Distance between left/right cameras (meters) |
| `fov` | float | 60.0 | Field of view (degrees) |
| `img_width` | int | 128 | Image width (pixels) |
| `img_height` | int | 128 | Image height (pixels) |
| `near_clip` | float | 0.1 | Near clipping plane (meters) |
| `far_clip` | float | 50.0 | Far clipping plane (meters) |
| `focal_length` | float | auto | Computed from FOV if not specified |

## Methods

### `capture_stereo_images(drone_pos, drone_quat, client_id)`
Captures images from both left and right cameras.

**Returns:**
```python
{
    'left_rgb': (H, W, 4),      # RGBA image from left camera
    'right_rgb': (H, W, 4),     # RGBA image from right camera
    'left_depth': (H, W),       # Depth buffer from left camera
    'right_depth': (H, W),      # Depth buffer from right camera
    'left_seg': (H, W),         # Segmentation from left camera
    'right_seg': (H, W)         # Segmentation from right camera
}
```

### `get_depth_map(drone_pos, drone_quat, client_id, method='pybullet')`
Computes depth map using specified method.

**Methods:**
- `'pybullet'`: Use PyBullet's depth buffer (fast, accurate)
- `'stereo'`: Use stereo matching on RGB images (realistic, slower)

**Returns:**
- `depth_map`: (H, W) array with depth in meters
- `images`: Dictionary of captured images

### `get_point_cloud(depth_map, rgb_image=None)`
Converts depth map to 3D point cloud.

**Returns:**
- (N, 3) array if no RGB: [x, y, z]
- (N, 6) array with RGB: [x, y, z, r, g, b]

## Integration with RL Environments

### Adding to Observation Space

```python
from stereo_vision import StereoVisionSystem, StereoConfig
from gymnasium import spaces

class MyDroneEnv(BaseRLAviary):
    def __init__(self, use_stereo=True, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize stereo vision
        if use_stereo:
            self.stereo = StereoVisionSystem(StereoConfig(
                img_width=128,
                img_height=128,
                baseline=0.08  # 8cm for small drone
            ))
        else:
            self.stereo = None
    
    def _observationSpace(self):
        # Add depth map to observation space
        obs_dict = {
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(20,)),  # Kinematic state
        }
        
        if self.stereo:
            obs_dict['depth'] = spaces.Box(
                low=0, 
                high=self.stereo.config.far_clip,
                shape=(self.stereo.config.img_height, self.stereo.config.img_width),
                dtype=np.float32
            )
            obs_dict['rgb'] = spaces.Box(
                low=0,
                high=255,
                shape=(self.stereo.config.img_height, self.stereo.config.img_width, 3),
                dtype=np.uint8
            )
        
        return spaces.Dict(obs_dict)
    
    def _computeObs(self):
        obs = {
            'state': self._getDroneStateVector(0)  # Kinematic data
        }
        
        if self.stereo:
            # Get depth map and images
            depth_map, images = self.stereo.get_depth_map(
                self.pos[0], self.quat[0], self.CLIENT
            )
            obs['depth'] = depth_map.astype(np.float32)
            obs['rgb'] = images['left_rgb'][:, :, :3].astype(np.uint8)
        
        return obs
```

### Example: Obstacle Avoidance

```python
def _computeReward(self):
    reward = 0
    
    if self.stereo:
        # Get depth map
        depth_map, _ = self.stereo.get_depth_map(
            self.pos[0], self.quat[0], self.CLIENT
        )
        
        # Penalize if obstacles are too close
        min_depth = np.min(depth_map)
        if min_depth < 1.0:  # Less than 1m
            reward -= 10.0 * (1.0 - min_depth)
        
        # Bonus for maintaining safe distance
        center_depth = depth_map[64, 64]  # Center pixel
        if center_depth > 2.0:
            reward += 0.5
    
    return reward
```

## Performance Considerations

**Image Resolution vs Speed:**
- 64x64: Very fast, low detail (~0.1ms per frame)
- 128x128: Fast, good detail (~0.5ms per frame) **← Recommended**
- 256x256: Slower, high detail (~2ms per frame)
- 512x512: Slow, very high detail (~8ms per frame)

**Method Comparison:**
- **PyBullet depth**: ~0.5ms, GPU accelerated, most accurate
- **Stereo matching**: ~5-20ms, CPU intensive, more realistic

**Recommendation:** Use PyBullet depth for RL training, optionally validate with stereo matching.

## Camera Geometry

```
     drone_forward (+X)
           ↑
           |
    Left   |   Right
    Camera | Camera
       ●---●---●
           ^
       baseline
```

- Cameras are offset ±baseline/2 on the Y-axis (drone's right)
- Both cameras look forward along drone's +X axis
- Up vector is always world Z-up
- Depth is measured along camera's view direction

## Applications

1. **Obstacle Avoidance**: Detect and avoid walls, trees, buildings
2. **Landing Assistance**: Measure ground distance for safe landing
3. **Object Detection**: Identify victims, survivors by depth + color
4. **Mapping**: Build 3D maps of search areas
5. **Navigation**: Visual odometry and SLAM

## Testing

```bash
# Quick test (no GUI)
python test_stereo_vision.py

# Full test with GUI visualization
python stereo_vision.py
```

## Advanced: Point Cloud Processing

```python
# Get colored point cloud
depth_map, images = stereo.get_depth_map(drone_pos, drone_quat, client_id)
point_cloud = stereo.get_point_cloud(depth_map, images['left_rgb'])

# Filter points within range
points_xyz = point_cloud[:, :3]
points_rgb = point_cloud[:, 3:6]

distances = np.linalg.norm(points_xyz, axis=1)
close_points = points_xyz[distances < 5.0]  # Within 5m

# Find obstacles in front
forward_points = points_xyz[points_xyz[:, 0] > 0]  # +X is forward

# Save point cloud (for visualization with Open3D, etc.)
np.save('point_cloud.npy', point_cloud)
```

## Technical Details

**Depth Conversion (PyBullet):**
```python
depth_meters = far * near / (far - (far - near) * depth_buffer)
```

**Depth from Disparity (Stereo):**
```python
depth = (baseline * focal_length) / disparity
```

**3D Reprojection:**
```python
x = (u - cx) * depth / fx
y = (v - cy) * depth / fy
z = depth
```

Where:
- `(u, v)` = pixel coordinates
- `(cx, cy)` = principal point (image center)
- `(fx, fy)` = focal lengths in pixels

## See Also

- `stereo_vision.py` - Main implementation
- `test_stereo_vision.py` - Test script
- PyBullet documentation: https://pybullet.org/
- OpenCV stereo: https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html
