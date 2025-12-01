"""
Stereo Vision System for Drones
Provides depth estimation using dual cameras mounted on the drone
"""
import numpy as np
import pybullet as p
import cv2
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class StereoConfig:
    """Configuration for stereo camera system"""
    baseline: float = 0.1  # Distance between cameras in meters (10cm for small drone)
    fov: float = 60.0  # Field of view in degrees
    img_width: int = 128  # Image width in pixels
    img_height: int = 128  # Image height in pixels
    near_clip: float = 0.1  # Near clipping plane in meters
    far_clip: float = 50.0  # Far clipping plane in meters
    focal_length: Optional[float] = None  # Computed from FOV if None
    
    def __post_init__(self):
        """Compute focal length from FOV if not provided"""
        if self.focal_length is None:
            # focal_length = (image_width / 2) / tan(FOV / 2)
            self.focal_length = (self.img_width / 2.0) / np.tan(np.radians(self.fov / 2.0))


class StereoVisionSystem:
    """
    Stereo vision system for depth estimation on drones
    
    Uses two virtual cameras (left and right) to capture images and compute depth maps.
    The cameras are offset from the drone's center by baseline/2 in the y-axis.
    """
    
    def __init__(self, config: StereoConfig = None):
        """
        Initialize stereo vision system
        
        Args:
            config: StereoConfig object with camera parameters
        """
        self.config = config or StereoConfig()
        
        # Precompute projection matrix (same for both cameras)
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.config.fov,
            aspect=self.config.img_width / self.config.img_height,
            nearVal=self.config.near_clip,
            farVal=self.config.far_clip
        )
        
        # For stereo matching (if using OpenCV)
        self.stereo_matcher = None
        self._init_stereo_matcher()
    
    def _init_stereo_matcher(self):
        """Initialize OpenCV stereo matcher for depth estimation"""
        try:
            # StereoBM (Block Matching) is faster
            self.stereo_matcher = cv2.StereoBM_create(
                numDisparities=16*5,  # Must be divisible by 16
                blockSize=15  # Odd number, typically 5-21
            )
            
            # Optionally use StereoSGBM for better quality (slower)
            # self.stereo_matcher = cv2.StereoSGBM_create(
            #     minDisparity=0,
            #     numDisparities=16*5,
            #     blockSize=5,
            #     P1=8 * 3 * 5**2,
            #     P2=32 * 3 * 5**2,
            #     disp12MaxDiff=1,
            #     uniquenessRatio=10,
            #     speckleWindowSize=100,
            #     speckleRange=32
            # )
        except AttributeError:
            print("[WARNING] OpenCV stereo matchers not available. Using PyBullet depth buffer only.")
            self.stereo_matcher = None
    
    def get_camera_views(
        self,
        drone_position: np.ndarray,
        drone_quaternion: np.ndarray,
        client_id: int
    ) -> Tuple[list, list]:
        """
        Compute view matrices for left and right cameras
        
        Args:
            drone_position: [x, y, z] position of drone
            drone_quaternion: [x, y, z, w] orientation quaternion
            client_id: PyBullet physics client ID
            
        Returns:
            (left_view_matrix, right_view_matrix)
        """
        # Get rotation matrix from quaternion
        rot_mat = np.array(p.getMatrixFromQuaternion(drone_quaternion)).reshape(3, 3)
        
        # Camera offset in drone's local frame (±baseline/2 in y-axis)
        half_baseline = self.config.baseline / 2.0
        
        # Transform offsets to world frame
        left_offset = rot_mat @ np.array([0, -half_baseline, 0])
        right_offset = rot_mat @ np.array([0, half_baseline, 0])
        
        # Camera positions
        left_pos = drone_position + left_offset
        right_pos = drone_position + right_offset
        
        # Target point (1000m forward from drone)
        target = rot_mat @ np.array([1000, 0, 0]) + drone_position
        
        # Camera up vector (always world z-up)
        up_vector = [0, 0, 1]
        
        # Compute view matrices
        left_view = p.computeViewMatrix(
            cameraEyePosition=left_pos.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up_vector,
            physicsClientId=client_id
        )
        
        right_view = p.computeViewMatrix(
            cameraEyePosition=right_pos.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up_vector,
            physicsClientId=client_id
        )
        
        return left_view, right_view
    
    def capture_stereo_images(
        self,
        drone_position: np.ndarray,
        drone_quaternion: np.ndarray,
        client_id: int
    ) -> Dict[str, np.ndarray]:
        """
        Capture images from both cameras
        
        Args:
            drone_position: [x, y, z] position of drone
            drone_quaternion: [x, y, z, w] orientation quaternion
            client_id: PyBullet physics client ID
            
        Returns:
            Dictionary containing:
                'left_rgb': (H, W, 4) RGB image from left camera
                'right_rgb': (H, W, 4) RGB image from right camera
                'left_depth': (H, W) depth buffer from left camera
                'right_depth': (H, W) depth buffer from right camera
                'left_seg': (H, W) segmentation from left camera
                'right_seg': (H, W) segmentation from right camera
        """
        # Get view matrices
        left_view, right_view = self.get_camera_views(
            drone_position, drone_quaternion, client_id
        )
        
        # Capture left camera
        w_l, h_l, rgb_l, dep_l, seg_l = p.getCameraImage(
            width=self.config.img_width,
            height=self.config.img_height,
            viewMatrix=left_view,
            projectionMatrix=self.projection_matrix,
            shadow=1,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=client_id
        )
        
        # Capture right camera
        w_r, h_r, rgb_r, dep_r, seg_r = p.getCameraImage(
            width=self.config.img_width,
            height=self.config.img_height,
            viewMatrix=right_view,
            projectionMatrix=self.projection_matrix,
            shadow=1,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=client_id
        )
        
        # Reshape arrays
        left_rgb = np.reshape(rgb_l, (h_l, w_l, 4))
        right_rgb = np.reshape(rgb_r, (h_r, w_r, 4))
        left_depth = np.reshape(dep_l, (h_l, w_l))
        right_depth = np.reshape(dep_r, (h_r, w_r))
        left_seg = np.reshape(seg_l, (h_l, w_l))
        right_seg = np.reshape(seg_r, (h_r, w_r))
        
        return {
            'left_rgb': left_rgb,
            'right_rgb': right_rgb,
            'left_depth': left_depth,
            'right_depth': right_depth,
            'left_seg': left_seg,
            'right_seg': right_seg
        }
    
    def compute_depth_from_pybullet(
        self,
        depth_buffer: np.ndarray
    ) -> np.ndarray:
        """
        Convert PyBullet depth buffer to real depth in meters
        
        PyBullet returns depth in normalized [0, 1] range that needs conversion.
        
        Args:
            depth_buffer: (H, W) normalized depth buffer from PyBullet
            
        Returns:
            (H, W) depth map in meters
        """
        # PyBullet depth buffer formula
        # depth_meters = far * near / (far - (far - near) * depth_buffer)
        near = self.config.near_clip
        far = self.config.far_clip
        
        depth_meters = far * near / (far - (far - near) * depth_buffer)
        
        return depth_meters
    
    def compute_disparity_map(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Compute disparity map from stereo images using OpenCV
        
        Args:
            left_rgb: (H, W, 4) left camera RGB image
            right_rgb: (H, W, 4) right camera RGB image
            
        Returns:
            (H, W) disparity map or None if stereo matcher not available
        """
        if self.stereo_matcher is None:
            return None
        
        # Convert to grayscale (stereo matchers work on grayscale)
        left_gray = cv2.cvtColor(left_rgb[:, :, :3], cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_rgb[:, :, :3], cv2.COLOR_RGB2GRAY)
        
        # Compute disparity
        disparity = self.stereo_matcher.compute(left_gray, right_gray)
        
        # Convert to float and normalize
        disparity = disparity.astype(np.float32) / 16.0  # StereoBM outputs 16x scaled values
        
        return disparity
    
    def disparity_to_depth(
        self,
        disparity: np.ndarray
    ) -> np.ndarray:
        """
        Convert disparity map to depth map
        
        Depth = (baseline * focal_length) / disparity
        
        Args:
            disparity: (H, W) disparity map in pixels
            
        Returns:
            (H, W) depth map in meters
        """
        # Avoid division by zero
        disparity_safe = np.where(disparity > 0, disparity, 0.1)
        
        # depth = baseline * focal_length / disparity
        depth = (self.config.baseline * self.config.focal_length) / disparity_safe
        
        # Clip to valid range
        depth = np.clip(depth, self.config.near_clip, self.config.far_clip)
        
        return depth
    
    def get_depth_map(
        self,
        drone_position: np.ndarray,
        drone_quaternion: np.ndarray,
        client_id: int,
        method: str = 'pybullet'
    ) -> Tuple[np.ndarray, Dict]:
        """
        Get depth map using specified method
        
        Args:
            drone_position: [x, y, z] position of drone
            drone_quaternion: [x, y, z, w] orientation quaternion
            client_id: PyBullet physics client ID
            method: 'pybullet' (use PyBullet depth buffer) or 
                   'stereo' (use stereo matching on RGB images)
            
        Returns:
            (depth_map, images_dict)
            depth_map: (H, W) depth in meters
            images_dict: Dictionary with captured images
        """
        # Capture stereo images
        images = self.capture_stereo_images(drone_position, drone_quaternion, client_id)
        
        if method == 'pybullet':
            # Use PyBullet's depth buffer (faster, more accurate)
            depth_map = self.compute_depth_from_pybullet(images['left_depth'])
        
        elif method == 'stereo':
            # Use stereo matching on RGB images (more realistic simulation)
            disparity = self.compute_disparity_map(images['left_rgb'], images['right_rgb'])
            if disparity is not None:
                depth_map = self.disparity_to_depth(disparity)
            else:
                # Fallback to PyBullet depth
                print("[WARNING] Stereo matching not available, using PyBullet depth")
                depth_map = self.compute_depth_from_pybullet(images['left_depth'])
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pybullet' or 'stereo'")
        
        return depth_map, images
    
    def get_point_cloud(
        self,
        depth_map: np.ndarray,
        rgb_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert depth map to 3D point cloud
        
        Args:
            depth_map: (H, W) depth in meters
            rgb_image: Optional (H, W, 3/4) RGB image for colored point cloud
            
        Returns:
            (N, 3) or (N, 6) point cloud array
            If rgb_image provided: (N, 6) with [x, y, z, r, g, b]
            Otherwise: (N, 3) with [x, y, z]
        """
        h, w = depth_map.shape
        
        # Create pixel coordinate grids
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Camera intrinsics
        cx = w / 2.0  # Principal point x
        cy = h / 2.0  # Principal point y
        fx = self.config.focal_length  # Focal length x
        fy = self.config.focal_length  # Focal length y
        
        # Convert to 3D coordinates (camera frame)
        # x = (u - cx) * depth / fx
        # y = (v - cy) * depth / fy
        # z = depth
        x = (u - cx) * depth_map / fx
        y = (v - cy) * depth_map / fy
        z = depth_map
        
        # Stack coordinates
        points = np.stack([x, y, z], axis=-1)  # (H, W, 3)
        points = points.reshape(-1, 3)  # (H*W, 3)
        
        # Filter invalid points (too close or too far)
        valid_mask = (points[:, 2] > self.config.near_clip) & (points[:, 2] < self.config.far_clip)
        points = points[valid_mask]
        
        # Add color if provided
        if rgb_image is not None:
            colors = rgb_image[:, :, :3].reshape(-1, 3)[valid_mask]
            colors = colors / 255.0  # Normalize to [0, 1]
            points = np.concatenate([points, colors], axis=1)  # (N, 6)
        
        return points


if __name__ == "__main__":
    """Test stereo vision system"""
    import pybullet as p
    import time
    
    print("="*70)
    print("STEREO VISION SYSTEM TEST")
    print("="*70)
    
    # Start PyBullet
    client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)
    
    # Add ground plane
    plane_shape = p.createCollisionShape(p.GEOM_PLANE)
    plane_visual = p.createVisualShape(p.GEOM_PLANE, rgbaColor=[0.5, 0.5, 0.5, 1])
    plane_id = p.createMultiBody(0, plane_shape, plane_visual)
    
    # Add some box objects for depth testing
    box_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.25])
    box_visual1 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.25], rgbaColor=[1, 0, 0, 1])
    box_visual2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.25], rgbaColor=[0, 1, 0, 1])
    box_visual3 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.25], rgbaColor=[0, 0, 1, 1])
    
    cube1 = p.createMultiBody(1, box_shape, box_visual1, [2, 0, 0.5])
    cube2 = p.createMultiBody(1, box_shape, box_visual2, [3, 1, 0.5])
    cube3 = p.createMultiBody(1, box_shape, box_visual3, [4, -1, 0.5])
    
    # Create stereo vision system
    config = StereoConfig(
        baseline=0.1,  # 10cm between cameras
        img_width=256,
        img_height=256,
        fov=60.0
    )
    stereo = StereoVisionSystem(config)
    
    print(f"\nStereo Configuration:")
    print(f"  Baseline: {config.baseline}m")
    print(f"  FOV: {config.fov}°")
    print(f"  Image size: {config.img_width}x{config.img_height}")
    print(f"  Focal length: {config.focal_length:.2f} pixels")
    print(f"  Depth range: {config.near_clip}m to {config.far_clip}m")
    
    # Simulate drone at position
    drone_pos = np.array([0.0, 0.0, 2.0])
    drone_quat = np.array([0, 0, 0, 1])  # No rotation
    
    print(f"\nDrone position: {drone_pos}")
    print("\nCapturing stereo images...")
    
    # Method 1: PyBullet depth buffer
    depth_pybullet, images = stereo.get_depth_map(
        drone_pos, drone_quat, client, method='pybullet'
    )
    
    print(f"\n✓ Captured images:")
    print(f"  Left RGB: {images['left_rgb'].shape}")
    print(f"  Right RGB: {images['right_rgb'].shape}")
    print(f"  Left Depth: {images['left_depth'].shape}")
    print(f"  Right Depth: {images['right_depth'].shape}")
    
    print(f"\n✓ Depth map (PyBullet method):")
    print(f"  Shape: {depth_pybullet.shape}")
    print(f"  Min depth: {np.min(depth_pybullet):.2f}m")
    print(f"  Max depth: {np.max(depth_pybullet):.2f}m")
    print(f"  Mean depth: {np.mean(depth_pybullet):.2f}m")
    
    # Method 2: Stereo matching (if OpenCV available)
    if stereo.stereo_matcher is not None:
        print("\nComputing disparity with stereo matching...")
        depth_stereo, _ = stereo.get_depth_map(
            drone_pos, drone_quat, client, method='stereo'
        )
        print(f"\n✓ Depth map (Stereo matching method):")
        print(f"  Shape: {depth_stereo.shape}")
        print(f"  Min depth: {np.min(depth_stereo):.2f}m")
        print(f"  Max depth: {np.max(depth_stereo):.2f}m")
        print(f"  Mean depth: {np.mean(depth_stereo):.2f}m")
    
    # Generate point cloud
    print("\nGenerating point cloud...")
    point_cloud = stereo.get_point_cloud(depth_pybullet, images['left_rgb'])
    print(f"\n✓ Point cloud:")
    print(f"  Points: {point_cloud.shape[0]}")
    print(f"  Dimensions: {point_cloud.shape[1]} (XYZ + RGB)" if point_cloud.shape[1] == 6 else f"  Dimensions: {point_cloud.shape[1]} (XYZ only)")
    
    print("\n" + "="*70)
    print("Test complete! Close the PyBullet window to exit.")
    print("="*70)
    
    # Keep window open
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        p.disconnect()
