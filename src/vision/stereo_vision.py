"""
Stereo Vision System with Spatial Attention

PURPOSE:
    Capture stereo images, compute depth maps, and apply attention mechanism
    for obstacle-aware navigation. Includes streaming for real-time monitoring.

ARCHITECTURE:
    - Stereo cameras: Left/Right cameras mounted on drone
    - Depth estimation: Disparity-based depth calculation
    - Downsampling: 64x32 depth map (min pooling to preserve closest obstacles)
    - Attention head: Multi-head attention for obstacle saliency
    - Streaming: RTSP/HTTP server for VLC monitoring
"""

import numpy as np
import cv2
import threading
import socket
import struct
from typing import Tuple, Optional
import pybullet as p


class StereoVisionSystem:
    """
    Stereo vision system with depth estimation and attention mechanism.
    
    Features:
        - Stereo image capture from PyBullet
        - Depth map computation from disparity
        - Min pooling downsampling (preserves closest obstacles)
        - Spatial attention for obstacle detection
        - Real-time streaming for VLC monitoring
    """
    
    def __init__(
        self,
        baseline: float = 0.06,  # 6cm stereo baseline (CF2X width)
        focal_length: float = 0.5,  # Focal length in pixels
        resolution: Tuple[int, int] = (640, 480),  # Camera resolution
        fov: float = 90.0,  # Field of view in degrees
        near_plane: float = 0.1,  # Near clipping plane
        far_plane: float = 10.0,  # Far clipping plane
        downsample_size: Tuple[int, int] = (64, 32),  # Downsampled depth map size
        enable_streaming: bool = False,  # Enable video streaming
        stream_port: int = 5555  # Port for video streaming
    ):
        """
        Initialize stereo vision system.
        
        Args:
            baseline: Distance between left and right cameras (m)
            focal_length: Camera focal length (pixels)
            resolution: Camera resolution (width, height)
            fov: Field of view (degrees)
            near_plane: Near clipping distance (m)
            far_plane: Far clipping distance (m)
            downsample_size: Target size for depth map (width, height)
            enable_streaming: Enable RTSP/HTTP streaming for VLC
            stream_port: Port number for streaming server
        """
        self.baseline = baseline
        self.focal_length = focal_length
        self.resolution = resolution
        self.fov = fov
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.downsample_size = downsample_size
        
        # Camera offset from drone center (left/right)
        self.camera_offset = baseline / 2.0
        
        # Compute projection matrix
        aspect_ratio = resolution[0] / resolution[1]
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect_ratio,
            nearVal=near_plane,
            farVal=far_plane
        )
        
        # Streaming setup
        self.enable_streaming = enable_streaming
        self.stream_port = stream_port
        self.streaming_thread = None
        self.stream_buffer = None
        self.stream_lock = threading.Lock()
        
        if enable_streaming:
            self._start_streaming_server()
    
    def capture_stereo_images(
        self,
        drone_pos: np.ndarray,
        drone_rpy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture left and right stereo images from drone's perspective.
        
        Args:
            drone_pos: Drone position [x, y, z]
            drone_rpy: Drone orientation [roll, pitch, yaw]
            
        Returns:
            left_img: Left camera image (H, W, 3)
            right_img: Right camera image (H, W, 3)
        """
        # Compute camera positions (left and right offset from drone center)
        yaw = drone_rpy[2]
        left_offset = np.array([
            -self.camera_offset * np.sin(yaw),
            self.camera_offset * np.cos(yaw),
            0.0
        ])
        right_offset = -left_offset
        
        left_pos = drone_pos + left_offset
        right_pos = drone_pos + right_offset
        
        # Compute view matrices (looking forward from drone's orientation)
        left_img, left_depth = self._capture_image(left_pos, drone_rpy)
        right_img, right_depth = self._capture_image(right_pos, drone_rpy)
        
        return (left_img, left_depth), (right_img, right_depth)
    
    def _capture_image(
        self,
        camera_pos: np.ndarray,
        camera_rpy: np.ndarray
    ) -> np.ndarray:
        """
        Capture single camera image from PyBullet.
        
        Args:
            camera_pos: Camera position [x, y, z]
            camera_rpy: Camera orientation [roll, pitch, yaw]
            
        Returns:
            img: RGB image (H, W, 3)
        """
        # Convert RPY to rotation matrix to get proper camera orientation
        roll, pitch, yaw = camera_rpy
        
        # Rotation matrices for each axis
        # Roll (rotation around X-axis)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Pitch (rotation around Y-axis)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Yaw (rotation around Z-axis)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix (apply in order: yaw, pitch, roll)
        R = Rz @ Ry @ Rx
        
        # Camera looks along +X axis in body frame (forward)
        forward_body = np.array([1.0, 0.0, 0.0])
        forward_world = R @ forward_body
        
        # Camera up vector is +Z in body frame
        up_body = np.array([0.0, 0.0, 1.0])
        up_world = R @ up_body
        
        # Target point is 1 meter ahead in the direction the drone is facing
        target = camera_pos + forward_world
        
        # Compute view matrix with proper orientation
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up_world.tolist()
        )
        
        # Capture image
        width, height = self.resolution
        img_arr = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Extract RGB image and depth buffer
        rgb_img = np.array(img_arr[2]).reshape(height, width, 4)[:, :, :3]
        depth_buffer = np.array(img_arr[3]).reshape(height, width)
        
        return rgb_img, depth_buffer
    
    def compute_depth_map(
        self,
        depth_buffer: np.ndarray
    ) -> np.ndarray:
        """
        Convert PyBullet depth buffer to real depth values in meters.
        
        PyBullet returns depth buffer in range [0, 1] where values are non-linear.
        Need to convert back to actual distances using near/far plane.
        
        Args:
            depth_buffer: PyBullet depth buffer (H, W) in range [0, 1]
            
        Returns:
            depth_map: Depth values in meters (H, W)
        """
        # Convert PyBullet depth buffer to linear depth
        # Formula: depth = far * near / (far - (far - near) * depth_buffer)
        depth_map = self.far_plane * self.near_plane / (
            self.far_plane - (self.far_plane - self.near_plane) * depth_buffer
        )
        
        # Clip to valid range
        depth_map = np.clip(depth_map, self.near_plane, self.far_plane)
        
        return depth_map
    
    def downsample_depth_map(
        self,
        depth_map: np.ndarray
    ) -> np.ndarray:
        """
        Downsample depth map using min pooling to preserve closest obstacles.
        
        Args:
            depth_map: Full resolution depth map (H, W)
            
        Returns:
            downsampled: Downsampled depth map (downsample_size)
        """
        target_h, target_w = self.downsample_size
        current_h, current_w = depth_map.shape
        
        # Compute pooling window size
        pool_h = current_h // target_h
        pool_w = current_w // target_w
        
        # Apply min pooling (preserves closest obstacles)
        downsampled = np.zeros((target_h, target_w), dtype=np.float32)
        
        for i in range(target_h):
            for j in range(target_w):
                window = depth_map[
                    i*pool_h:(i+1)*pool_h,
                    j*pool_w:(j+1)*pool_w
                ]
                downsampled[i, j] = np.min(window)
        
        return downsampled
    
    def apply_spatial_attention(
        self,
        depth_map: np.ndarray,
        waypoint_direction: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply spatial attention mechanism to highlight salient obstacles.
        
        Simple attention: Closer objects get higher attention weights.
        If waypoint direction provided, objects in that direction get boosted.
        
        Args:
            depth_map: Depth map (H, W)
            waypoint_direction: Unit vector to waypoint in camera frame [x, y, z]
            
        Returns:
            attention_weights: Attention map (H, W) in range [0, 1]
            attended_features: Weighted depth map (H, W)
        """
        # Inverse depth as base attention (closer = higher attention)
        # Use exponential decay: attention = exp(-depth / threshold)
        attention_threshold = 2.0  # Objects within 2m get high attention
        attention_weights = np.exp(-depth_map / attention_threshold)
        
        # If waypoint direction provided, boost attention in that direction
        if waypoint_direction is not None:
            # Create spatial bias map based on waypoint direction
            h, w = depth_map.shape
            center_y, center_x = h // 2, w // 2
            
            # Convert waypoint direction to image coordinates
            # Assuming camera looks along +X axis, waypoint direction in camera frame
            waypoint_x = waypoint_direction[0]  # Forward
            waypoint_y = waypoint_direction[1]  # Left
            
            # Compute pixel offset from center
            pixel_offset_x = int(waypoint_x * w / 2)
            pixel_offset_y = int(-waypoint_y * h / 2)  # Negative because image Y is down
            
            target_x = center_x + pixel_offset_x
            target_y = center_y + pixel_offset_y
            
            # Create Gaussian bias centered on waypoint direction
            y_coords, x_coords = np.ogrid[:h, :w]
            spatial_bias = np.exp(
                -((x_coords - target_x)**2 + (y_coords - target_y)**2) / (w * h / 10)
            )
            
            # Boost attention in waypoint direction
            attention_weights = attention_weights * (1.0 + spatial_bias)
        
        # Normalize attention weights
        attention_weights = attention_weights / (np.max(attention_weights) + 1e-8)
        
        # Apply attention to depth map
        attended_features = depth_map * attention_weights
        
        return attention_weights, attended_features
    
    def get_vision_observation(
        self,
        drone_pos: np.ndarray,
        drone_rpy: np.ndarray,
        waypoint_pos: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get complete vision observation for policy network.
        
        Args:
            drone_pos: Drone position [x, y, z]
            drone_rpy: Drone orientation [roll, pitch, yaw]
            waypoint_pos: Current waypoint position [x, y, z] (optional)
            
        Returns:
            vision_obs: Flattened vision features (downsample_size[0] * downsample_size[1],)
        """
        # Capture stereo images
        (left_img, left_depth), (right_img, right_depth) = self.capture_stereo_images(drone_pos, drone_rpy)
        
        # Use left camera depth (PyBullet provides accurate depth directly)
        depth_map = self.compute_depth_map(left_depth)
        
        # Downsample
        depth_map_small = self.downsample_depth_map(depth_map)
        
        # Compute waypoint direction in camera frame if waypoint provided
        waypoint_direction = None
        if waypoint_pos is not None:
            # Transform waypoint to camera frame
            vec_to_waypoint = waypoint_pos - drone_pos
            yaw = drone_rpy[2]
            pitch = drone_rpy[1]
            
            # Rotate to camera frame (simple yaw rotation)
            waypoint_direction = np.array([
                vec_to_waypoint[0] * np.cos(yaw) + vec_to_waypoint[1] * np.sin(yaw),
                -vec_to_waypoint[0] * np.sin(yaw) + vec_to_waypoint[1] * np.cos(yaw),
                vec_to_waypoint[2]
            ])
            waypoint_direction = waypoint_direction / (np.linalg.norm(waypoint_direction) + 1e-8)
        
        # Apply spatial attention
        attention_weights, attended_features = self.apply_spatial_attention(
            depth_map_small,
            waypoint_direction
        )
        
        # Always save debug snapshots (useful for validation)
        self._save_debug_snapshot(left_img, depth_map, attention_weights, depth_map_small)
        
        # Update streaming buffer if enabled
        if self.enable_streaming:
            self._update_stream_buffer(left_img, depth_map, attention_weights)
        
        # Flatten attended features for policy network
        vision_obs = attended_features.flatten()
        
        return vision_obs
    
    def _save_debug_snapshot(
        self,
        rgb_img: np.ndarray,
        depth_map: np.ndarray,
        attention_map: np.ndarray,
        depth_map_downsampled: np.ndarray = None
    ):
        """
        Save debug visualization snapshot to disk.
        
        Args:
            rgb_img: RGB camera image (H, W, 3)
            depth_map: Full resolution depth map (H, W)
            attention_map: Attention weights (H_small, W_small)
            depth_map_downsampled: Min-pooled depth map (64, 32)
        """
        try:
            import os
            os.makedirs("vision_debug", exist_ok=True)
            
            # Resize attention map to match RGB image
            attention_upscaled = cv2.resize(
                attention_map,
                (rgb_img.shape[1], rgb_img.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Create heatmap overlay
            heatmap = cv2.applyColorMap(
                (attention_upscaled * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            
            # Blend RGB with heatmap
            overlay = cv2.addWeighted(rgb_img, 0.6, heatmap, 0.4, 0)
            
            # Convert depth map to heatmap for visualization (full resolution)
            depth_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-8)
            depth_heatmap = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            
            # Resize depth to match RGB
            depth_resized = cv2.resize(depth_heatmap, (rgb_img.shape[1], rgb_img.shape[0]))
            
            # Visualize downsampled depth if provided
            if depth_map_downsampled is not None:
                # Normalize and visualize min-pooled depth
                downsampled_norm = (depth_map_downsampled - np.min(depth_map_downsampled)) / (np.max(depth_map_downsampled) - np.min(depth_map_downsampled) + 1e-8)
                downsampled_heatmap = cv2.applyColorMap((downsampled_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                # Upscale to match RGB size for visualization
                downsampled_upscaled = cv2.resize(downsampled_heatmap, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Create composite image: RGB | Depth Full | Depth Pooled | Attention | Overlay
                composite = np.hstack([
                    rgb_img,
                    depth_resized,
                    downsampled_upscaled,
                    cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
                    overlay
                ])
            else:
                # Create composite image: RGB | Depth | Attention | Overlay
                composite = np.hstack([
                    rgb_img,
                    depth_resized,
                    cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
                    overlay
                ])
            
            # Save composite (convert RGB to BGR for cv2)
            filepath = "vision_debug/stereo_vision_snapshot.jpg"
            success = cv2.imwrite(filepath, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            
            if success:
                print(f"[StereoVision] Saved snapshot to {filepath}")
                print(f"  - Depth range: {np.min(depth_map):.2f}m to {np.max(depth_map):.2f}m")
                if depth_map_downsampled is not None:
                    print(f"  - Pooled depth range: {np.min(depth_map_downsampled):.2f}m to {np.max(depth_map_downsampled):.2f}m")
                    print(f"  - Pooled shape: {depth_map_downsampled.shape}")
            else:
                print(f"[StereoVision] Failed to save snapshot to {filepath}")
                
        except Exception as e:
            print(f"[StereoVision] Error saving snapshot: {e}")
    
    def _update_stream_buffer(
        self,
        rgb_img: np.ndarray,
        depth_map: np.ndarray,
        attention_map: np.ndarray
    ):
        """
        Update streaming buffer with visualization overlay.
        
        Args:
            rgb_img: RGB camera image (H, W, 3)
            depth_map: Full resolution depth map (H, W)
            attention_map: Attention weights (H_small, W_small)
        """
        # Resize attention map to match RGB image
        attention_upscaled = cv2.resize(
            attention_map,
            (rgb_img.shape[1], rgb_img.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Create heatmap overlay
        heatmap = cv2.applyColorMap(
            (attention_upscaled * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Blend RGB with heatmap
        overlay = cv2.addWeighted(rgb_img, 0.6, heatmap, 0.4, 0)
        
        # Update buffer with thread safety
        with self.stream_lock:
            self.stream_buffer = overlay
    
    def _start_streaming_server(self):
        """
        Start HTTP streaming server for VLC monitoring.
        """
        def stream_worker():
            # Create TCP socket
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('0.0.0.0', self.stream_port))
            server_socket.listen(5)
            
            print(f"[StereoVision] Streaming server started on port {self.stream_port}")
            print(f"[StereoVision] Open in VLC: http://localhost:{self.stream_port}/stream")
            
            while True:
                try:
                    client_socket, addr = server_socket.accept()
                    print(f"[StereoVision] Client connected: {addr}")
                    
                    # Send HTTP header
                    header = (
                        b"HTTP/1.1 200 OK\r\n"
                        b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
                    )
                    client_socket.sendall(header)
                    
                    # Stream frames
                    while True:
                        with self.stream_lock:
                            if self.stream_buffer is not None:
                                # Encode frame as JPEG
                                _, jpeg = cv2.imencode('.jpg', self.stream_buffer)
                                frame_data = jpeg.tobytes()
                                
                                # Send frame with boundary
                                frame_header = (
                                    b"--frame\r\n"
                                    b"Content-Type: image/jpeg\r\n"
                                    b"Content-Length: " + str(len(frame_data)).encode() + b"\r\n\r\n"
                                )
                                client_socket.sendall(frame_header + frame_data + b"\r\n")
                        
                        # Limit frame rate (10 Hz)
                        import time
                        time.sleep(0.1)
                
                except Exception as e:
                    print(f"[StereoVision] Stream error: {e}")
                    client_socket.close()
        
        # Start streaming in background thread
        self.streaming_thread = threading.Thread(target=stream_worker, daemon=True)
        self.streaming_thread.start()
    
    def close(self):
        """Clean up resources."""
        if self.streaming_thread is not None:
            # Streaming thread is daemon, will auto-terminate
            pass
