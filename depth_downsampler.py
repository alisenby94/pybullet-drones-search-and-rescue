"""
Depth Map Downsampler with Min Pooling

Downsamples high-resolution depth maps to 64x32 for efficient RL processing.
Uses MIN pooling to preserve closest obstacles (most critical for navigation).
"""

import numpy as np
import cv2
from typing import Tuple


class DepthDownsampler:
    """
    Downsamples depth maps with min pooling to preserve closest obstacles.
    
    Input: High-res depth map (e.g., 128x128, 256x256)
    Output: 64x32 depth map (wide format for stereo vision)
    
    Min pooling ensures that the closest obstacle in each region is preserved,
    which is critical for collision avoidance in navigation.
    """
    
    def __init__(self, target_width: int = 64, target_height: int = 32):
        """
        Initialize downsampler.
        
        Args:
            target_width: Output width (default 64 for wide stereo)
            target_height: Output height (default 32 for wide stereo)
        """
        self.target_width = target_width
        self.target_height = target_height
        
        print(f"Depth Downsampler: {target_width}x{target_height} output")
        print(f"  Using MIN pooling (preserves closest obstacles)")
    
    def downsample(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Downsample depth map using min pooling.
        
        Args:
            depth_map: (H, W) high-resolution depth map in meters
            
        Returns:
            (target_height, target_width) downsampled depth map
        """
        h, w = depth_map.shape
        
        # Calculate pooling window size
        pool_h = h // self.target_height
        pool_w = w // self.target_width
        
        # If dimensions don't divide evenly, resize first
        if h % self.target_height != 0 or w % self.target_width != 0:
            new_h = self.target_height * pool_h
            new_w = self.target_width * pool_w
            depth_map = cv2.resize(depth_map, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Apply MIN pooling (preserve closest obstacles)
        downsampled = np.zeros((self.target_height, self.target_width), dtype=np.float32)
        
        for i in range(self.target_height):
            for j in range(self.target_width):
                # Extract window
                window = depth_map[
                    i*pool_h:(i+1)*pool_h,
                    j*pool_w:(j+1)*pool_w
                ]
                
                # Take minimum (closest obstacle)
                # Filter out invalid values (0 or very large)
                valid_depths = window[(window > 0.01) & (window < 100.0)]
                
                if len(valid_depths) > 0:
                    downsampled[i, j] = np.min(valid_depths)
                else:
                    downsampled[i, j] = 50.0  # Far clip default
        
        return downsampled
    
    def downsample_with_features(self, depth_map: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Downsample with additional features extracted during pooling.
        
        Returns both the downsampled map and useful statistics that could
        help the transformer make better decisions.
        
        Args:
            depth_map: (H, W) high-resolution depth map in meters
            
        Returns:
            downsampled_map: (target_height, target_width) depth map
            features: Dict with additional extracted features:
                - 'variance_map': (H, W) variance in each pooling window
                - 'min_distances': (H, W) minimum distances (same as downsampled)
                - 'mean_distances': (H, W) mean distances in each window
                - 'obstacle_density': (H, W) ratio of close obstacles (<1m) in each window
        """
        h, w = depth_map.shape
        
        # Calculate pooling window size
        pool_h = h // self.target_height
        pool_w = w // self.target_width
        
        # Resize if needed
        if h % self.target_height != 0 or w % self.target_width != 0:
            new_h = self.target_height * pool_h
            new_w = self.target_width * pool_w
            depth_map = cv2.resize(depth_map, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Initialize output arrays
        downsampled = np.zeros((self.target_height, self.target_width), dtype=np.float32)
        variance_map = np.zeros((self.target_height, self.target_width), dtype=np.float32)
        mean_map = np.zeros((self.target_height, self.target_width), dtype=np.float32)
        density_map = np.zeros((self.target_height, self.target_width), dtype=np.float32)
        
        close_threshold = 1.0  # Obstacles within 1m are "close"
        
        for i in range(self.target_height):
            for j in range(self.target_width):
                # Extract window
                window = depth_map[
                    i*pool_h:(i+1)*pool_h,
                    j*pool_w:(j+1)*pool_w
                ]
                
                # Filter valid depths
                valid_depths = window[(window > 0.01) & (window < 100.0)]
                
                if len(valid_depths) > 0:
                    # Min distance (closest obstacle)
                    downsampled[i, j] = np.min(valid_depths)
                    
                    # Mean distance
                    mean_map[i, j] = np.mean(valid_depths)
                    
                    # Variance (indicates depth complexity/clutter)
                    variance_map[i, j] = np.var(valid_depths)
                    
                    # Obstacle density (% of pixels within close threshold)
                    close_count = np.sum(valid_depths < close_threshold)
                    density_map[i, j] = close_count / len(valid_depths)
                else:
                    downsampled[i, j] = 50.0
                    mean_map[i, j] = 50.0
                    variance_map[i, j] = 0.0
                    density_map[i, j] = 0.0
        
        features = {
            'variance_map': variance_map,
            'min_distances': downsampled,
            'mean_distances': mean_map,
            'obstacle_density': density_map
        }
        
        return downsampled, features
    
    def visualize_comparison(self, original: np.ndarray, downsampled: np.ndarray):
        """
        Create visualization comparing original and downsampled depth maps.
        
        Args:
            original: (H, W) original depth map
            downsampled: (target_height, target_width) downsampled map
            
        Returns:
            (H, W*2, 3) comparison image for display
        """
        # Normalize for visualization
        orig_vis = np.clip(original / 10.0, 0, 1)  # Clip to 10m for vis
        down_vis = np.clip(downsampled / 10.0, 0, 1)
        
        # Apply colormap
        orig_color = cv2.applyColorMap((orig_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Resize downsampled to match original height for comparison
        down_resized = cv2.resize(down_vis, (original.shape[1], original.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
        down_color = cv2.applyColorMap((down_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Concatenate side by side
        comparison = np.hstack([orig_color, down_color])
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, f"Downsampled {self.target_width}x{self.target_height}", 
                   (original.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return comparison


def test_downsampler():
    """Test the downsampler with synthetic data."""
    print("\n" + "="*70)
    print("DEPTH DOWNSAMPLER TEST")
    print("="*70 + "\n")
    
    # Create synthetic depth map with obstacles
    h, w = 256, 256
    depth_map = np.ones((h, w), dtype=np.float32) * 10.0  # Background at 10m
    
    # Add some close obstacles
    depth_map[50:80, 100:130] = 0.5  # Close obstacle at 0.5m
    depth_map[150:200, 50:100] = 2.0  # Medium obstacle at 2m
    depth_map[100:120, 200:230] = 0.3  # Very close obstacle at 0.3m
    
    # Add some noise
    depth_map += np.random.randn(h, w) * 0.1
    depth_map = np.clip(depth_map, 0.1, 50.0)
    
    print(f"Original depth map: {h}x{w}")
    print(f"  Min: {np.min(depth_map):.2f}m")
    print(f"  Max: {np.max(depth_map):.2f}m")
    print(f"  Mean: {np.mean(depth_map):.2f}m")
    
    # Create downsampler
    downsampler = DepthDownsampler(target_width=64, target_height=32)
    
    # Test basic downsampling
    print("\n" + "-"*70)
    print("Basic MIN Pooling:")
    print("-"*70)
    downsampled = downsampler.downsample(depth_map)
    
    print(f"\nDownsampled depth map: {downsampled.shape[1]}x{downsampled.shape[0]}")
    print(f"  Min: {np.min(downsampled):.2f}m (preserved closest obstacle!)")
    print(f"  Max: {np.max(downsampled):.2f}m")
    print(f"  Mean: {np.mean(downsampled):.2f}m")
    print(f"  Compression ratio: {(h*w) / (downsampled.shape[0]*downsampled.shape[1]):.1f}x")
    
    # Test with features
    print("\n" + "-"*70)
    print("Downsampling with Feature Extraction:")
    print("-"*70)
    downsampled_feat, features = downsampler.downsample_with_features(depth_map)
    
    print(f"\nExtracted features:")
    print(f"  Variance map: {features['variance_map'].shape}")
    print(f"    Mean variance: {np.mean(features['variance_map']):.3f}")
    print(f"  Obstacle density map: {features['obstacle_density'].shape}")
    print(f"    Max density: {np.max(features['obstacle_density']):.2%}")
    print(f"  Mean distance map: {features['mean_distances'].shape}")
    
    # Compare closest obstacles
    print("\n" + "-"*70)
    print("Obstacle Preservation Check:")
    print("-"*70)
    print(f"  Closest obstacle in original: {np.min(depth_map):.2f}m")
    print(f"  Closest obstacle in downsampled: {np.min(downsampled):.2f}m")
    print(f"  ✓ Preservation error: {abs(np.min(depth_map) - np.min(downsampled)):.3f}m")
    
    # Memory savings
    original_bytes = depth_map.nbytes
    downsampled_bytes = downsampled.nbytes
    print(f"\nMemory usage:")
    print(f"  Original: {original_bytes:,} bytes")
    print(f"  Downsampled: {downsampled_bytes:,} bytes")
    print(f"  Savings: {(1 - downsampled_bytes/original_bytes)*100:.1f}%")
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70 + "\n")
    
    return depth_map, downsampled, features


if __name__ == "__main__":
    test_downsampler()
