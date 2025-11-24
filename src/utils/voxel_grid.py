"""
Voxel Grid for Occupancy Mapping

PURPOSE:
    Discretize 3D space into voxel cubes for obstacle representation.
    Build occupancy grid from PyBullet collision objects.
    
FEATURES:
    - 0.5m voxel resolution (configurable)
    - Fast occupancy queries
    - Collision checking for waypoint generation
    - Grid visualization support
"""

import numpy as np
import pybullet as p
from typing import Tuple, List, Optional, Set


class VoxelGrid:
    """
    3D voxel grid for spatial occupancy mapping.
    
    Discretizes continuous 3D space into uniform cubic voxels.
    Tracks which voxels are occupied by obstacles.
    """
    
    def __init__(
        self,
        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        voxel_size: float = 0.5,
        ground_clearance: float = 0.1
    ):
        """
        Initialize voxel grid.
        
        Args:
            bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
            voxel_size: Size of each voxel cube (meters)
            ground_clearance: Height below which voxels are always occupied (ground plane)
        """
        self.voxel_size = voxel_size
        self.ground_clearance = ground_clearance
        
        # World bounds
        self.x_min, self.x_max = bounds[0]
        self.y_min, self.y_max = bounds[1]
        self.z_min, self.z_max = bounds[2]
        
        # Grid dimensions (number of voxels in each axis)
        self.nx = int(np.ceil((self.x_max - self.x_min) / voxel_size))
        self.ny = int(np.ceil((self.y_max - self.y_min) / voxel_size))
        self.nz = int(np.ceil((self.z_max - self.z_min) / voxel_size))
        
        # Occupied voxels (set of (ix, iy, iz) tuples for memory efficiency)
        self.occupied: Set[Tuple[int, int, int]] = set()
        
        print(f"[VoxelGrid] Created {self.nx}×{self.ny}×{self.nz} grid "
              f"({self.nx*self.ny*self.nz} voxels, {voxel_size}m resolution)")
    
    def world_to_grid(self, position: np.ndarray) -> Tuple[int, int, int]:
        """
        Convert world coordinates to grid indices.
        
        Args:
            position: [x, y, z] in world frame
            
        Returns:
            (ix, iy, iz) grid indices
        """
        ix = int(np.floor((position[0] - self.x_min) / self.voxel_size))
        iy = int(np.floor((position[1] - self.y_min) / self.voxel_size))
        iz = int(np.floor((position[2] - self.z_min) / self.voxel_size))
        
        # Clamp to grid bounds
        ix = np.clip(ix, 0, self.nx - 1)
        iy = np.clip(iy, 0, self.ny - 1)
        iz = np.clip(iz, 0, self.nz - 1)
        
        return (ix, iy, iz)
    
    def grid_to_world(self, grid_idx: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert grid indices to world coordinates (voxel center).
        
        Args:
            grid_idx: (ix, iy, iz) grid indices
            
        Returns:
            [x, y, z] position at voxel center
        """
        ix, iy, iz = grid_idx
        x = self.x_min + (ix + 0.5) * self.voxel_size
        y = self.y_min + (iy + 0.5) * self.voxel_size
        z = self.z_min + (iz + 0.5) * self.voxel_size
        return np.array([x, y, z])
    
    def is_occupied(self, position: np.ndarray) -> bool:
        """
        Check if world position is inside an occupied voxel.
        
        Args:
            position: [x, y, z] in world frame
            
        Returns:
            True if voxel is occupied
        """
        # Ground plane check
        if position[2] < self.ground_clearance:
            return True
        
        # Grid bounds check
        if not (self.x_min <= position[0] <= self.x_max and
                self.y_min <= position[1] <= self.y_max and
                self.z_min <= position[2] <= self.z_max):
            return True  # Out of bounds = occupied
        
        grid_idx = self.world_to_grid(position)
        return grid_idx in self.occupied
    
    def mark_occupied(self, position: np.ndarray):
        """
        Mark voxel at world position as occupied.
        
        Args:
            position: [x, y, z] in world frame
        """
        grid_idx = self.world_to_grid(position)
        self.occupied.add(grid_idx)
    
    def mark_box_occupied(
        self,
        center: np.ndarray,
        half_extents: np.ndarray
    ):
        """
        Mark all voxels inside an axis-aligned box as occupied.
        
        Args:
            center: [x, y, z] box center
            half_extents: [hx, hy, hz] box half-extents
        """
        # Calculate box bounds
        x_min = center[0] - half_extents[0]
        x_max = center[0] + half_extents[0]
        y_min = center[1] - half_extents[1]
        y_max = center[1] + half_extents[1]
        z_min = center[2] - half_extents[2]
        z_max = center[2] + half_extents[2]
        
        # Convert to grid indices
        ix_min, iy_min, iz_min = self.world_to_grid(np.array([x_min, y_min, z_min]))
        ix_max, iy_max, iz_max = self.world_to_grid(np.array([x_max, y_max, z_max]))
        
        # Mark all voxels in the box
        count = 0
        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                for iz in range(iz_min, iz_max + 1):
                    self.occupied.add((ix, iy, iz))
                    count += 1
        
        return count
    
    def mark_sphere_occupied(
        self,
        center: np.ndarray,
        radius: float
    ):
        """
        Mark all voxels inside a sphere as occupied.
        
        Args:
            center: [x, y, z] sphere center
            radius: Sphere radius
        """
        # Bounding box for sphere
        x_min = center[0] - radius
        x_max = center[0] + radius
        y_min = center[1] - radius
        y_max = center[1] + radius
        z_min = center[2] - radius
        z_max = center[2] + radius
        
        ix_min, iy_min, iz_min = self.world_to_grid(np.array([x_min, y_min, z_min]))
        ix_max, iy_max, iz_max = self.world_to_grid(np.array([x_max, y_max, z_max]))
        
        # Check each voxel in bounding box
        count = 0
        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                for iz in range(iz_min, iz_max + 1):
                    voxel_center = self.grid_to_world((ix, iy, iz))
                    
                    # Check if voxel center is inside sphere
                    if np.linalg.norm(voxel_center - center) <= radius:
                        self.occupied.add((ix, iy, iz))
                        count += 1
        
        return count
    
    def is_path_clear(
        self,
        start: np.ndarray,
        end: np.ndarray,
        safety_margin: float = 0.0
    ) -> bool:
        """
        Check if straight-line path between two points is collision-free.
        
        Uses raycast through voxel grid (DDA algorithm).
        
        Args:
            start: [x, y, z] start position
            end: [x, y, z] end position
            safety_margin: Additional clearance (meters)
            
        Returns:
            True if path is clear
        """
        # Check endpoints
        if self.is_occupied(start) or self.is_occupied(end):
            return False
        
        # Sample points along path
        direction = end - start
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return not self.is_occupied(start)
        
        # Sample every 0.25*voxel_size to ensure we don't miss voxels
        num_samples = int(np.ceil(distance / (self.voxel_size * 0.25)))
        
        for i in range(num_samples + 1):
            t = i / num_samples
            point = start + t * direction
            
            if self.is_occupied(point):
                return False
            
            # Check safety margin (simple sphere around point)
            if safety_margin > 0:
                for dx in [-safety_margin, 0, safety_margin]:
                    for dy in [-safety_margin, 0, safety_margin]:
                        for dz in [-safety_margin, 0, safety_margin]:
                            test_point = point + np.array([dx, dy, dz])
                            if self.is_occupied(test_point):
                                return False
        
        return True
    
    def get_random_free_position(
        self,
        z_range: Optional[Tuple[float, float]] = None,
        max_attempts: int = 100,
        safety_margin: float = 0.2
    ) -> Optional[np.ndarray]:
        """
        Generate random position in free space.
        
        Args:
            z_range: (z_min, z_max) altitude constraints
            max_attempts: Maximum sampling attempts
            safety_margin: Minimum distance from obstacles (meters)
            
        Returns:
            [x, y, z] position or None if failed
        """
        if z_range is None:
            z_min = self.z_min
            z_max = self.z_max
        else:
            z_min, z_max = z_range
        
        for _ in range(max_attempts):
            # Random position in bounds
            position = np.array([
                np.random.uniform(self.x_min + safety_margin, self.x_max - safety_margin),
                np.random.uniform(self.y_min + safety_margin, self.y_max - safety_margin),
                np.random.uniform(z_min, z_max)
            ])
            
            # Check if clear
            if not self.is_occupied(position):
                # Check safety margin
                clear = True
                if safety_margin > 0:
                    for dx in np.linspace(-safety_margin, safety_margin, 3):
                        for dy in np.linspace(-safety_margin, safety_margin, 3):
                            for dz in np.linspace(-safety_margin, safety_margin, 3):
                                test_point = position + np.array([dx, dy, dz])
                                if self.is_occupied(test_point):
                                    clear = False
                                    break
                            if not clear:
                                break
                        if not clear:
                            break
                
                if clear:
                    return position
        
        return None  # Failed to find free position
    
    def clear(self):
        """Clear all occupied voxels."""
        self.occupied.clear()
    
    def get_occupancy_stats(self) -> dict:
        """
        Get occupancy statistics.
        
        Returns:
            Dictionary with occupancy info
        """
        total_voxels = self.nx * self.ny * self.nz
        occupied_voxels = len(self.occupied)
        occupancy_ratio = occupied_voxels / total_voxels if total_voxels > 0 else 0
        
        return {
            'total_voxels': total_voxels,
            'occupied_voxels': occupied_voxels,
            'free_voxels': total_voxels - occupied_voxels,
            'occupancy_ratio': occupancy_ratio,
            'grid_shape': (self.nx, self.ny, self.nz),
            'voxel_size': self.voxel_size
        }
