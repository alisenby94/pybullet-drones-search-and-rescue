"""
Random Obstacle Generation for Training Environments

PURPOSE:
    Generate random obstacles (boxes, spheres, cylinders) in PyBullet.
    Register obstacles with voxel grid for collision checking.
    
FEATURES:
    - Multiple obstacle types (box, sphere, cylinder)
    - Random sizes, positions, and colors
    - Automatic voxel grid registration
    - Collision-free placement
"""

import numpy as np
import pybullet as p
from typing import List, Tuple, Optional
from src.utils.voxel_grid import VoxelGrid


class ObstacleGenerator:
    """
    Generate random obstacles for drone navigation environments.
    """
    
    def __init__(
        self,
        physics_client: int,
        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        voxel_grid: Optional[VoxelGrid] = None
    ):
        """
        Initialize obstacle generator.
        
        Args:
            physics_client: PyBullet physics client ID
            bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
            voxel_grid: Optional voxel grid for occupancy tracking
        """
        self.client = physics_client
        self.bounds = bounds
        self.voxel_grid = voxel_grid
        
        self.obstacle_ids: List[int] = []
        self.obstacle_info: List[dict] = []
    
    def clear_obstacles(self):
        """Remove all generated obstacles."""
        for obstacle_id in self.obstacle_ids:
            try:
                p.removeBody(obstacle_id, physicsClientId=self.client)
            except:
                pass
        
        self.obstacle_ids.clear()
        self.obstacle_info.clear()
        
        if self.voxel_grid is not None:
            self.voxel_grid.clear()
    
    def add_box_obstacle(
        self,
        position: np.ndarray,
        half_extents: np.ndarray,
        color: Optional[np.ndarray] = None,
        mass: float = 0.0
    ) -> int:
        """
        Add box obstacle.
        
        Args:
            position: [x, y, z] center position
            half_extents: [hx, hy, hz] box half-extents
            color: [r, g, b, a] color (default: random)
            mass: Object mass (0 = static)
            
        Returns:
            PyBullet body ID
        """
        if color is None:
            color = np.random.uniform([0.3, 0.3, 0.3, 1.0], [1.0, 1.0, 1.0, 1.0])
        
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.client
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=self.client
        )
        
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self.client
        )
        
        self.obstacle_ids.append(body_id)
        self.obstacle_info.append({
            'type': 'box',
            'position': position.copy(),
            'half_extents': half_extents.copy(),
            'color': color.copy(),
            'body_id': body_id
        })
        
        # Update voxel grid
        if self.voxel_grid is not None:
            self.voxel_grid.mark_box_occupied(position, half_extents)
        
        return body_id
    
    def add_sphere_obstacle(
        self,
        position: np.ndarray,
        radius: float,
        color: Optional[np.ndarray] = None,
        mass: float = 0.0
    ) -> int:
        """
        Add sphere obstacle.
        
        Args:
            position: [x, y, z] center position
            radius: Sphere radius
            color: [r, g, b, a] color (default: random)
            mass: Object mass (0 = static)
            
        Returns:
            PyBullet body ID
        """
        if color is None:
            color = np.random.uniform([0.3, 0.3, 0.3, 1.0], [1.0, 1.0, 1.0, 1.0])
        
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius,
            physicsClientId=self.client
        )
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color,
            physicsClientId=self.client
        )
        
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self.client
        )
        
        self.obstacle_ids.append(body_id)
        self.obstacle_info.append({
            'type': 'sphere',
            'position': position.copy(),
            'radius': radius,
            'color': color.copy(),
            'body_id': body_id
        })
        
        # Update voxel grid
        if self.voxel_grid is not None:
            self.voxel_grid.mark_sphere_occupied(position, radius)
        
        return body_id
    
    def add_cylinder_obstacle(
        self,
        position: np.ndarray,
        radius: float,
        height: float,
        color: Optional[np.ndarray] = None,
        mass: float = 0.0
    ) -> int:
        """
        Add cylinder obstacle.
        
        Args:
            position: [x, y, z] center position
            radius: Cylinder radius
            height: Cylinder height
            color: [r, g, b, a] color (default: random)
            mass: Object mass (0 = static)
            
        Returns:
            PyBullet body ID
        """
        if color is None:
            color = np.random.uniform([0.3, 0.3, 0.3, 1.0], [1.0, 1.0, 1.0, 1.0])
        
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=height,
            physicsClientId=self.client
        )
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=color,
            physicsClientId=self.client
        )
        
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self.client
        )
        
        self.obstacle_ids.append(body_id)
        self.obstacle_info.append({
            'type': 'cylinder',
            'position': position.copy(),
            'radius': radius,
            'height': height,
            'color': color.copy(),
            'body_id': body_id
        })
        
        # Update voxel grid (approximate as box for simplicity)
        if self.voxel_grid is not None:
            half_extents = np.array([radius, radius, height / 2.0])
            self.voxel_grid.mark_box_occupied(position, half_extents)
        
        return body_id
    
    def generate_random_obstacles(
        self,
        num_obstacles: int = 10,
        obstacle_types: List[str] = ['box', 'sphere', 'cylinder'],
        size_range: Tuple[float, float] = (0.2, 0.6),
        z_range: Optional[Tuple[float, float]] = None,
        min_spacing: float = 0.5,
        colorful: bool = True,
        exclusion_zones: Optional[List[Tuple[np.ndarray, float]]] = None
    ) -> List[int]:
        """
        Generate random obstacles in the environment.
        
        Args:
            num_obstacles: Number of obstacles to generate
            obstacle_types: List of obstacle types to use
            size_range: (min_size, max_size) for obstacle dimensions
            z_range: (z_min, z_max) altitude range (default: full bounds)
            min_spacing: Minimum distance between obstacle centers
            colorful: Use colorful obstacles (else grayscale)
            exclusion_zones: List of (center, radius) tuples defining no-spawn zones
            
        Returns:
            List of PyBullet body IDs
        """
        if z_range is None:
            z_range = self.bounds[2]
        
        new_obstacle_ids = []
        
        for i in range(num_obstacles):
            # Random obstacle type
            obs_type = np.random.choice(obstacle_types)
            
            # Random position (avoid existing obstacles if voxel grid exists)
            max_attempts = 50
            position = None
            
            for attempt in range(max_attempts):
                pos = np.array([
                    np.random.uniform(self.bounds[0][0] + 1.0, self.bounds[0][1] - 1.0),
                    np.random.uniform(self.bounds[1][0] + 1.0, self.bounds[1][1] - 1.0),
                    np.random.uniform(z_range[0], z_range[1])
                ])
                
                # Check exclusion zones (e.g., drone spawn point)
                in_exclusion_zone = False
                if exclusion_zones is not None:
                    for zone_center, zone_radius in exclusion_zones:
                        dist_to_zone = np.linalg.norm(pos - zone_center)
                        if dist_to_zone < zone_radius:
                            in_exclusion_zone = True
                            break
                
                if in_exclusion_zone:
                    continue
                
                # Check spacing from existing obstacles
                too_close = False
                for info in self.obstacle_info:
                    dist = np.linalg.norm(pos - info['position'])
                    if dist < min_spacing:
                        too_close = True
                        break
                
                if not too_close:
                    position = pos
                    break
            
            if position is None:
                # print(f"Warning: Could not place obstacle {i+1}/{num_obstacles}")
                continue
            
            # Random size
            size = np.random.uniform(size_range[0], size_range[1])
            
            # Random color
            if colorful:
                color = np.array([
                    np.random.uniform(0.3, 1.0),
                    np.random.uniform(0.3, 1.0),
                    np.random.uniform(0.3, 1.0),
                    1.0
                ])
            else:
                gray = np.random.uniform(0.4, 0.8)
                color = np.array([gray, gray, gray, 1.0])
            
            # Create obstacle
            try:
                if obs_type == 'box':
                    half_extents = np.random.uniform(size * 0.5, size * 1.5, 3)
                    body_id = self.add_box_obstacle(position, half_extents, color)
                    new_obstacle_ids.append(body_id)
                
                elif obs_type == 'sphere':
                    radius = size
                    body_id = self.add_sphere_obstacle(position, radius, color)
                    new_obstacle_ids.append(body_id)
                
                elif obs_type == 'cylinder':
                    radius = size * 0.7
                    height = np.random.uniform(size * 1.0, size * 2.5)
                    body_id = self.add_cylinder_obstacle(position, radius, height, color)
                    new_obstacle_ids.append(body_id)
            
            except Exception as e:
                print(f"Warning: Failed to create obstacle {i+1}: {e}")
        
        print(f"[ObstacleGenerator] Created {len(new_obstacle_ids)}/{num_obstacles} obstacles")
        
        if self.voxel_grid is not None:
            stats = self.voxel_grid.get_occupancy_stats()
            print(f"[ObstacleGenerator] Voxel occupancy: {stats['occupied_voxels']}/{stats['total_voxels']} "
                  f"({stats['occupancy_ratio']*100:.1f}%)")
        
        return new_obstacle_ids
    
    def get_obstacle_count(self) -> int:
        """Get number of obstacles."""
        return len(self.obstacle_ids)
    
    def get_obstacle_info(self) -> List[dict]:
        """Get list of obstacle information dictionaries."""
        return self.obstacle_info.copy()
