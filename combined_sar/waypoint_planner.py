"""Waypoint planning from LM actions"""

from typing import List, Optional
import numpy as np


class WaypointPlanner:
    """Convert LM actions to waypoint sequences"""
    
    def __init__(self, hover_alt=1.0, max_step=2.0):
        """Initialize waypoint planner.
        
        Args:
            hover_alt: Altitude for all waypoints (1.0m matches drone start altitude)
            max_step: Maximum step size between waypoints (2.0m reduces waypoints vs 0.6m)
        """
        self.hover_alt = hover_alt
        self.max_step = max_step
    
    def plan_explore(self, start_pos, direction, distance) -> List[np.ndarray]:
        """Explore in direction"""
        angles = {
            "N": 90, "NE": 45, "E": 0, "SE": 315,
            "S": 270, "SW": 225, "W": 180, "NW": 135
        }
        angle_rad = np.radians(angles.get(direction, 0))
        target_x = start_pos[0] + distance * np.cos(angle_rad)
        target_y = start_pos[1] + distance * np.sin(angle_rad)
        target = np.array([target_x, target_y, self.hover_alt])
        return self._segment_path(start_pos, target)
    
    def plan_scan(self, pos) -> List[np.ndarray]:
        """Scan with small square loop"""
        loop_r = 1.0
        return [
            np.array([pos[0] + loop_r, pos[1], self.hover_alt]),
            np.array([pos[0] + loop_r, pos[1] + loop_r, self.hover_alt]),
            np.array([pos[0], pos[1] + loop_r, self.hover_alt]),
            np.array([pos[0], pos[1], self.hover_alt]),
        ]
    
    def plan_explore_obstacle(self, start_pos, direction) -> List[np.ndarray]:
        """Circumnavigate obstacle"""
        wps = []
        angles = {
            "N": 90, "NE": 45, "E": 0, "SE": 315,
            "S": 270, "SW": 225, "W": 180, "NW": 135
        }
        angle_rad = np.radians(angles.get(direction, 0))
        approach_target = start_pos + 2.0 * np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
        wps.extend(self._segment_path(start_pos, approach_target))
        
        # Octagon circumnavigation
        for i in range(8):
            theta = 2 * np.pi * i / 8
            circ = approach_target[:2] + 6.0 * np.array([np.cos(theta), np.sin(theta)])
            wps.append(np.array([circ[0], circ[1], self.hover_alt]))
        
        return wps
    
    def _segment_path(self, start, end, step_size=None):
        """Segment path with maximum step size to reduce waypoints.
        
        Larger step_size = fewer waypoints = drone moves faster.
        
        Example: 20m distance
        - With 2.0m step_size: 10 waypoints
        - With 0.6m step_size: 34 waypoints (3x more!)
        
        This lesson from combine: fewer waypoints = cleaner movement.
        
        Args:
            start: Starting position
            end: Ending position
            step_size: Maximum distance between waypoints (default from __init__)
            
        Returns:
            List of waypoints from start to end
        """
        if step_size is None:
            step_size = self.max_step
        
        delta = end - start
        dist = np.linalg.norm(delta)
        
        if dist < 0.01:
            return [end]
        
        num_steps = max(1, int(np.ceil(dist / step_size)))
        wps = []
        
        for i in range(1, num_steps + 1):
            t = i / num_steps
            wp = start + t * delta
            wps.append(wp)
        
        return wps
