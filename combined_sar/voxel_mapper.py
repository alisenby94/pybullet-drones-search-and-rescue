"""
2D Perpendicular Band Voxel Mapper

Lightweight 2D occupancy grid mapping from depth measurements.
Only points in a narrow perpendicular band around the camera principal axis
are projected to the ground plane for efficient obstacle detection.
"""

import numpy as np
from pathlib import Path


class PerpendicularBand2DMapper:
    """2D voxel mapper from perpendicular depth band"""
    
    def __init__(
        self,
        world_size=60.0,
        resolution=0.25,
        fov_deg=60.0,
        img_width=64,
        img_height=48,
        pixels_half_width=2,
        min_depth=0.15,
        max_depth=30.0,
        debug=False,
        height_band=5.0,
        allowed_regions=None,
    ):
        self.world_size = world_size
        self.resolution = resolution
        self.grid_dim = int(world_size / resolution)
        self.grid = np.zeros((self.grid_dim, self.grid_dim), dtype=np.uint8)
        
        self.W = img_width
        self.H = img_height
        self.fov = np.deg2rad(fov_deg)
        self.fx = self.W / (2 * np.tan(self.fov / 2))
        self.fy = self.H / (2 * np.tan(self.fov / 2))
        self.cx = self.W / 2
        self.cy = self.H / 2
        
        self.pixels_half_width_u = self.W // 2
        self.pixels_half_width_v = pixels_half_width
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.debug = debug
        self.height_band = float(height_band)
        
        self.allowed_regions = None
        if allowed_regions is not None:
            self.allowed_regions = [
                (np.asarray(min_xy, dtype=float), np.asarray(max_xy, dtype=float))
                for (min_xy, max_xy) in allowed_regions
            ]
    
    def reset(self):
        """Clear the grid"""
        self.grid.fill(0)
    
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        gx = int((x + self.world_size / 2) / self.resolution)
        gy = int((y + self.world_size / 2) / self.resolution)
        if gx < 0 or gy < 0 or gx >= self.grid_dim or gy >= self.grid_dim:
            return None
        return gx, gy
    
    def grid_to_world(self, gx, gy):
        """Convert grid indices to world coordinates"""
        wx = gx * self.resolution - self.world_size / 2
        wy = gy * self.resolution - self.world_size / 2
        return np.array([wx, wy])
    
    def integrate_depth(self, depth_m, drone_pos, drone_rpy):
        """Integrate depth measurements into grid"""
        updated = 0
        depth_vals = []
        
        camera_rpy = np.array([0.0, -drone_rpy[1], drone_rpy[2]])
        R_camera = self._rpy_to_matrix(camera_rpy)
        
        for du in range(-self.pixels_half_width_u, self.pixels_half_width_u + 1):
            for dv in range(-self.pixels_half_width_v, self.pixels_half_width_v + 1):
                u = int(self.cx + du)
                v = int(self.cy + dv)
                if not (0 <= u < self.W and 0 <= v < self.H):
                    continue
                
                depth = depth_m[v, u]
                if depth < self.min_depth or depth > self.max_depth:
                    continue
                
                depth_vals.append(depth)
                
                dx = (u - self.cx) / self.fx
                dy = (v - self.cy) / self.fy
                
                point_cam = np.array([dx * depth, dy * depth, depth])
                point_body = np.array([point_cam[2], -point_cam[0], -point_cam[1]])
                point_world = drone_pos + R_camera @ point_body
                
                if abs(point_world[2] - drone_pos[2]) > self.height_band:
                    continue
                
                wx, wy = point_world[0], point_world[1]
                
                if self.allowed_regions is not None:
                    inside_any = False
                    for (min_xy, max_xy) in self.allowed_regions:
                        if (min_xy[0] <= wx <= max_xy[0]) and (min_xy[1] <= wy <= max_xy[1]):
                            inside_any = True
                            break
                    if not inside_any:
                        continue
                
                g = self.world_to_grid(wx, wy)
                if g is None:
                    continue
                self.grid[g[0], g[1]] = 1
                updated += 1
        
        if self.debug and depth_vals:
            occ = int(np.count_nonzero(self.grid))
            depth_range = f"[{np.min(depth_vals):.2f}, {np.max(depth_vals):.2f}]m"
            print(f"[VoxelMapper] Integrated {updated}/{len(depth_vals)} rays, "
                  f"depth: {depth_range}, occupied: {occ}")
        
        return updated
    
    def get_coverage(self, dynamic=True):
        """Get grid coverage percentage"""
        occupied = np.argwhere(self.grid > 0)
        if occupied.size == 0:
            return 0.0
        if not dynamic:
            return float(occupied.shape[0]) / float(self.grid_dim * self.grid_dim)
        mins = occupied.min(axis=0)
        maxs = occupied.max(axis=0)
        bbox_cells = (maxs[0] - mins[0] + 1) * (maxs[1] - mins[1] + 1)
        return float(occupied.shape[0]) / float(bbox_cells)
    
    def export_png(self, save_path=None, step=None):
        """Export grid to PNG
        
        Args:
            save_path: Either a directory path or a full file path ending in .png
            step: Optional step number for filename generation
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("[VoxelMapper] Warning: matplotlib not available, skipping PNG export")
            return None
        
        if save_path is None:
            out_dir = Path("results/voxel_maps")
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = "perp2d_latest.png" if step is None else f"perp2d_{step:04d}.png"
            out_path = out_dir / fname
        else:
            save_path = Path(save_path)
            if save_path.suffix.lower() == '.png':
                # Full filename provided
                out_path = save_path
                save_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Directory provided
                save_path.mkdir(parents=True, exist_ok=True)
                fname = "perp2d_latest.png" if step is None else f"perp2d_{step:04d}.png"
                out_path = save_path / fname
        
        fig, ax = plt.subplots(figsize=(6, 6))
        extent = [-self.world_size / 2, self.world_size / 2, -self.world_size / 2, self.world_size / 2]
        ax.imshow(self.grid.T, origin='lower', cmap='gray', extent=extent)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('2D Occupancy Map')
        ax.set_aspect('equal')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return str(out_path)
    
    def _rpy_to_matrix(self, rpy):
        """Convert roll-pitch-yaw to rotation matrix"""
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return Rz @ Ry @ Rx
