"""
Action Coordinator Environment (Stage 1)

PURPOSE:
    Learn high-level navigation using velocity + heading commands.
    
INPUTS:
    - Current velocity (3D)
    - Angular velocity (3D)
    - Vector to waypoint (3D)
    - Tracking compliance metric (1D)
    - Stereovision depth map (512D) - optional
    
OUTPUTS:
    - Velocity commands [vx, vy, vz] in range [-1, 1] (scaled to ±1.0 m/s)
    - Target heading [yaw] in range [-1, 1] (scaled to ±π rad = ±180°)
    
REWARD:
    - Progress toward waypoint
    - Roll/pitch stability
    - Altitude control
    - Survival bonus
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../simulation/gym-pybullet-drones'))

import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class ActionCoordinatorEnv(VelocityAviary):
    """
    Stage 1: High-level action coordinator environment.
    
    Extends the library's VelocityAviary to add:
    - Yaw control (4th action component)
    - Waypoint navigation rewards
    - Optional obstacle avoidance
    - Optional stereovision
    
    Uses library's PID controller for low-level control (exact same as working test files).
    """
    
    def __init__(self, gui=False, enable_vision=False, enable_streaming=False, 
                 num_obstacles=10, enable_obstacles=True):
        """
        Initialize action coordinator environment.
        
        Args:
            gui: Enable PyBullet GUI
            enable_vision: Enable stereovision system (adds 512D to observation)
            enable_streaming: Enable video streaming to VLC (requires enable_vision=True)
            num_obstacles: Number of random obstacles to generate
            enable_obstacles: Enable obstacle generation
        """
        self.control_freq = 48  # Action coordinator runs at 48 Hz (PID frequency)
        # Note: Originally intended 10 Hz but super().step() runs one PID cycle
        # 48 Hz gives agent more control authority which helps learning!
        self.max_episode_steps = 500
        self.current_step = 0
        
        # Velocity + Heading control constants
        # CRITICAL: This MUST match what we set in super().__init__() later!
        # VelocityAviary sets SPEED_LIMIT = 0.03 * MAX_SPEED_KMH * (1000/3600) ≈ 0.25 m/s
        # We override it to 2.0 m/s after super().__init__()
        self.MAX_VELOCITY = 2.0  # m/s - maximum velocity in any direction
        
        # Waypoints
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_radius = 1.0  # m
        self.waypoints_reached = 0
        
        # Curriculum learning - late-stage waypoint perturbations
        self.curriculum_timesteps = 0  # Tracks total timesteps across all episodes
        self.perturbation_start_timesteps = 1_000_000  # Start after 1M timesteps
        self.perturbation_probability = 0.1  # 10% chance per step
        
        # State tracking
        self.previous_pos = np.zeros(3)
        self.previous_vel = np.zeros(3)
        self.previous_action = np.zeros(4)  # For smoothness penalty
        self.compliance = 1.0  # Tracking quality (placeholder, not used in RPM mode)
        
        # Velocity tracking metrics
        self.target_velocity = np.zeros(3)  # Commanded velocity to PID
        self.velocity_tracking_error = 0.0  # |target_vel - actual_vel|
        
        # Obstacle configuration
        self.enable_obstacles = enable_obstacles
        self.num_obstacles = num_obstacles
        self.voxel_grid = None
        self.obstacle_generator = None
        
        # Vision system (optional)
        self.enable_vision = enable_vision
        self.vision_system = None
        if enable_vision:
            from src.vision.stereo_vision import StereoVisionSystem
            self.vision_system = StereoVisionSystem(
                baseline=0.06,  # 6cm stereo baseline
                resolution=(160, 120),  # Reduced from 640x480 for speed
                downsample_size=(32, 16),  # 512D vision features (reduced from 2048D)
                enable_streaming=enable_streaming,
                stream_port=5555,
                verbose=False  # Disable debug output during training
            )
            print(f"[ActionCoordinator] Stereovision enabled (512D features, optimized for training)")
            if enable_streaming:
                print(f"[ActionCoordinator] Video streaming enabled - open in VLC: http://localhost:5555/stream")
        
        # VelocityYawAviary handles velocity + yaw control with internal PID
        # Action coordinator runs at 10 Hz, VelocityYawAviary at 48 Hz
        
        super().__init__(
            gui=gui,
            num_drones=1,
            initial_xyzs=np.array([[0, 0, 1.75]]),  # Middle of waypoint range [1.0, 2.5]
            initial_rpys=np.zeros((1, 3)),
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=48,  # VelocityYawAviary control frequency (internal PID)
            record=False
        )
        
        # Override SPEED_LIMIT to match our MAX_VELOCITY (2.0 m/s)
        # Parent class sets it to ~0.25 m/s which is too slow
        self.SPEED_LIMIT = self.MAX_VELOCITY
        print(f"[ActionCoordinator] SPEED_LIMIT set to {self.SPEED_LIMIT} m/s")
        
        # Initialize voxel grid and obstacles
        if self.enable_obstacles:
            from src.utils.voxel_grid import VoxelGrid
            from src.utils.obstacle_generator import ObstacleGenerator
            
            # Voxel grid bounds (match waypoint generation space)
            bounds = ((-8.0, 8.0), (-8.0, 8.0), (0.5, 3.0))
            self.voxel_grid = VoxelGrid(bounds=bounds, voxel_size=0.5, ground_clearance=0.1)
            
            # Get physics client ID from PyBullet (assumes single client)
            import pybullet as p
            physics_client = p.getConnectionInfo()['connectionMethod']  # This gives the client type, not ID
            # For single client, use default 0
            physics_client = 0
            
            self.obstacle_generator = ObstacleGenerator(
                physics_client=physics_client,
                bounds=bounds,
                voxel_grid=self.voxel_grid
            )
            
            print(f"[ActionCoordinator] Obstacle system initialized (voxel_size=0.5m)")
    
    def _actionSpace(self):
        """
        Action: Velocity direction + heading commands [-1, 1]^4
        
        CRITICAL: This matches VelocityAviary pattern!
        
        Commands: [vx_dir, vy_dir, vz_dir, target_heading]
        - vx_dir, vy_dir, vz_dir: Velocity DIRECTION (normalized internally)
        - target_heading: Absolute yaw angle in [-1, 1] → [-π, +π] radians
        
        The magnitude of [vx_dir, vy_dir, vz_dir] acts as speed_fraction ∈ [0, 1]
        Final velocity = SPEED_LIMIT * speed_fraction * normalized_direction
        
        Example:
            [1.0, 0.0, 0.0, 0.0] → Full speed East, heading East
            [0.5, 0.5, 0.0, 0.5] → Half speed NE, heading 90° right
        
        Returns:
            Box space (4,) in range [-1, 1]
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    
    def _observationSpace(self):
        """
        Observation: Raw sensor data for GRU to learn from
        
        Components (10D or 522D depending on vision):
            Without vision (10D):
                - Velocity (world frame): 3D       (how am I moving?)
                - Angular velocity (world frame): 3D (how am I rotating?)
                - Vector to waypoint: 3D            (where to go?)
                - Tracking compliance: 1D           (placeholder, not used)
            
            With vision (522D):
                - Same 10D sensors as above
                - Stereovision depth map: 512D (32x16 attention-weighted depth, optimized)
            
        Philosophy: Direct RPM control → deterministic physics.
        GRU learns F=ma dynamics directly without hidden control layers.
            
        Returns:
            Box space (10,) or (10 + vision_size) depending on enable_vision
        """
        base_obs_size = 10  # vel(3) + ang_vel(3) + vec_to_wp(3) + compliance(1)
        
        if self.enable_vision and self.vision_system is not None:
            # Calculate vision observation size from downsample dimensions
            vision_size = self.vision_system.downsample_size[0] * self.vision_system.downsample_size[1]
            obs_size = base_obs_size + vision_size
        else:
            obs_size = base_obs_size
            
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
    
    def _computeObs(self):
        """Compute observation: raw sensor data with minimal preprocessing."""
        state = self._getDroneStateVector(0)
        
        # Raw sensor data (world frame - natural sensor output)
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        ang_vel = state[13:16]
        
        # Only "preprocessing": compute vector to current waypoint
        current_wp = self.waypoints[self.current_waypoint_idx]
        vec_to_waypoint = current_wp - pos
        
        # Update previous_vel for reward computation (still needed there)
        self.previous_vel = vel.copy()
        
        # Construct base observation (10D - raw sensors)
        # Philosophy: Direct RPM control → simple, deterministic physics
        obs = np.concatenate([
            vel,                # 3 - velocity (world frame, natural sensor output)
            ang_vel,            # 3 - angular velocity (world frame, natural sensor output)
            vec_to_waypoint,    # 3 - vector to goal (direction + distance combined)
            [self.compliance]   # 1 - placeholder (not used in RPM mode)
        ])
        
        # Add vision if enabled (2048D stereo depth with attention)
        if self.enable_vision and self.vision_system is not None:
            vision_obs = self.vision_system.get_vision_observation(
                drone_pos=pos,
                drone_rpy=rpy,
                waypoint_pos=current_wp
            )
            obs = np.concatenate([obs, vision_obs])
        
        return obs.astype(np.float32)
    
    def _computeReward(self):
        """
        OBJECTIVE-ORIENTED REWARDS: Clear goals with strong signals
        
        Primary Goal: Reach waypoints as fast as possible
        Secondary Goal: Fly forward (stereo vision requirement)
        Tertiary Goal: Don't crash
        
        Returns:
            float: Total reward
        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        rpy = state[7:10]
        
        # Initialize reward
        reward = 0.0
        
        # ============================================================================
        # PRIMARY OBJECTIVE: Reach waypoints (DOMINANT reward component)
        # ============================================================================
        current_wp = self.waypoints[self.current_waypoint_idx]
        dist = np.linalg.norm(pos - current_wp)
        prev_dist = np.linalg.norm(self.previous_pos - current_wp)
        progress = prev_dist - dist
        
        # STRONG progress reward: Moving toward waypoint is THE GOAL
        # At 2 m/s, can move 0.2m per step → reward of +200 per step
        reward_progress = 1000.0 * progress  # 10x stronger than before
        reward += reward_progress
        
        # MASSIVE waypoint completion bonus
        if dist < self.waypoint_radius:
            reward += 10000.0  # 10x stronger - reaching waypoint is SUCCESS
        
        # ============================================================================
        # SECONDARY OBJECTIVE: Fly forward (stereo vision requirement)
        # ============================================================================
        # Body-frame forward direction (positive X in body frame)
        yaw = rpy[2]
        forward_dir = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        vel_forward = np.dot(vel, forward_dir)
        
        # LINEAR RAMP: Reward increases smoothly from 0
        # This makes it easier to learn - any forward motion is rewarded proportionally
        # IMPORTANT: This should be MUCH weaker than progress reward (1000.0)
        # We want the agent to prioritize reaching waypoints over flying forward
        forward_weight = 50.0  # Reduced 5x - progress toward waypoint is more important
        reward_forward = forward_weight * vel_forward  # Linear: 0 at stationary, +20 at 2 m/s
        
        # NO backward penalty - let the agent move however it needs to reach waypoints
        # The stereo vision is a nice-to-have, not a hard requirement
        # Waypoint progress (1000x) is 100x stronger than forward penalty anyway
        
        reward += reward_forward
        
        # ============================================================================
        # TERTIARY OBJECTIVE: Time penalty (encourages speed)
        # ============================================================================
        # Small constant penalty per timestep - encourages completing mission quickly
        reward_time = -1.0  # -1 per step, -500 total if using all 500 steps
        reward += reward_time
        
        # ============================================================================
        # SAFETY: Crash avoidance (only penalize dangerous situations)
        # ============================================================================
        # Ground proximity penalty (only when dangerously low)
        min_safe_altitude = 0.2
        danger_zone_start = 0.5  # Start warning below 0.5m
        
        if pos[2] < danger_zone_start:
            height_above_crash = max(0.01, pos[2] - min_safe_altitude)
            reward_ground_danger = -500.0 * np.exp(-height_above_crash / 0.2)
        else:
            reward_ground_danger = 0.0
        
        reward += reward_ground_danger
        
        # ============================================================================
        # Store metrics for logging
        # ============================================================================
        roll = rpy[0]
        pitch = rpy[1]
        speed = np.linalg.norm(vel)
        direction_to_wp = (current_wp - pos) / (dist + 1e-6)
        alignment = np.dot(vel, direction_to_wp) / (speed + 1e-6) if speed > 0.1 else 0.0
        
        reward_roll = 0.0  # Disabled - agent doesn't control orientation
        reward_pitch = 0.0  # Disabled
        reward_altitude = 0.0  # Disabled - waypoint 3D position is enough
        reward_vel_align = 0.0  # Disabled - progress reward handles this
        reward_hover = 0.0  # Disabled - not needed
        reward_overspeed = 0.0  # Disabled - action space limits speed
        is_final_waypoint = (self.current_waypoint_idx == len(self.waypoints) - 1)
        
        # Store metrics for logging
        vel_toward_wp = np.dot(vel, direction_to_wp)
        vel_lateral = vel - vel_toward_wp * direction_to_wp
        vel_lateral_mag = np.linalg.norm(vel_lateral)
        accel = (vel - self.previous_vel) / (1.0 / self.control_freq)
        accel_magnitude = np.linalg.norm(accel)
        altitude = pos[2]
        
        # Store metrics for logging
        vel_toward_wp = np.dot(vel, direction_to_wp)
        vel_lateral = vel - vel_toward_wp * direction_to_wp
        vel_lateral_mag = np.linalg.norm(vel_lateral)
        accel = (vel - self.previous_vel) / (1.0 / self.control_freq)
        accel_magnitude = np.linalg.norm(accel)
        altitude = pos[2]
        altitude_error = 0.0  # Not used anymore
        
        self._progress = progress
        self._reward_progress = reward_progress
        self._vel_toward_wp = vel_toward_wp
        self._vel_lateral_mag = vel_lateral_mag
        self._accel_magnitude = accel_magnitude
        self._vel_forward = vel_forward
        self._reward_forward = reward_forward
        self._reward_time = reward_time
        self._reward_hover = reward_hover
        self._distance_factor = 0.0
        self._alignment = alignment
        self._alignment_scale = 1.0
        self._reward_vel_align = reward_vel_align
        self._misalignment_factor = 0.0
        self._reward_lateral = 0.0
        self._reward_accel = 0.0
        self._roll = roll
        self._pitch = pitch
        self._reward_roll = reward_roll
        self._reward_pitch = reward_pitch
        self._altitude = altitude
        self._altitude_error = altitude_error
        self._reward_altitude = reward_altitude
        self._speed = speed
        self._reward_overspeed = reward_overspeed
        self._reward_global_altitude = reward_ground_danger
        
        # Obstacle proximity penalty (if voxel grid available)
        reward_obstacle = 0.0
        min_obstacle_dist = float('inf')
        if self.enable_obstacles and self.voxel_grid is not None:
            # Check distance to nearest obstacle using voxel grid
            # Sample points around drone to find closest obstacle
            search_radius = 2.0  # Check within 2m radius
            num_samples = 8  # Sample 8 directions around drone
            
            for i in range(num_samples):
                angle = 2 * np.pi * i / num_samples
                for r in np.linspace(0.1, search_radius, 10):
                    test_point = pos + np.array([
                        r * np.cos(angle),
                        r * np.sin(angle),
                        0.0
                    ])
                    
                    if self.voxel_grid.is_occupied(test_point):
                        dist_to_obstacle = r
                        min_obstacle_dist = min(min_obstacle_dist, dist_to_obstacle)
                        break
            
            # Exponential penalty for getting close to obstacles
            # No penalty beyond 1.5m, moderate penalty below 0.5m
            if min_obstacle_dist < 1.5:
                if min_obstacle_dist < 0.3:
                    # Critical proximity - very dangerous
                    reward_obstacle = -5.0 * np.exp(-(min_obstacle_dist - 0.3)**2 / 0.1)
                else:
                    # Warning zone - encourage keeping distance
                    safe_distance = 1.5
                    proximity_factor = (safe_distance - min_obstacle_dist) / safe_distance
                    reward_obstacle = -1.0 * proximity_factor**2
            
            reward += reward_obstacle
        
        self._min_obstacle_dist = min_obstacle_dist
        self._reward_obstacle = reward_obstacle
        
        # Waypoint completion: Switch to next waypoint when close enough
        if dist < self.waypoint_radius:
            if not is_final_waypoint:
                # Intermediate waypoints: Switch immediately, keep moving!
                self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
                self.waypoints_reached += 1
            else:
                # Final waypoint: Stay here and accumulate hover rewards
                # (waypoint_idx doesn't increment, so we keep rewarding this position)
                pass
        
        # No explicit survival bonus - gamma=0.997 naturally encourages longevity
        # Discount factor handles survival incentive: 0.997^500 ≈ 22% (future rewards still matter)
        
        self.previous_pos = pos.copy()
        self.previous_vel = vel.copy()
        
        # Final safety clamp: prevent catastrophic reward explosions
        # Normal operation: reward in [-50, +350] range
        # Allow some flexibility but prevent numerical disasters
        reward = np.clip(reward, -100.0, 500.0)
        
        return reward
    
    def _computeTerminated(self):
        """Check if episode should terminate (crash or all waypoints reached)."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        roll = state[7]
        pitch = state[8]
        
        # Ground crash
        if pos[2] < 0.1:
            self._crash_type = 'ground'
            return True
        
        # Upside-down crash: Terminate if drone flips over
        # Prevents "hucking" strategy where drone flips to throw itself at target
        # Real quadcopters can't fly inverted (without special hardware)
        if abs(roll) > np.pi/2 or abs(pitch) > np.pi/2:
            self._crash_type = 'inverted'
            return True
        
        # Obstacle collision - use PyBullet contact detection directly
        if self.enable_obstacles:
            import pybullet as p
            contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0])
            # Filter out ground plane contacts (check if other body is not ground)
            for contact in contact_points:
                # contact[2] is bodyB ID
                # Ground plane typically has ID 0, obstacles have higher IDs
                if contact[2] > 0:  # Not ground plane
                    self._crash_type = 'obstacle'
                    return True
        
        # No crash detected
        self._crash_type = None
        return False
    
    def _computeTruncated(self):
        """Check if episode should truncate (max steps)."""
        return self.current_step >= self.max_episode_steps
    
    def _computeInfo(self):
        """Compute info dict with detailed waypoint and reward component metrics."""
        pos = self._getDroneStateVector(0)[0:3]
        current_wp = self.waypoints[self.current_waypoint_idx]
        dist = np.linalg.norm(pos - current_wp)
        
        return {
            'waypoints_reached': self.waypoints_reached,
            'waypoint_distance': dist,
            'compliance': self.compliance,
            # Detailed reward components (for visualization)
            'progress': getattr(self, '_progress', 0.0),
            'vel_forward': getattr(self, '_vel_forward', 0.0),
            'reward_forward': getattr(self, '_reward_forward', 0.0),
            'vel_lateral_mag': getattr(self, '_vel_lateral_mag', 0.0),
            'accel_magnitude': getattr(self, '_accel_magnitude', 0.0),
            'roll': abs(getattr(self, '_roll', 0.0)),
            'pitch': abs(getattr(self, '_pitch', 0.0)),
            'altitude': getattr(self, '_altitude', 0.0),
            'alignment': getattr(self, '_alignment', 0.0),
            'alignment_scale': getattr(self, '_alignment_scale', 0.0),
            'misalignment_factor': getattr(self, '_misalignment_factor', 0.0),
            'reward_vel_align': getattr(self, '_reward_vel_align', 0.0),
            'reward_lateral': getattr(self, '_reward_lateral', 0.0),
            'reward_accel': getattr(self, '_reward_accel', 0.0),
            'reward_roll': getattr(self, '_reward_roll', 0.0),
            'reward_pitch': getattr(self, '_reward_pitch', 0.0),
            'min_obstacle_dist': getattr(self, '_min_obstacle_dist', float('inf')),
            'reward_obstacle': getattr(self, '_reward_obstacle', 0.0),
            'reward_altitude': getattr(self, '_reward_altitude', 0.0),
            'altitude_error': getattr(self, '_altitude_error', 0.0),
            'speed': getattr(self, '_speed', 0.0),
            'reward_overspeed': getattr(self, '_reward_overspeed', 0.0),
            'action_change': getattr(self, '_action_change', 0.0),
            'reward_smoothness': getattr(self, '_reward_smoothness', 0.0),
            'reward_hover': getattr(self, '_reward_hover', 0.0),
            'distance_factor': getattr(self, '_distance_factor', 0.0),
            'reward_global_altitude': getattr(self, '_reward_global_altitude', 0.0),
            # Velocity tracking metrics
            'target_velocity_mag': np.linalg.norm(self.target_velocity),
            'actual_velocity_mag': np.linalg.norm(self._getDroneStateVector(0)[10:13]),
            'velocity_tracking_error': self.velocity_tracking_error,
        }
    
    def _preprocessAction(self, action):
        """
        Convert normalized action to RPM commands via velocity control.
        
        CRITICAL: This must match the exact pattern from VelocityAviary and test files!
        
        Action format (from agent): [vx_dir, vy_dir, vz_dir, target_heading]
        - First 3 components: velocity DIRECTION (will be normalized)
        - 4th component: target heading angle (scaled to ±π)
        
        Process (matching VelocityAviary.py line 164):
        1. Normalize velocity direction vector (first 3 components)
        2. Compute speed_fraction from action magnitude
        3. Call PID with target_pos=current_pos (velocity-only control)
        4. Set target_vel = SPEED_LIMIT * speed_fraction * v_unit_vector
        5. Set target_yaw from action[3]
        
        Args:
            action: Normalized commands [-1, 1]^4 = [vx, vy, vz, target_heading]
            
        Returns:
            RPM array (1, 4) for motors
        """
        # Store current action for smoothness reward calculation
        self._current_action = action.copy()
        
        # Get current drone state
        state = self._getDroneStateVector(0)
        
        # Extract velocity direction and heading from action
        # action[0:3] is velocity DIRECTION (can be any magnitude in [-1,1])
        # action[3] is target heading angle in [-1, 1] → scaled to [-π, π]
        velocity_dir = action[0:3]
        target_heading = action[3] * np.pi  # Scale to radians
        
        # Normalize velocity direction (EXACTLY like VelocityAviary line 158-161)
        if np.linalg.norm(velocity_dir) != 0:
            v_unit_vector = velocity_dir / np.linalg.norm(velocity_dir)
        else:
            v_unit_vector = np.zeros(3)
        
        # Compute speed_fraction from action magnitude (how fast to go in that direction)
        # Use the magnitude of the velocity command as the speed fraction
        speed_fraction = np.clip(np.linalg.norm(velocity_dir), 0.0, 1.0)
        
        # Compute target velocity (EXACTLY like VelocityAviary line 172)
        target_vel = self.SPEED_LIMIT * speed_fraction * v_unit_vector
        
        # Store target velocity for tracking metrics
        self.target_velocity = target_vel.copy()
        
        # Compute velocity tracking error (will be used in step())
        actual_vel = state[10:13]
        self.velocity_tracking_error = np.linalg.norm(target_vel - actual_vel)
        
        # Call PID controller directly (EXACTLY like VelocityAviary line 164-172)
        # CRITICAL: target_pos = current_pos (NO position tracking, pure velocity control)
        rpm_single, _, _ = self.ctrl[0].computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=state[0:3],  # CRITICAL: Same as current position (velocity-only)
            target_rpy=np.array([0, 0, target_heading]),  # Control yaw, keep roll/pitch=0
            target_vel=target_vel  # Target velocity vector
        )
        
        # Update previous action for next step's smoothness calculation
        self.previous_action = action.copy()
        
        # Return as (1, 4) array for single drone
        return rpm_single.reshape(1, 4)
    
    def reset(self, seed=None, options=None):
        """Reset environment with new waypoint path and obstacles."""
        self.current_step = 0
        self.current_waypoint_idx = 0
        self.waypoints_reached = 0
        
        self.previous_pos = np.array([0, 0, 1.75])
        self.previous_vel = np.zeros(3)
        self.previous_action = np.zeros(4)
        self.compliance = 1.0
        self._crash_type = None
        
        # Reset velocity tracking metrics
        self.target_velocity = np.zeros(3)
        self.velocity_tracking_error = 0.0
        
        # Initialize empty waypoints so _computeObs() doesn't crash
        self.waypoints = [np.array([0.0, 0.0, 1.75]) for _ in range(5)]
        
        # Clear old obstacles BEFORE resetting PyBullet world
        if self.enable_obstacles and self.obstacle_generator is not None:
            self.obstacle_generator.clear_obstacles()
        
        # Call super().reset() to initialize PyBullet world
        obs, info = super().reset(seed=seed)
        
        # Now generate obstacles and waypoints (PyBullet world is ready)
        if self.enable_obstacles and self.obstacle_generator is not None:
            
            # Get drone spawn position
            drone_state = self._getDroneStateVector(0)
            drone_spawn_pos = drone_state[0:3]
            
            # Generate random obstacles with exclusion zone around drone
            # Keep 2.0m clearance around drone spawn point
            exclusion_zones = [(drone_spawn_pos, 2.0)]
            
            self.obstacle_generator.generate_random_obstacles(
                num_obstacles=self.num_obstacles,
                obstacle_types=['box', 'sphere', 'cylinder'],
                size_range=(0.3, 0.8),
                z_range=(0.5, 2.5),
                min_spacing=1.0,
                colorful=True,
                exclusion_zones=exclusion_zones
            )
            
            # Generate collision-free waypoints using voxel grid
            self.waypoints = []
            max_waypoint_attempts = 100
            
            for i in range(5):
                waypoint = self.voxel_grid.get_random_free_position(
                    z_range=(1.0, 2.5),
                    max_attempts=max_waypoint_attempts,
                    safety_margin=0.5  # Stay 0.5m away from obstacles
                )
                
                if waypoint is not None:
                    self.waypoints.append(waypoint)
                else:
                    # Fallback: use a safe position
                    print(f"[ActionCoordinator] Warning: Could not find free position for waypoint {i+1}")
                    self.waypoints.append(np.array([0.0, 0.0, 1.75]))
            
            if len(self.waypoints) < 5:
                print(f"[ActionCoordinator] Warning: Only generated {len(self.waypoints)}/5 waypoints")
        
        else:
            # No obstacles - generate waypoints freely
            self.waypoints = []
            for _ in range(5):
                wp = np.array([
                    np.random.uniform(-8, 8),   # 16m range
                    np.random.uniform(-8, 8),   # 16m range
                    np.random.uniform(1.0, 2.5) # Keep same vertical range
                ])
                self.waypoints.append(wp)
        
        # Curriculum: Add random initial velocity after some training
        # This helps the agent learn to handle momentum and breaks hover bias
        # Start with 30% probability, increase to 70% over time
        init_velocity_probability = np.clip(
            0.3 + (self.curriculum_timesteps / 500_000) * 0.4,  # 30% → 70% over 500k steps
            0.3, 0.7
        )
        
        if np.random.random() < init_velocity_probability:
            # Apply random initial velocity
            # Magnitude: 0.2 to 1.0 m/s (not too fast, not static)
            velocity_magnitude = np.random.uniform(0.2, 1.0)
            
            # Random direction (horizontal bias - less vertical velocity)
            velocity_direction = np.array([
                np.random.uniform(-1, 1),      # X
                np.random.uniform(-1, 1),      # Y  
                np.random.uniform(-0.3, 0.3)   # Z (smaller range - avoid ground/ceiling)
            ])
            velocity_direction = velocity_direction / np.linalg.norm(velocity_direction)
            
            initial_velocity = velocity_magnitude * velocity_direction
            
            # Apply velocity to drone using PyBullet
            import pybullet as p
            p.resetBaseVelocity(
                self.DRONE_IDS[0],
                linearVelocity=initial_velocity,
                angularVelocity=[0, 0, 0],  # No initial spin
                physicsClientId=self.CLIENT
            )
            
            # Update previous_vel so reward calculation is consistent
            self.previous_vel = initial_velocity.copy()
        
        # Recompute observation
        obs = self._computeObs()
        
        return obs, info
    
    def _perturb_current_waypoint(self):
        """
        Randomly perturb the current waypoint position.
        
        Simulates dynamic mission changes, sensor noise, or moving targets.
        Only happens after 1M timesteps, 10% of the time.
        """
        current_wp = self.waypoints[self.current_waypoint_idx]
        
        # Random perturbation: ±1.5m in X/Y, ±0.5m in Z
        perturbation = np.array([
            np.random.uniform(-1.5, 1.5),  # X offset
            np.random.uniform(-1.5, 1.5),  # Y offset
            np.random.uniform(-0.5, 0.5)   # Z offset (smaller range)
        ])
        
        # Apply perturbation
        new_wp = current_wp + perturbation
        
        # Clamp to safe bounds
        new_wp[0] = np.clip(new_wp[0], -8.0, 8.0)  # X bounds
        new_wp[1] = np.clip(new_wp[1], -8.0, 8.0)  # Y bounds
        new_wp[2] = np.clip(new_wp[2], 0.5, 2.5)   # Z bounds (safe altitude)
        
        # Update waypoint
        self.waypoints[self.current_waypoint_idx] = new_wp
        
        return perturbation

    def step(self, action):
        """
        Execute one control step.
        
        Action coordinator runs at 10 Hz with direct heading control.
        No integration needed - agent directly specifies target heading.
        """
        self.current_step += 1
        self.curriculum_timesteps += 1  # Track total timesteps for curriculum
        
        # No yaw integration needed - action[3] is already absolute heading!
        # _preprocessAction will handle scaling to [-π, +π]
        
        # Curriculum learning: Randomly perturb waypoints after 1M timesteps
        waypoint_perturbed = False
        perturbation_magnitude = 0.0
        
        if self.curriculum_timesteps >= self.perturbation_start_timesteps:
            if np.random.random() < self.perturbation_probability:
                perturbation = self._perturb_current_waypoint()
                perturbation_magnitude = np.linalg.norm(perturbation)
                waypoint_perturbed = True
        
        # Action smoothness: DISABLED - PID controller handles smooth transitions
        # Agent should be free to make rapid velocity/heading changes when needed
        action_change = action - self.previous_action
        action_change_magnitude = np.linalg.norm(action_change)
        reward_smoothness = 0.0  # Disabled
        
        # Execute action - _preprocessAction will be called by parent
        # and will use the target_yaw we just computed
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Smoothness reward disabled (no change to total reward)
        
        # Store smoothness metrics in info (for logging only)
        info['action_change'] = action_change_magnitude
        info['reward_smoothness'] = reward_smoothness
        
        # Store curriculum metrics
        info['waypoint_perturbed'] = waypoint_perturbed
        info['perturbation_magnitude'] = perturbation_magnitude
        info['curriculum_timesteps'] = self.curriculum_timesteps
        
        # CRITICAL FIX: Update velocity tracking metrics AFTER super().step()
        # _preprocessAction() was called during super().step() and updated these values
        # Now we need to add them to the info dict that will be returned
        info['target_velocity_mag'] = np.linalg.norm(self.target_velocity)
        info['actual_velocity_mag'] = np.linalg.norm(self._getDroneStateVector(0)[10:13])
        info['velocity_tracking_error'] = self.velocity_tracking_error
        
        # Crash penalty: Apply appropriate penalty based on crash type
        # Waypoint reward (+300) should dominate to ensure positive learning signal
        if terminated:
            # Different penalties for different crash types
            crash_type = getattr(self, '_crash_type', 'ground')
            
            if crash_type == 'inverted':
                crash_penalty = -100.0  # Harsh penalty for flipping over (degenerate strategy)
                info['crash_type'] = 'inverted'
            elif crash_type == 'obstacle':
                crash_penalty = -75.0  # Extra penalty for obstacle crashes
                info['crash_type'] = 'obstacle'
            else:  # 'ground' or unknown
                crash_penalty = -50.0  # Standard ground crash penalty
                info['crash_type'] = 'ground'
            
            reward += crash_penalty
            info['crashed'] = True
            info['crash_penalty'] = crash_penalty
            info['obstacle_crash'] = (crash_type == 'obstacle')
        else:
            info['crashed'] = False
            info['obstacle_crash'] = False
            info['crash_penalty'] = 0.0
            info['crash_type'] = 'none'
        
        return obs, reward, terminated, truncated, info
