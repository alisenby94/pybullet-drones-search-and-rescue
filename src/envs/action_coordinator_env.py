"""
Action Coordinator Environment (Stage 1)

PURPOSE:
    Learn high-level navigation by directly commanding motor RPMs.
    
INPUTS:
    - Current velocity (3D)
    - Angular velocity (3D)
    - Vector to waypoint (3D)
    - Tracking compliance metric (1D)
    - Stereovision depth map (512D) - optional
    
OUTPUTS:
    - Normalized RPM commands [-1, 1]^4
    - Centered at hover RPM (16000) ±5% variation
    
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
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class ActionCoordinatorEnv(BaseRLAviary):
    """
    Stage 1: High-level action coordinator environment.
    
    Directly commands motor RPMs to navigate through waypoints.
    Simple, deterministic physics with no hidden control layers.
    """
    
    def __init__(self, gui=False, enable_vision=False, enable_streaming=False, 
                 num_obstacles=10, enable_obstacles=True):
        """
        Initialize action coordinator environment.
        
        Args:
            gui: Enable PyBullet GUI
            enable_vision: Enable stereovision system (adds 2048D to observation)
            enable_streaming: Enable video streaming to VLC (requires enable_vision=True)
            num_obstacles: Number of random obstacles to generate
            enable_obstacles: Enable obstacle generation
        """
        self.control_freq = 10  # Action coordinator runs at 10 Hz
        self.max_episode_steps = 500
        self.current_step = 0
        
        # RPM control constants
        self.HOVER_RPM = 16000  # Nominal hover RPM for CF2X
        self.RPM_VARIATION = 0.10  # ±10% variation around hover
        
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
        
        # Motor control uses PID controller (no trained model needed)
        # The DSL PID controller is initialized after super().__init__()
        
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=np.array([[0, 0, 1.75]]),  # Middle of waypoint range [1.0, 2.5]
            initial_rpys=np.zeros((1, 3)),
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=30,  # Inner loop at 30 Hz
            gui=gui,
            record=False,
            obs=ObservationType.KIN,
            act=ActionType.RPM
        )
        
        # Initialize DSL PID controller for velocity tracking
        self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        
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
        Action: Normalized RPM commands [-1, 1]^4
        
        Commands: [rpm0, rpm1, rpm2, rpm3] for each motor
        action=0 → hover RPM, action=±1 → hover ±5%
        
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
            Box space (10,) or (522,) depending on enable_vision
        """
        obs_size = 522 if self.enable_vision else 10
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
        SPARSE REWARD: Simple pass/fail with minimal shaping
        
        Philosophy: Let the agent figure out HOW to fly. We just tell it WHAT to achieve.
        - Big reward for reaching waypoints
        - Small time penalty (encourages efficiency)
        - Everything else (attitude, velocity, smoothness) emerges naturally
        
        Returns:
            float: Total reward
        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        
        # Start with survival bonus (staying alive has value)
        reward = 1.0
        
        # Progress reward: Encourage moving toward waypoint
        current_wp = self.waypoints[self.current_waypoint_idx]
        dist = np.linalg.norm(pos - current_wp)
        prev_dist = np.linalg.norm(self.previous_pos - current_wp)
        progress = prev_dist - dist
        reward_progress = 10.0 * progress  # Reward getting closer
        reward += reward_progress
        
        if dist < self.waypoint_radius:
            reward += 200.0  # SUCCESS! Found waypoint (increased to dominate crash penalty)
        
        # Roll/Pitch stability: Encourage level, forward-facing flight
        # This replaces velocity-based heading alignment with absolute attitude rewards
        # Decouples attitude control from instantaneous velocity direction
        roll = state[7]   # Roll angle (rotation about X-axis)
        pitch = state[8]  # Pitch angle (rotation about Y-axis, positive = nose down)
        
        # Roll stability: sech(4*roll) - 1
        # CRITICAL: Must learn stable flight before navigation
        # roll=0°: 0, roll=±15°: -0.43, roll=±30°: -0.86, roll=±45°: -0.96
        roll_weight = 2.0  # Increased from 1.0 - stability is CRITICAL
        reward_roll = (1.0 / np.cosh(4.0 * roll) - 1.0) * roll_weight
        reward += reward_roll
        
        # Pitch stability: (tanh(12*pitch*π + 5) - 3) / 2 + sech(2*pitch)
        # CRITICAL: Asymmetric penalty - heavily punishes backward tilt
        # Allows/encourages slight forward tilt for forward flight
        pitch_weight = 2.0  # Increased from 1.0 - stability is CRITICAL
        reward_pitch = ((np.tanh(12.0 * pitch * np.pi + 5) - 3.0) / 2.0 + 
                        1.0 / np.cosh(2.0 * pitch)) * pitch_weight
        reward += reward_pitch
        
        # Distance-scaled altitude penalty: Altitude matters MORE as we approach target
        # Far away: altitude doesn't matter much (freedom to explore vertical space)
        # Close: altitude is critical (precision landing/hovering)
        target_altitude = current_wp[2]
        altitude_error = abs(pos[2] - target_altitude)
        
        # Scale altitude importance with proximity: starts mattering around 3-4m
        # dist>5m → 0.15, dist=3m → 0.5, dist=2m → 0.75, dist=1m → 0.95, dist=0.5m → 1.0
        altitude_scale = 1.0 * np.exp(-dist / 2.5)  # FIXED: Now increases as we get closer
        altitude_weight = 1.5  # Base weight (total max ~1.5 for 1m error at close range)
        reward_altitude = -altitude_weight * altitude_scale * altitude_error
        reward += reward_altitude
        
        # Forward velocity reward: Encourage moving forward when far, hovering when close
        # Get drone's forward direction from yaw angle
        yaw = state[9]  # Yaw angle (rotation about Z-axis)
        forward_direction = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        
        # Project velocity onto forward direction
        vel_forward = np.dot(vel, forward_direction)
        
        # Distance-aware velocity reward: only reward speed when far from target
        # When close, reward approaches zero (hovering is better than speed)
        forward_weight = 1.0
        safe_forward_vel = 2.0  # m/s - reasonable cruise speed
        
        # Distance modulation: far = full reward, close = zero reward
        # Use smooth transition with tanh: dist>3m → 1.0, dist<1m → ~0.0
        distance_factor = np.tanh(dist / 2.0)  # dist=0→0, dist=2→0.76, dist=4→0.96
        
        if vel_forward > 0:
            # Saturating reward: tanh gives smooth plateau at safe speed
            # vel=0→0, vel=1→0.76, vel=2→0.96, vel=3→0.995 (plateau)
            normalized_vel = vel_forward / safe_forward_vel
            reward_forward = forward_weight * safe_forward_vel * np.tanh(normalized_vel) * distance_factor
        else:
            # Penalize backward motion linearly (also modulated by distance)
            reward_forward = forward_weight * vel_forward * distance_factor
        
        reward += reward_forward
        
        # Velocity alignment reward: MASSIVELY reward moving directly toward target
        # Start rewarding good trajectories early (3-4m out), amplify as we get closer
        # Philosophy: Speed is OK if you're aimed correctly, dangerous if you're not
        direction_to_wp = (current_wp - pos) / (dist + 1e-6)
        speed = np.linalg.norm(vel)
        
        # Velocity alignment: dot product of velocity unit vector with direction to target
        # Returns: -1 (moving away), 0 (perpendicular), +1 (perfect alignment)
        if speed > 0.1:  # Only compute alignment if moving
            vel_unit = vel / speed
            alignment = np.dot(vel_unit, direction_to_wp)
        else:
            alignment = 0.0  # Hovering = neutral
        
        # Distance-scaled alignment reward: wider range, starts mattering at 4-5m
        # Scaling factor should grow but stay comparable to other rewards
        # dist>5m → 0.5, dist=3m → 1.0, dist=2m → 1.5, dist=1m → 2.2, dist=0.5m → 2.7
        # Uses exponential decay instead of tanh for smoother, wider influence
        alignment_scale = 0.5 + 2.5 * np.exp(-dist / 3.0)  # Modest baseline, grows as we approach
        alignment_weight = 1.5  # Base weight (total max ~4.0-6.0 at close range)
        
        # Reward perfect alignment, penalize misalignment
        # alignment=+1.0, close → ~4-6, alignment=0.0 → 0, alignment=-1.0 → -4 to -6
        reward_vel_align = alignment_weight * alignment_scale * alignment
        reward += reward_vel_align
        
        # Velocity magnitude penalty when close BUT ONLY if misaligned
        # Well-aimed high speed: minimal penalty
        # Poorly-aimed high speed: moderate penalty (prevent wild overshooting)
        # Wider effective range: starts mattering at 3-4m out
        # dist>5m → 0.15, dist=3m → 0.5, dist=2m → 0.75, dist=1m → 0.95, dist=0.5m → 1.0
        proximity_factor = np.exp(-dist / 2.5)  # FIXED: Now increases as we get closer
        
        # Misalignment factor: alignment=+1.0 → 0.0 (no penalty), alignment=0.0 → 0.5, alignment=-1.0 → 1.0 (max penalty)
        misalignment_factor = (1.0 - alignment) / 2.0
        
        # Combined speed penalty: only applies when close AND misaligned
        # Aimed correctly + close + fast → minimal penalty
        # Aimed wrong + close + fast → moderate penalty (max ~1.5 at full speed)
        speed_penalty_weight = 1.0  # Reduced to match other reward magnitudes
        reward_speed_limit = -speed_penalty_weight * proximity_factor * misalignment_factor * (speed**2 / safe_forward_vel**2)
        reward += reward_speed_limit
        
        # Hover reward: ONLY on the final waypoint (encourages continuous motion through checkpoints)
        # First waypoints: pass through quickly (no hover reward)
        # Last waypoint: hover for massive sustained reward
        is_final_waypoint = (self.current_waypoint_idx == len(self.waypoints) - 1)
        
        if is_final_waypoint:
            # Final waypoint: Massive bonus for staying near target
            # Exponential reward: closer = better, with smooth falloff
            # dist=0.1m → 50.0, dist=0.5m → 44.1, dist=1.0m → 30.3, dist>2m → ~0
            hover_weight = 50.0
            reward_hover = hover_weight * np.exp(-dist**2 / (2 * self.waypoint_radius**2))
            reward += reward_hover
        else:
            # Intermediate waypoints: No hover reward
            reward_hover = 0.0
        
        # Store metrics for logging
        vel_toward_wp = np.dot(vel, direction_to_wp)
        vel_lateral = vel - vel_toward_wp * direction_to_wp
        vel_lateral_mag = np.linalg.norm(vel_lateral)
        accel = (vel - self.previous_vel) / (1.0 / self.control_freq)
        accel_magnitude = np.linalg.norm(accel)
        altitude = pos[2]
        
        # Store components for metrics
        self._progress = progress
        self._reward_progress = reward_progress
        self._vel_toward_wp = vel_toward_wp
        self._vel_lateral_mag = vel_lateral_mag
        self._accel_magnitude = accel_magnitude
        self._vel_forward = vel_forward
        self._reward_forward = reward_forward
        self._reward_hover = reward_hover
        self._distance_factor = distance_factor
        self._alignment = alignment
        self._alignment_scale = alignment_scale
        self._reward_vel_align = reward_vel_align
        self._misalignment_factor = misalignment_factor
        self._reward_lateral = 0.0
        self._reward_accel = 0.0
        self._roll = roll
        self._pitch = pitch
        self._reward_roll = reward_roll
        self._reward_pitch = reward_pitch
        self._altitude = altitude
        self._altitude_error = altitude_error
        self._altitude_scale = altitude_scale
        self._reward_altitude = reward_altitude
        self._speed = speed
        self._reward_speed_limit = reward_speed_limit
        
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
        
        return reward
    
    def _computeTerminated(self):
        """Check if episode should terminate (crash or all waypoints reached)."""
        pos = self._getDroneStateVector(0)[0:3]
        
        # Ground crash only (removed ceiling and horizontal bounds for testing)
        if pos[2] < 0.1:
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
                    return True
        
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
            'altitude_scale': getattr(self, '_altitude_scale', 0.0),
            'altitude_error': getattr(self, '_altitude_error', 0.0),
            'speed': getattr(self, '_speed', 0.0),
            'reward_speed_limit': getattr(self, '_reward_speed_limit', 0.0),
            'action_change': getattr(self, '_action_change', 0.0),
            'reward_smoothness': getattr(self, '_reward_smoothness', 0.0),
            'reward_hover': getattr(self, '_reward_hover', 0.0),
            'distance_factor': getattr(self, '_distance_factor', 0.0),
        }
    
    def _preprocessAction(self, action):
        """
        Convert normalized action to RPM commands.
        
        Simple direct mapping: action ∈ [-1, 1] → RPM = HOVER_RPM * (1 + action * 0.05)
        Also tracks action for smoothness penalty.
        
        Args:
            action: Normalized RPM commands [-1, 1]^4
            
        Returns:
            RPM array (1, 4)
        """
        # Store current action for smoothness reward calculation (before update)
        self._current_action = action.copy()
        
        # Direct RPM mapping: action=0 → hover, action=±1 → hover ±5%
        rpm = self.HOVER_RPM * (1.0 + action * self.RPM_VARIATION)
        
        # Clip to valid range (avoid negative RPMs)
        rpm = np.clip(rpm, 0, self.HOVER_RPM * 1.2)
        
        # Update previous action for next step's smoothness calculation
        self.previous_action = action.copy()
        
        return rpm.reshape(1, 4)
    
    def reset(self, seed=None, options=None):
        """Reset environment with new waypoint path and obstacles."""
        self.current_step = 0
        self.current_waypoint_idx = 0
        self.waypoints_reached = 0
        
        self.previous_pos = np.array([0, 0, 1.75])
        self.previous_vel = np.zeros(3)
        self.previous_action = np.zeros(4)
        self.compliance = 1.0
        
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
        
        # CRITICAL FIX: Initialize rotors at hover speed so drone doesn't fall
        # BaseAviary.reset() sets last_clipped_action to zeros, causing immediate drop
        # Apply hover action immediately to stabilize drone
        hover_action = np.ones(4) * self.HOVER_RPM
        for _ in range(10):  # Let drone stabilize for ~0.3 seconds
            self._physics(hover_action, 0)
        self.last_clipped_action = np.ones((self.NUM_DRONES, 4)) * self.HOVER_RPM
        
        # Recompute observation after stabilization
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
        
        Action coordinator runs at 10 Hz, but physics at 30 Hz,
        so we repeat the velocity command for 3 physics steps.
        """
        self.current_step += 1
        self.curriculum_timesteps += 1  # Track total timesteps for curriculum
        
        # Curriculum learning: Randomly perturb waypoints after 1M timesteps
        waypoint_perturbed = False
        perturbation_magnitude = 0.0
        
        if self.curriculum_timesteps >= self.perturbation_start_timesteps:
            if np.random.random() < self.perturbation_probability:
                perturbation = self._perturb_current_waypoint()
                perturbation_magnitude = np.linalg.norm(perturbation)
                waypoint_perturbed = True
        
        # Calculate smoothness penalty BEFORE stepping (so we have previous_action)
        action_change = action - self.previous_action
        action_change_magnitude = np.linalg.norm(action_change)
        smoothness_weight = 0.5
        reward_smoothness = -smoothness_weight * action_change_magnitude
        
        # Execute action for 3 substeps (10 Hz coordinator, 30 Hz physics)
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Add smoothness reward to total reward
        reward += reward_smoothness
        
        # Store smoothness metrics in info
        info['action_change'] = action_change_magnitude
        info['reward_smoothness'] = reward_smoothness
        
        # Store curriculum metrics
        info['waypoint_perturbed'] = waypoint_perturbed
        info['perturbation_magnitude'] = perturbation_magnitude
        info['curriculum_timesteps'] = self.curriculum_timesteps
        
        # Crash penalty: Moderate - must enable long exploration episodes
        # Waypoint reward (200) should dominate to ensure positive learning signal
        # Even with 2:1 crash-to-success ratio, net reward is positive
        if terminated:
            crash_penalty = -50.0  # Reduced to allow positive net learning
            
            # Additional penalty for obstacle collision
            pos = self._getDroneStateVector(0)[0:3]
            if self.enable_obstacles and self.voxel_grid is not None:
                if self.voxel_grid.is_occupied(pos):
                    crash_penalty = -75.0  # Extra penalty for obstacle crashes
                    info['obstacle_crash'] = True
                else:
                    info['obstacle_crash'] = False
            else:
                info['obstacle_crash'] = False
            
            reward += crash_penalty
            info['crashed'] = True
            info['crash_penalty'] = crash_penalty
        else:
            info['crashed'] = False
            info['obstacle_crash'] = False
            info['crash_penalty'] = 0.0
        
        return obs, reward, terminated, truncated, info
