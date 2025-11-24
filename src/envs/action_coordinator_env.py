"""
Action Coordinator Environment (Stage 1)

PURPOSE:
    Learn high-level navigation by issuing acceleration commands.
    
INPUTS:
    - Current velocity (3D)
    - Angular velocity (3D)
    - Vector to waypoint (3D)
    - Tracking compliance metric (1D)
    - Desired velocity state (4D) - integrator state
    - Stereovision depth map (512D) - optional
    
OUTPUTS:
    - Normalized acceleration commands [-1, 1]^4
    - Scaled to ±1.5 m/s² linear, ±π/4 rad/s² angular
    
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
    
    Plans acceleration commands to navigate through waypoints.
    Commands are integrated to desired velocities, then PID tracks them.
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
        
        # Action limits - acceleration commands for navigation
        # action=0 → maintain current velocity, action=±1 → max acceleration
        # Scaled for smooth, predictable control
        self.max_linear_accel = 1.5      # m/s² (1.5 m/s² - smooth, PID-friendly)
        self.max_angular_accel = np.pi / 4  # rad/s² (45 deg/s² - smooth rotation)
        self.max_linear_vel = 2.5      # m/s (velocity clipping limit)
        self.max_angular_vel = np.pi / 2  # rad/s (yaw rate clipping limit)
        
        # Waypoints
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_radius = 1.0  # m
        self.waypoints_reached = 0
        
        # State tracking
        self.desired_vel = np.zeros(4)  # Current velocity command
        self.previous_pos = np.zeros(3)
        self.previous_vel = np.zeros(3)
        self.compliance = 1.0  # Tracking quality (for monitoring)
        
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
        Action: Normalized acceleration commands [-1, 1]^4
        
        Commands: [ax, ay, az, yaw_accel] in body frame
        Integrated to desired velocities, then tracked by PID.
        
        Returns:
            Box space (4,) in range [-1, 1]
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    
    def _observationSpace(self):
        """
        Observation: Raw sensor data + integrator state for GRU to learn from
        
        Components (14D or 526D depending on vision):
            Without vision (14D):
                - Velocity (world frame): 3D       (how am I moving?)
                - Angular velocity (world frame): 3D (how am I rotating?)
                - Vector to waypoint: 3D            (where to go?)
                - Tracking compliance: 1D           (is motor responding?)
                - Desired velocity: 4D              (integrator state - critical for Markov property!)
            
            With vision (526D):
                - Same 14D sensors as above
                - Stereovision depth map: 512D (32x16 attention-weighted depth, optimized)
            
        Philosophy: Include desired_vel so agent can predict effects of acceleration commands.
        This makes the environment Markovian - agent can learn deterministic dynamics.
            
        Returns:
            Box space (14,) or (526,) depending on enable_vision
        """
        obs_size = 526 if self.enable_vision else 14
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
        
        # Construct base observation (14D - raw sensors + integrator state)
        # Philosophy: Include desired_vel so agent can predict dynamics (Markov property)
        obs = np.concatenate([
            vel,                # 3 - velocity (world frame, natural sensor output)
            ang_vel,            # 3 - angular velocity (world frame, natural sensor output)
            vec_to_waypoint,    # 3 - vector to goal (direction + distance combined)
            [self.compliance],  # 1 - motor tracking quality (external metric)
            self.desired_vel    # 4 - desired velocity (integrator state for predictability)
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
            reward += 100.0  # SUCCESS! Found waypoint
        
        # Roll/Pitch stability: Encourage level, forward-facing flight
        # This replaces velocity-based heading alignment with absolute attitude rewards
        # Decouples attitude control from instantaneous velocity direction
        roll = state[7]   # Roll angle (rotation about X-axis)
        pitch = state[8]  # Pitch angle (rotation about Y-axis, positive = nose down)
        
        # Roll stability: sech(4*roll) - 1
        # Aggressive penalty for banking left/right
        # roll=0°: 0 (perfect!), roll=±15°: -0.043, roll=±30°: -0.086, roll=±45°: -0.096
        roll_weight = 0.5
        reward_roll = (1.0 / np.cosh(4.0 * roll) - 1.0) * roll_weight
        reward += reward_roll
        
        # Pitch stability: (tanh(12*pitch*π + 5) - 3) / 2 + sech(2*pitch)
        # Asymmetric penalty: heavily punishes backward tilt (positive pitch = nose down)
        # Allows/encourages slight forward tilt (negative pitch = nose up) for forward flight
        # The sech term adds smooth penalty for any pitch deviation
        pitch_weight = 0.5
        reward_pitch = ((np.tanh(12.0 * pitch * np.pi + 5) - 3.0) / 2.0 + 
                        1.0 / np.cosh(2.0 * pitch)) * pitch_weight
        reward += reward_pitch
        
        # Altitude penalty: Encourage staying near target waypoint altitude
        # Linear penalty centered on target altitude to discourage vertical exploration
        target_altitude = current_wp[2]
        altitude_error = abs(pos[2] - target_altitude)
        altitude_weight = 0.0  # Moderate weight - don't dominate other rewards
        reward_altitude = -altitude_weight * altitude_error
        reward += reward_altitude
        
        # Store metrics for logging
        direction_to_wp = (current_wp - pos) / (dist + 1e-6)
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
        self._reward_vel_align = 0.0
        self._reward_lateral = 0.0
        self._reward_accel = 0.0
        self._roll = roll
        self._pitch = pitch
        self._reward_roll = reward_roll
        self._reward_pitch = reward_pitch
        self._altitude = altitude
        self._altitude_error = altitude_error
        self._reward_altitude = reward_altitude
        
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
        
        # Waypoint reached bonus
        if dist < self.waypoint_radius:
            reward += 200.0
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
            self.waypoints_reached += 1
        
        # No explicit survival bonus - gamma=0.997 naturally encourages longevity
        # Discount factor handles survival incentive: 0.997^500 ≈ 22% (future rewards still matter)
        
        self.previous_pos = pos.copy()
        self.previous_vel = vel.copy()
        
        return reward
    
    def _computeTerminated(self):
        """Check if episode should terminate (crash or all waypoints reached)."""
        pos = self._getDroneStateVector(0)[0:3]
        
        # Ground/ceiling crash - much higher ceiling for navigation
        if pos[2] < 0.1 or pos[2] > 10.0:
            return True
        
        # Out of bounds
        if abs(pos[0]) > 12.0 or abs(pos[1]) > 12.0:
            return True
        
        # Obstacle collision (if voxel grid available)
        if self.enable_obstacles and self.voxel_grid is not None:
            # Check if drone center is inside an occupied voxel
            if self.voxel_grid.is_occupied(pos):
                # Additional check: verify with PyBullet collision detection
                # This is more accurate than voxel grid alone
                import pybullet as p
                contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0])
                if len(contact_points) > 0:
                    # Collision with obstacle detected
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
            'vel_error': abs(getattr(self, '_vel_error', 0.0)),
            'vel_lateral_mag': getattr(self, '_vel_lateral_mag', 0.0),
            'accel_magnitude': getattr(self, '_accel_magnitude', 0.0),
            'roll': abs(getattr(self, '_roll', 0.0)),
            'pitch': abs(getattr(self, '_pitch', 0.0)),
            'altitude': getattr(self, '_altitude', 0.0),
            'reward_vel_align': getattr(self, '_reward_vel_align', 0.0),
            'reward_lateral': getattr(self, '_reward_lateral', 0.0),
            'reward_accel': getattr(self, '_reward_accel', 0.0),
            'reward_roll': getattr(self, '_reward_roll', 0.0),
            'reward_pitch': getattr(self, '_reward_pitch', 0.0),
            'min_obstacle_dist': getattr(self, '_min_obstacle_dist', float('inf')),
            'reward_obstacle': getattr(self, '_reward_obstacle', 0.0),
            'reward_altitude': getattr(self, '_reward_altitude', 0.0),
        }
    
    def _preprocessAction(self, action):
        """
        Convert normalized acceleration commands to RPM commands.
        
        Process:
        1. Scale action to physical accelerations (body frame)
        2. Integrate accelerations into desired_vel (explicit integrator)
        3. Clip desired_vel to safe limits
        4. PID tracks desired_vel → RPMs
        
        Args:
            action: Normalized acceleration commands [-1, 1]^4
            
        Returns:
            RPM array (1, 4)
        """
        # Scale normalized action to physical accelerations (body frame)
        dt = 1.0 / self.control_freq
        accel_cmd = np.array([
            action[0] * self.max_linear_accel,
            action[1] * self.max_linear_accel,
            action[2] * self.max_linear_accel,
            action[3] * self.max_angular_accel
        ])
        
        # Integrate acceleration into desired velocity (explicit, observable by agent)
        self.desired_vel += accel_cmd * dt
        
        # Clip to safe velocity limits
        self.desired_vel[:3] = np.clip(self.desired_vel[:3], -self.max_linear_vel, self.max_linear_vel)
        self.desired_vel[3] = np.clip(self.desired_vel[3], -self.max_angular_vel, self.max_angular_vel)
        
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel_world = state[10:13]
        rpy = state[7:10]
        ang_vel = state[13:16]
        yaw = rpy[2]
        
        # Transform desired velocity from body to world frame
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        desired_vel_world = np.array([
            cos_yaw * self.desired_vel[0] - sin_yaw * self.desired_vel[1],
            sin_yaw * self.desired_vel[0] + cos_yaw * self.desired_vel[1],
            self.desired_vel[2]
        ])
        
        # Compute target position for DSL controller (integrate velocity for short horizon)
        # DSL controller expects position target, so we create a virtual target ahead
        target_pos = pos + desired_vel_world * dt * 3  # Look ahead 3 timesteps
        
        # Target yaw rate integration
        target_yaw = yaw + self.desired_vel[3] * dt
        target_rpy = np.array([0, 0, target_yaw])  # DSL will compute roll/pitch
        
        # Use DSL PID controller to compute RPMs for velocity tracking
        rpm, _, _ = self.ctrl.computeControlFromState(
            control_timestep=dt,
            state=state,
            target_pos=target_pos,
            target_rpy=target_rpy,
            target_vel=desired_vel_world  # Pass velocity for better tracking
        )
        
        # Compute compliance (tracking quality)
        vel_body_actual = np.array([
            cos_yaw * vel_world[0] + sin_yaw * vel_world[1],
            -sin_yaw * vel_world[0] + cos_yaw * vel_world[1],
            vel_world[2]
        ])
        tracking_error = np.linalg.norm(vel_body_actual[:3] - self.desired_vel[:3])
        self.compliance = np.exp(-0.5 * tracking_error)
        
        return rpm.reshape(1, 4)
    
    def reset(self, seed=None, options=None):
        """Reset environment with new waypoint path and obstacles."""
        self.current_step = 0
        self.current_waypoint_idx = 0
        self.waypoints_reached = 0
        
        self.desired_vel = np.zeros(4)
        self.previous_pos = np.array([0, 0, 1.75])
        self.previous_vel = np.zeros(3)
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
    
    def step(self, action):
        """
        Execute one control step.
        
        Action coordinator runs at 10 Hz, but physics at 30 Hz,
        so we repeat the velocity command for 3 physics steps.
        """
        self.current_step += 1
        
        # Execute action for 3 substeps (10 Hz coordinator, 30 Hz physics)
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Crash penalty: Moderate - must enable long exploration episodes
        # Agent needs to stay alive to learn navigation, make crashes costly but not excessive
        if terminated:
            crash_penalty = -100.0  # Reduced from -1000 to -10 (100× less severe)
            
            # Additional penalty for obstacle collision
            pos = self._getDroneStateVector(0)[0:3]
            if self.enable_obstacles and self.voxel_grid is not None:
                if self.voxel_grid.is_occupied(pos):
                    crash_penalty = -100.0  # Double penalty for obstacle crashes!
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
