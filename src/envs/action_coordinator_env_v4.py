"""
ActionCoordinatorEnv V4 - Car-Like Control (Forward/Backward + Yaw Only)

KEY DIFFERENCE FROM V3:
    - V3: 3DOF control [vx, vy, yaw_delta] - can strafe sideways
    - V4: 2DOF control [vx, yaw_delta] - car-like movement, no lateral velocity
    
RATIONALE:
    Removing lateral control forces the agent to:
    1. Learn proper turning behavior (can't just slide sideways)
    2. Plan ahead (must turn to face obstacles/waypoints)
    3. Use vision more effectively (must align heading)
    
    This is more realistic for many drone applications and simpler to learn.

ACTION SPACE (2D):
    - vx: forward/backward velocity [-1, 1] → [-2.0, +2.0] m/s
    - yaw_delta: turn left/right [-1, 1] → [-10°, +10°] per step

OBSERVATION SPACE (same as V3):
    - Depth map: 512D (32x16 downsampled stereovision)
    - Velocity (2D): [vx, vy] in body frame (vy should be ~0 now)
    - Yaw (1D): current heading angle
    - Vector to waypoint (2D): [x, y] in body frame
    Total: 512 + 2 + 1 + 2 = 517D

ENVIRONMENT:
    - Fixed altitude (1.0m) - no vertical control
    - Forward/backward movement only (no strafing)
    - Discrete yaw control (turn in place)
    - Vision-based obstacle avoidance
    - Progressive waypoint navigation
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../simulation/gym-pybullet-drones'))

import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class ActionCoordinatorEnvV4(VelocityAviary):
    """
    V4: Car-like control with forward/backward + yaw only.
    
    The drone is locked to 1.0m altitude and can only:
    - Move forward/backward (velocity command in body X-axis)
    - Rotate in place (±10° yaw adjustments)
    
    NO LATERAL MOVEMENT - forces agent to turn to navigate.
    
    This is simpler to learn and more realistic for many applications.
    """
    
    def __init__(
        self,
        gui=False,
        initial_xyzs=None,
        initial_rpys=None,
        enable_streaming=False,
        enable_obstacles=True,
        max_episode_steps=250,
    ):
        """
        Initialize V4 environment.
        
        Args:
            gui: Show PyBullet GUI
            initial_xyzs: Starting position (default: [0, 0, 1.0] - fixed altitude)
            initial_rpys: Starting orientation (default: random yaw)
            enable_streaming: Enable video streaming to VLC
            enable_obstacles: Enable obstacle avoidance training
            max_episode_steps: Maximum steps per episode
        """
        # Fixed altitude for all navigation
        self.FIXED_ALTITUDE = 1.0
        
        # Default starting position (fixed altitude)
        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, self.FIXED_ALTITUDE]])
        else:
            # Force altitude to fixed value
            initial_xyzs = np.array(initial_xyzs)
            initial_xyzs[:, 2] = self.FIXED_ALTITUDE
            
        if initial_rpys is None:
            initial_rpys = np.array([[0.0, 0.0, 0.0]])
        
        # Environment parameters
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Waypoint system (2D at fixed altitude)
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_radius = 0.5  # m
        self.waypoints_reached = 0
        
        # Velocity limits (forward/backward only)
        self.max_velocity = 2.0  # m/s
        
        # Yaw control
        self.max_yaw_delta = np.deg2rad(10.0)  # ±10° per step
        # At 48 Hz: max turn rate = 480°/s = 1.33 full rotations/second
        self.current_yaw = 0.0
        
        # Spin-out detection (prevent reward hacking via continuous spinning)
        self.spin_window_steps = 48  # 1 second at 48 Hz
        self.spin_threshold = np.deg2rad(200.0)  # 200° in 1 second
        self.yaw_history = []  # Track recent yaw changes
        
        # Cumulative spin tracking for gradual penalty
        self.cumulative_rotation = 0.0  # Total rotation in current episode
        self.spin_penalty_threshold = np.deg2rad(180.0)  # Start penalizing after 180°
        self.spin_penalty_scale = 0.001  # Small penalty per degree over threshold
        
        # State tracking
        self.previous_pos = None
        
        # Velocity tracking (forward only)
        self.commanded_velocity = np.zeros(2)  # Still track 2D for compatibility
        self.actual_velocity = np.zeros(2)
        
        # Crash tracking
        self._crash_type = None
        
        # Reward components
        self._reward_components = {}
        
        # Vision system
        self.enable_vision = True
        
        # Visualization
        self.waypoint_visual_ids = []  # Store debug visual IDs
        self.gui = gui  # Store GUI flag
        
        # Initialize parent VelocityAviary
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=48,
            gui=gui,
            record=False,
            obstacles=enable_obstacles,
            user_debug_gui=False
        )
        
        # Override speed limit
        self.SPEED_LIMIT = self.max_velocity
        
        # Initialize stereo vision system
        from src.vision.stereo_vision import StereoVisionSystem
        self.vision_system = StereoVisionSystem(
            baseline=0.06,
            focal_length=0.5,
            resolution=(64, 64),
            fov=60.0,
            near_plane=0.1,
            far_plane=10.0,
            downsample_size=(32, 16),
            enable_streaming=enable_streaming,
            stream_port=5555,
            verbose=False
        )
        
        print(f"[ActionCoordinatorV4] Initialized - Car-Like Control")
        print(f"[ActionCoordinatorV4] Fixed altitude: {self.FIXED_ALTITUDE}m")
        print(f"[ActionCoordinatorV4] Max velocity: {self.max_velocity} m/s (forward/backward only)")
        print(f"[ActionCoordinatorV4] Yaw control: ±{np.rad2deg(self.max_yaw_delta):.1f}° per step")
        print(f"[ActionCoordinatorV4] Control: 2DOF (no lateral movement)")
        print(f"[ActionCoordinatorV4] Obstacles: {'Enabled' if enable_obstacles else 'Disabled'}")
        print(f"[ActionCoordinatorV4] Control freq: {self.CTRL_FREQ} Hz")
    
    def _actionSpace(self):
        """
        Action space: [vx, yaw_delta] (forward/backward + yaw control)
        
        All components in [-1, 1]:
        - vx: forward/backward [-1, 1] → [-2.0, +2.0] m/s
        - yaw_delta: turn left/right [-1, 1] → [-10°, +10°] per step
        
        NOTE: Only 2D action space (removed vy lateral control)
        """
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),  # Only 2 actions now
            dtype=np.float32
        )
    
    def _preprocessAction(self, action):
        """
        Convert forward velocity + yaw actions to motor RPMs.
        
        The drone is locked to fixed altitude and can only move forward/backward.
        NO LATERAL MOVEMENT.
        
        Args:
            action: (NUM_DRONES, 2) array [vx, yaw_delta]
        
        Returns:
            rpm: (NUM_DRONES, 4) array of motor RPMs
        """
        rpm = np.zeros((self.NUM_DRONES, 4))
        
        for k in range(action.shape[0]):
            state = self._getDroneStateVector(k)
            quat = state[3:7]
            current_pos = state[0:3]
            
            # Forward/backward velocity command (body frame)
            # vy is ALWAYS 0 (no lateral movement)
            vx_body = action[k, 0] * self.max_velocity
            vy_body = 0.0  # LOCKED - no strafing
            
            # Clip to max velocity (should already be in range)
            vel_2d = np.array([vx_body, vy_body])
            vel_mag = np.linalg.norm(vel_2d)
            if vel_mag > self.max_velocity:
                vel_2d = vel_2d * (self.max_velocity / vel_mag)
            
            # Yaw adjustment
            yaw_delta = action[k, 1] * self.max_yaw_delta
            self.current_yaw += yaw_delta
            
            # Track yaw changes for spin-out detection
            self.yaw_history.append(abs(yaw_delta))
            if len(self.yaw_history) > self.spin_window_steps:
                self.yaw_history.pop(0)  # Remove oldest
            
            # Accumulate total rotation for gradual spin penalty
            self.cumulative_rotation += abs(yaw_delta)
            
            # Normalize yaw to [-π, +π]
            while self.current_yaw > np.pi:
                self.current_yaw -= 2 * np.pi
            while self.current_yaw < -np.pi:
                self.current_yaw += 2 * np.pi
            
            # Transform velocity from body frame to world frame
            rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
            vel_body_3d = np.array([vel_2d[0], vel_2d[1], 0.0])
            vel_world = rotation_matrix @ vel_body_3d
            
            # CRITICAL: Force vertical velocity to maintain altitude
            altitude_error = self.FIXED_ALTITUDE - current_pos[2]
            vel_world[2] = altitude_error * 2.0  # Proportional altitude correction
            
            # Call PID controller
            temp, _, _ = self.ctrl[k].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=np.array([current_pos[0], current_pos[1], self.FIXED_ALTITUDE]),
                target_rpy=np.array([0, 0, self.current_yaw]),
                target_vel=vel_world
            )
            rpm[k, :] = temp
        
        return rpm
    
    def _observationSpace(self):
        """
        Observation space: 517D (depth + velocity + yaw + waypoint vector)
        
        Same as V3 for compatibility:
        - Depth map: 512D (32x16 downsampled stereovision)
        - Velocity 2D: 2D [vx, vy] in body frame (vy should be ~0)
        - Yaw: 1D current heading angle
        - Vector to waypoint: 2D [x, y] in body frame
        
        Total: 512 + 2 + 1 + 2 = 517D
        """
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(517,),
            dtype=np.float32
        )
    
    def _computeObs(self):
        """
        Compute observation: depth map + 2D navigation state.
        
        Returns:
            obs: np.ndarray of shape (517,)
        """
        state = self._getDroneStateVector(0)
        
        pos = state[0:3]
        vel = state[10:13]
        quat = state[3:7]
        rpy = state[7:10]
        yaw = rpy[2]
        
        # 2D velocity in body frame
        rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        vel_body = rotation_matrix.T @ vel
        vel_2d = vel_body[:2]  # X (forward) and Y (lateral, should be ~0)
        
        # Current waypoint (2D only)
        target_wp = self.waypoints[self.current_waypoint_idx]
        vec_to_wp_world = target_wp - pos
        vec_to_wp_body = rotation_matrix.T @ vec_to_wp_world
        vec_to_wp_2d = vec_to_wp_body[:2]  # Only X, Y
        
        # Minimal sensor observation (5D)
        sensor_obs = np.concatenate([
            vel_2d,           # 2D - velocity (body frame)
            [yaw],            # 1D - current yaw
            vec_to_wp_2d,     # 2D - vector to waypoint (body frame)
        ])
        
        # Capture stereo vision (512D)
        try:
            vision_features = self.vision_system.get_vision_observation(
                drone_pos=pos,
                drone_rpy=rpy,
                waypoint_pos=target_wp
            )
        except Exception as e:
            vision_features = np.zeros(512)
            if self.current_step == 0:
                print(f"[V4] Vision warning: {e}")
        
        # Combine: vision first, then sensors
        obs = np.concatenate([vision_features, sensor_obs])
        
        return obs.astype(np.float32)
    
    def _addObstacles(self):
        """
        Add obstacles at fixed altitude (1.0m).
        
        All obstacles are simple shapes (cube/sphere) at the same height
        as the drone to create 2D navigation challenges.
        """
        # Cube obstacle at fixed altitude
        cube_pos = [
            np.random.uniform(-4, 4),
            np.random.uniform(-4, 4),
            self.FIXED_ALTITUDE
        ]
        p.loadURDF("cube_no_rotation.urdf",
                   cube_pos,
                   p.getQuaternionFromEuler([0, 0, 0]),
                   useFixedBase=True,
                   physicsClientId=self.CLIENT)
        
        # Sphere obstacle at fixed altitude
        sphere_pos = [
            np.random.uniform(-4, 4),
            np.random.uniform(-4, 4),
            self.FIXED_ALTITUDE
        ]
        p.loadURDF("sphere2.urdf",
                   sphere_pos,
                   p.getQuaternionFromEuler([0, 0, 0]),
                   useFixedBase=True,
                   physicsClientId=self.CLIENT)
    
    def _computeReward(self):
        """
        Car-like navigation reward (same as V3).
        
        Components:
        - Forward velocity (cameras pointed at travel direction)
        - Alignment with waypoint (encourages turning)
        - Progress toward waypoint (distance reduction)
        - Waypoint completion bonus
        - Crash penalties
        - Speed limit penalty
        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        quat = state[3:7]
        
        reward = 0.0
        self._reward_components = {}
        
        # Forward velocity reward
        rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        vel_body = rotation_matrix.T @ vel
        forward_vel = vel_body[0]  # X-axis velocity in body frame
        
        # Flat rate reward for forward motion
        if forward_vel > 0.5:
            reward_forward = 0.5
        elif forward_vel > 0.1:
            reward_forward = 0.2
        else:
            reward_forward = 0.0
        
        self._reward_components['forward_velocity'] = reward_forward
        reward += reward_forward
        
        # Speed limit penalty
        speed_2d = np.linalg.norm(vel[:2])
        reward_speeding = 0.0
        if speed_2d > 2.1:
            reward_speeding = -1.0
        self._reward_components['speeding_penalty'] = reward_speeding
        reward += reward_speeding
        
        # Spinning penalty (penalize ALL rotation to discourage spinning)
        # Penalty grows with total rotation - starts small, gets worse
        total_rotation_deg = np.rad2deg(self.cumulative_rotation)
        reward_spinning = -0.002 * (total_rotation_deg ** 1.3)
        # Cap the penalty so it doesn't completely dominate
        reward_spinning = max(reward_spinning, -3.0)
        
        # Progress toward waypoint
        current_wp = self.waypoints[self.current_waypoint_idx]
        dist_2d = np.linalg.norm(pos[:2] - current_wp[:2])
        
        # Alignment reward - MORE IMPORTANT NOW since we can't strafe
        vec_to_wp_world = current_wp - pos
        vec_to_wp_body = rotation_matrix.T @ vec_to_wp_world
        alignment_x = vec_to_wp_body[0]  # Forward component
        alignment_y = abs(vec_to_wp_body[1])  # Lateral component
        
        # Reward when waypoint is ahead and centered
        if alignment_x > 1.0 and alignment_y < 2.0:
            reward_alignment = 0.5
        elif alignment_x < 0:  # Waypoint behind
            reward_alignment = -0.5
        else:
            reward_alignment = 0.0
        
        self._reward_components['alignment'] = reward_alignment
        reward += reward_alignment
        
        # Distance penalty
        reward_distance = -0.1 * dist_2d
        self._reward_components['distance'] = reward_distance
        reward += reward_distance
        
        # Waypoint completion
        reward_waypoint = 0.0
        if dist_2d < self.waypoint_radius:
            self.current_waypoint_idx += 1
            self.waypoints_reached += 1
            reward_waypoint = 1000.0
            reward += reward_waypoint
        self._reward_components['waypoint_completion'] = reward_waypoint
        
        self.previous_pos = pos.copy()
        
        return reward
    
    def _computeTerminated(self):
        """
        Check termination: crashes or all waypoints reached.
        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        
        # Ground crash
        if pos[2] < 0.15:
            self._crash_type = 'ground'
            return True
        
        # Flipped over
        if abs(rpy[0]) > np.pi/2 or abs(rpy[1]) > np.pi/2:
            self._crash_type = 'inverted'
            return True
        
        # Obstacle collision
        if self.OBSTACLES:
            contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0])
            for contact in contact_points:
                if contact[2] > 0:
                    self._crash_type = 'obstacle'
                    return True
        
        # All waypoints reached
        if self.current_waypoint_idx >= len(self.waypoints):
            self._crash_type = None
            return True
        
        self._crash_type = None
        return False
    
    def _computeTruncated(self):
        """Check if episode should truncate (max steps)."""
        return self.current_step >= self.max_episode_steps
    
    def _updateCamera(self):
        """Update camera to follow the drone (only if GUI enabled)."""
        if not self.gui:
            return
        
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        
        # Camera parameters - follow-me style
        distance = 1.0
        pitch = -20
        
        # Camera yaw follows drone yaw
        camera_yaw = np.rad2deg(rpy[2]) + 270
        
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=camera_yaw,
            cameraPitch=pitch,
            cameraTargetPosition=pos,
            physicsClientId=self.CLIENT
        )
    
    def _visualizeWaypoints(self):
        """Draw debug spheres at waypoint locations."""
        if not self.gui:
            return
        
        # Remove old waypoint visuals
        for vis_id in self.waypoint_visual_ids:
            try:
                p.removeBody(vis_id, physicsClientId=self.CLIENT)
            except:
                pass
        self.waypoint_visual_ids = []
        
        # Draw new waypoint visuals
        for i, wp in enumerate(self.waypoints):
            # Color: green for current, gray for completed, blue for future
            if i == self.current_waypoint_idx:
                color = [0, 1, 0, 0.8]  # Green (current)
                radius = 0.3
            elif i < self.current_waypoint_idx:
                color = [0.5, 0.5, 0.5, 0.5]  # Gray (completed)
                radius = 0.2
            else:
                color = [0, 0.5, 1, 0.6]  # Blue (future)
                radius = 0.25
            
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=radius,
                rgbaColor=color,
                physicsClientId=self.CLIENT
            )
            
            body_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=wp,
                physicsClientId=self.CLIENT
            )
            
            self.waypoint_visual_ids.append(body_id)
    
    def _computeInfo(self):
        """Compute info dict with metrics."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        
        target_wp = self.waypoints[self.current_waypoint_idx]
        dist_2d = np.linalg.norm(pos[:2] - target_wp[:2])
        
        # Obstacle collision detection
        obstacle_collision = False
        if self.OBSTACLES:
            contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0])
            for contact in contact_points:
                if contact[2] > 0:
                    obstacle_collision = True
                    break
        
        return {
            'waypoint_distance': dist_2d,
            'waypoints_reached': self.waypoints_reached,
            'target_velocity_mag': np.linalg.norm(self.commanded_velocity),
            'actual_velocity_mag': np.linalg.norm(vel[:2]),
            'velocity_tracking_error': np.linalg.norm(self.commanded_velocity - vel[:2]),
            'obstacle_collision': obstacle_collision,
            'reward_forward_velocity': self._reward_components.get('forward_velocity', 0.0),
            'reward_speeding_penalty': self._reward_components.get('speeding_penalty', 0.0),
            'reward_spinning_penalty': self._reward_components.get('spinning_penalty', 0.0),
            'reward_alignment': self._reward_components.get('alignment', 0.0),
            'reward_distance': self._reward_components.get('distance', 0.0),
            'reward_waypoint_completion': self._reward_components.get('waypoint_completion', 0.0),
            'cumulative_rotation_deg': np.rad2deg(self.cumulative_rotation),
            'crash_type': self._crash_type,
            'waypoint_idx': self.current_waypoint_idx,
            'position': pos.copy(),
            'velocity': vel.copy(),
            'crash_penalty': 0.0,
        }
    
    def step(self, action):
        """Execute one timestep."""
        self.current_step += 1
        
        # Store commanded velocity (forward only, vy=0)
        self.commanded_velocity = np.array([action[0] * self.max_velocity, 0.0])
        
        # Reshape action to (1, 2) for preprocessing
        action_reshaped = action.reshape(1, 2)
        
        # Call parent step with dummy 3D action (add vy=0)
        action_3d = np.concatenate([action_reshaped, np.zeros((1, 1))], axis=1)
        obs, reward, terminated, truncated, info = super().step(action_3d)
        
        # Override with custom functions
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        
        # Update camera
        self._updateCamera()
        
        # Update waypoint visualization
        if self.gui and hasattr(self, '_last_waypoint_idx'):
            if self._last_waypoint_idx != self.current_waypoint_idx:
                self._visualizeWaypoints()
        self._last_waypoint_idx = self.current_waypoint_idx
        
        # Apply crash penalties
        if terminated and self._crash_type is not None:
            if self._crash_type == 'ground':
                reward -= 200.0
                info['crash_penalty'] = -200.0
            elif self._crash_type == 'inverted':
                reward -= 500.0
                info['crash_penalty'] = -500.0
            elif self._crash_type == 'obstacle':
                reward -= 100.0
                info['crash_penalty'] = -100.0
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset environment with new waypoints at fixed altitude."""
        self.current_step = 0
        self.current_waypoint_idx = 0
        self.waypoints_reached = 0
        self.commanded_velocity = np.zeros(2)
        self.actual_velocity = np.zeros(2)
        self._crash_type = None
        self.yaw_history = []
        self.cumulative_rotation = 0.0  # Reset cumulative rotation
        
        # Random initial yaw
        random_yaw = np.random.uniform(-np.pi, np.pi)
        self.current_yaw = random_yaw
        self.INIT_RPYS = np.array([[0.0, 0.0, random_yaw]])
        
        # Generate waypoints at fixed altitude
        self.waypoints = []
        for _ in range(5):
            wp = np.array([
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
                self.FIXED_ALTITUDE
            ])
            self.waypoints.append(wp)
        
        # Call parent reset
        obs = super().reset(seed=seed, options=options)
        
        # Get initial state
        state = self._getDroneStateVector(0)
        self.previous_pos = state[0:3].copy()
        
        # Visualize waypoints
        self._visualizeWaypoints()
        self._last_waypoint_idx = self.current_waypoint_idx
        
        # Update camera
        self._updateCamera()
        
        # Return custom observation
        obs = self._computeObs()
        info = self._computeInfo()
        
        return obs, info


if __name__ == "__main__":
    """Quick test of V4 environment."""
    print("="*80)
    print("TESTING ACTIONCOORDINATORENV V4 - Car-Like Control")
    print("="*80)
    
    env = ActionCoordinatorEnvV4(gui=True)
    
    print(f"\n✓ Environment created")
    print(f"  Action space: {env.action_space.shape} [vx, yaw_delta] (2DOF)")
    print(f"  Observation space: {env.observation_space.shape} (same as V3)")
    print(f"  Fixed altitude: {env.FIXED_ALTITUDE}m")
    print(f"  Control: Forward/backward + yaw only (no strafing)")
    
    obs, info = env.reset()
    print(f"\n✓ Environment reset")
    print(f"  Observation shape: {obs.shape}")
    print(f"  First waypoint: {env.waypoints[0]}")
    
    # Test a few steps - forward with turning
    print(f"\n✓ Testing car-like movement...")
    for i in range(100):
        # Alternate between forward and turning
        if i < 50:
            action = np.array([0.7, 0.0])  # Forward only
        else:
            action = np.array([0.5, 0.5])  # Forward + right turn
        
        obs, reward, term, trunc, info = env.step(action)
        
        if i % 20 == 0:
            print(f"  Step {i}: distance={info['waypoint_distance']:.2f}m, reward={reward:.2f}")
        
        if term or trunc:
            break
    
    env.close()
    print(f"\n✓ Test complete!")
    print("="*80)
