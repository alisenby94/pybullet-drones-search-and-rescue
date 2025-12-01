"""
ActionCoordinatorEnv V2 (V3 Auto-Yaw Architecture)

DESIGN PRINCIPLE:
    Simplified 3D velocity control with PID-managed yaw alignment.
    Agent controls ONLY velocity direction (3D), PID auto-aligns yaw.

V3 KEY CHANGES:
    1. Action space: 3D [vx, vy, vz] (removed manual yaw control)
    2. Observation: 7D minimal (velocity, yaw, vector to waypoint, vision)
    3. Auto-yaw: ALWAYS face waypoint ± 30° velocity offset
    4. Reward: Simplified to distance penalty + waypoint bonus
    5. Obstacle termination: Re-enabled
    
YAW STRATEGY:
    Base yaw = direction to waypoint (enforced alignment)
    Velocity offset = ±30° max deviation for maneuvering
    Result: "There is only ONE correct direction... MY direction."
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../simulation/gym-pybullet-drones'))

import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class ActionCoordinatorEnvV2(VelocityAviary):
    """
    Stage 1: High-level velocity control with waypoint navigation.
    
    VELOCITY CONTROL:
        Uses VelocityAviary's native implementation (no custom override).
        Action format: [vx_dir, vy_dir, vz_dir, speed_fraction]
        - First 3: velocity direction (world frame)
        - 4th: speed multiplier in [0, 1]
        
        VelocityAviary internally computes:
            target_vel = SPEED_LIMIT * speed_fraction * normalize(direction)
        
        We set SPEED_LIMIT = 2.0 m/s (increased from default 0.25 m/s).
    
    OBSERVATION:
        - Velocity (3D)
        - Angular velocity (3D)
        - Vector to waypoint (3D)
        - Distance to waypoint (1D)
        Total: 10D
    
    REWARD:
        - Progress toward waypoint
        - Waypoint completion bonus
        - Crash penalty
    """
    
    def __init__(
        self,
        gui=False,
        initial_xyzs=None,
        initial_rpys=None,
        enable_streaming=False,
        enable_obstacles=True,  # Disabled by default for early training
    ):
        """
        Initialize environment.
        
        Args:
            gui: Show PyBullet GUI
            initial_xyzs: Starting position (default: [0, 0, 1.5])
            initial_rpys: Starting orientation (default: [0, 0, 0])
            enable_streaming: Enable video streaming to VLC (for debugging)
        """
        # Default starting position
        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 1.5]])
        if initial_rpys is None:
            initial_rpys = np.array([[0.0, 0.0, 0.0]])
        
        # Environment parameters
        self.max_episode_steps = 250
        self.current_step = 0
        
        # Waypoint system
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_radius = 0.5  # m (reduced from 1.0m for tighter precision)
        self.waypoints_reached = 0
        
        # State tracking
        self.previous_pos = None
        self.previous_vel = None
        
        # Velocity tracking (for logging/debugging)
        self.commanded_velocity = np.zeros(3)
        self.actual_velocity = np.zeros(3)
        
        # Smooth motion state (body frame) 
        self.current_velocity_target = np.zeros(3)  # Smoothed velocity in body frame
        self.current_yaw_target = 0.0  # Smoothed yaw
        # Conservative acceleration limits for stability
        # At 48 Hz: 0.042 m/s per step → reach 2 m/s in ~48 steps (1.0s)
        #           1.875°/step → 180° rotation in ~96 steps (2.0s)
        self.max_linear_acceleration = 2.0  # m/s² (stable)
        self.max_angular_acceleration = np.pi / 2.0  # rad/s² (half speed - prevents PID destabilization)
        
        # Crash tracking
        self._crash_type = None  # 'ground', 'obstacle', 'inverted', or None
        
        # Reward component tracking (for TensorBoard analysis)
        self._reward_components = {}
        
        # Vision system - ALWAYS ENABLED for V2
        self.enable_vision = True
        
        # Initialize parent VelocityAviary
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=48,  # PID control frequency (stable flight control)
            gui=gui,
            record=False,
            obstacles=enable_obstacles,  # Configurable for curriculum training
            user_debug_gui=False
        )
        
        # CRITICAL: Increase SPEED_LIMIT to support 2 m/s velocities
        # Default is only ~0.25 m/s which is too slow
        self.SPEED_LIMIT = 2.0  # m/s
        
        # Initialize stereo vision system (ALWAYS enabled in V2)
        from src.vision.stereo_vision import StereoVisionSystem
        self.vision_system = StereoVisionSystem(
            baseline=0.06,  # 6cm stereo baseline (CF2X width)
            focal_length=0.5,
            resolution=(64, 64),  # Reduced from 128x128 for faster rendering
            fov=60.0,
            near_plane=0.1,
            far_plane=10.0,
            downsample_size=(32, 16),  # Match training downsampling
            enable_streaming=enable_streaming,  # Pass through streaming parameter
            stream_port=5555,  # Default HTTP streaming port for VLC
            verbose=False  # Disable debug prints during training
        )
        
        # Vision downsampling for training (128x128 -> 32x16 = 512D)
        self.vision_downsample_size = (32, 16)
        
        print(f"[ActionCoordinatorV2] Initialized")
        print(f"[ActionCoordinatorV2] SPEED_LIMIT: {self.SPEED_LIMIT} m/s")
        print(f"[ActionCoordinatorV2] Control freq: {self.CTRL_FREQ} Hz (PID stability)")
        print(f"[ActionCoordinatorV2] Obstacles: {'2 simple shapes (cube + sphere)' if enable_obstacles else 'DISABLED (curriculum training)'}")
        print(f"[ActionCoordinatorV2] Stereovision: 64x64 -> 32x16 downsampled (512D features)")
        print(f"[ActionCoordinatorV2] Vision config: baseline={self.vision_system.baseline}m, FOV={self.vision_system.fov}°")
        if enable_streaming:
            print(f"[ActionCoordinatorV2] HTTP streaming ENABLED on port 5555 (vlc http://localhost:5555)")
    
    def _actionSpace(self):
        """
        Action space: [vx, vy, vz] (3D velocity in body frame)
        
        ALL components in [-1, 1] for better neural network initialization.
        The PID controller will automatically align yaw to face the velocity direction.
        
        Mapped to velocity (asymmetric - prioritizes forward motion):
        - vx (forward/back) ∈ [-1, 1] → mapped to [-1.0, +1.0] m/s (PRIORITIZED)
        - vy (left/right) ∈ [-1, 1] → mapped to [-0.5, +0.5] m/s (slower)
        - vz (up/down) ∈ [-1, 1] → mapped to [-0.5, +0.5] m/s (slower)
        """
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
    
    def _preprocessAction(self, action):
        """
        Convert agent 3D velocity actions to motor RPMs.
        
        The agent outputs pure 3D velocity in BODY FRAME.
        PID automatically aligns yaw to face the velocity direction.
        
        Args:
            action: (NUM_DRONES, 3) array [vx_body, vy_body, vz_body]
                where all components are in [-1, 1]
                - vx_body: forward velocity (body frame)
                - vy_body: right velocity (body frame)  
                - vz_body: up velocity (body frame)
        
        Returns:
            rpm: (NUM_DRONES, 4) array of motor RPMs
        """
        rpm = np.zeros((self.NUM_DRONES, 4))
        
        for k in range(action.shape[0]):
            # Get current state
            state = self._getDroneStateVector(k)
            quat = state[3:7]
            
            # ================================================================
            # DIRECT VELOCITY CONTROL (body frame) - NO SMOOTHING
            # ================================================================
            # Map velocity from [-1, 1] to axis-specific speed limits
            # Forward/backward prioritized over side/vertical movement
            desired_vel_body = np.array([
                action[k, 0] * 1.0,   # X: forward/backward (±1.0 m/s) - PRIORITIZED
                action[k, 1] * 0.5,   # Y: left/right (±0.5 m/s) - slower
                action[k, 2] * 0.5    # Z: up/down (±0.5 m/s) - slower
            ])
            
            # Clip total velocity magnitude to SPEED_LIMIT
            # This prevents diagonal movement from exceeding limits
            vel_mag = np.linalg.norm(desired_vel_body)
            if vel_mag > self.SPEED_LIMIT:
                desired_vel_body = desired_vel_body * (self.SPEED_LIMIT / vel_mag)
            
            # NO ACCELERATION LIMITING - Use commanded velocity directly
            # This makes the drone more responsive to agent actions
            self.current_velocity_target = desired_vel_body
            
            dt = self.CTRL_TIMESTEP
            
            # ================================================================
            # AUTO-ALIGN YAW: Face waypoint with ±30° velocity offset
            # ================================================================
            # STRATEGY: Always center yaw on the waypoint direction,
            # then allow velocity commands to create ±30° offset for maneuvering.
            # This enforces "there is only ONE correct direction... MY direction."
            
            # Get current position and waypoint in world frame
            current_pos = state[0:3]
            vec_to_waypoint_world = self.waypoints[self.current_waypoint_idx] - current_pos
            
            # Calculate base yaw: direction to waypoint (world frame)
            base_yaw_to_waypoint = np.arctan2(vec_to_waypoint_world[1], vec_to_waypoint_world[0])
            
            # Calculate velocity direction offset (if moving)
            horizontal_vel_mag = np.linalg.norm(self.current_velocity_target[:2])
            
            if horizontal_vel_mag > 0.1:  # Only apply velocity offset if moving
                # Calculate velocity direction in body frame
                vel_yaw_body = np.arctan2(self.current_velocity_target[1], self.current_velocity_target[0])
                
                # Clamp velocity offset to ±30° (±π/6 radians)
                MAX_VEL_OFFSET = np.pi / 6  # 30 degrees
                vel_offset_clamped = np.clip(vel_yaw_body, -MAX_VEL_OFFSET, MAX_VEL_OFFSET)
                
                # Desired yaw = waypoint direction + velocity offset
                desired_yaw = base_yaw_to_waypoint + vel_offset_clamped
            else:
                # Not moving: face waypoint directly (no offset)
                desired_yaw = base_yaw_to_waypoint
            
            # Normalize to [-π, +π]
            while desired_yaw > np.pi:
                desired_yaw -= 2 * np.pi
            while desired_yaw < -np.pi:
                desired_yaw += 2 * np.pi
            
            # NO YAW SMOOTHING - Use target yaw directly
            # This makes yaw alignment instant (PID will handle smooth rotation)
            self.current_yaw_target = desired_yaw
            
            # ================================================================
            # TRANSFORM TO WORLD FRAME for PID controller
            # ================================================================
            rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
            target_vel_world = rotation_matrix @ self.current_velocity_target
            
            # ================================================================
            # ALTITUDE CLAMPING: Force PID to stay within ±1m of target
            # ================================================================
            # Get target altitude from current waypoint
            target_altitude = self.waypoints[self.current_waypoint_idx][2]
            current_altitude = state[2]
            altitude_error = current_altitude - target_altitude
            
            # Hard clamp velocity when outside or near altitude bounds
            # Use a margin to account for momentum and PID response time
            clamped_vel_world = target_vel_world.copy()
            ALTITUDE_LIMIT = 1.0  # ±1m from target
            CLAMP_MARGIN = 0.2  # Start clamping 0.2m before limit
            
            if altitude_error > ALTITUDE_LIMIT - CLAMP_MARGIN:
                # Above/near upper limit: prevent/reduce upward movement
                if altitude_error >= ALTITUDE_LIMIT:
                    # Hard clamp: no upward movement allowed
                    clamped_vel_world[2] = min(0.0, target_vel_world[2])
                else:
                    # Soft clamp: reduce upward velocity proportionally
                    scale = 1.0 - ((altitude_error - (ALTITUDE_LIMIT - CLAMP_MARGIN)) / CLAMP_MARGIN)
                    if target_vel_world[2] > 0:
                        clamped_vel_world[2] = target_vel_world[2] * scale
                        
            elif altitude_error < -(ALTITUDE_LIMIT - CLAMP_MARGIN):
                # Below/near lower limit: prevent/reduce downward movement
                if altitude_error <= -ALTITUDE_LIMIT:
                    # Hard clamp: no downward movement allowed
                    clamped_vel_world[2] = max(0.0, target_vel_world[2])
                else:
                    # Soft clamp: reduce downward velocity proportionally
                    scale = 1.0 - ((-(ALTITUDE_LIMIT - CLAMP_MARGIN) - altitude_error) / CLAMP_MARGIN)
                    if target_vel_world[2] < 0:
                        clamped_vel_world[2] = target_vel_world[2] * scale
            
            # Call PID controller with clamped vertical velocity
            temp, _, _ = self.ctrl[k].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=state[0:3],  # Stay at current position (velocity control)
                target_rpy=np.array([0, 0, self.current_yaw_target]),  # Smoothed yaw
                target_vel=clamped_vel_world  # CLAMPED vertical velocity
            )
            rpm[k, :] = temp
        
        return rpm
    
    def _observationSpace(self):
        """
        Observation space: 519D minimal sensors + vision
        
        Components (MINIMAL - let PID handle stabilization):
        - Velocity (body frame): 3D [vx, vy, vz]
        - Yaw (world frame): 1D [yaw angle]
        - Vector to waypoint (body frame): 3D [x, y, z]
        - Stereovision depth map: 512D (32x16 downsampled)
        
        Total: 7 + 512 = 519D
        
        Removed: angular velocity (PID handles), distance (redundant with ToWP)
        """
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(519,),
            dtype=np.float32
        )
    
    def _computeObs(self):
        """
        Compute minimal observation from current state + vision.
        
        Returns:
            obs: np.ndarray of shape (519,)
                - Sensors: 7D (minimal)
                - Vision: 512D (camera/body frame)
        """
        # Get state from parent
        state = self._getDroneStateVector(0)
        
        pos = state[0:3]
        vel = state[10:13]  # World frame velocity from PyBullet
        quat = state[3:7]
        rpy = state[7:10]
        yaw = rpy[2]
        
        # Convert world frame velocity to body frame
        # Rotation matrix from world to body frame
        rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        vel_body = rotation_matrix.T @ vel  # Transform world->body
        
        # Current waypoint
        target_wp = self.waypoints[self.current_waypoint_idx]
        vec_to_wp_world = target_wp - pos
        
        # Convert waypoint vector to body frame
        vec_to_wp_body = rotation_matrix.T @ vec_to_wp_world
        
        # Minimal sensor observation (7D)
        sensor_obs = np.concatenate([
            vel_body,        # 3D - current velocity (body frame: forward/right/up)
            [yaw],           # 1D - current yaw (world frame orientation)
            vec_to_wp_body,  # 3D - vector to waypoint (body frame)
        ])
        
        # Capture stereo vision (512D)
        try:
            rpy = state[7:10]  # Roll, pitch, yaw
            vision_features = self.vision_system.get_vision_observation(
                drone_pos=pos,
                drone_rpy=rpy,
                waypoint_pos=target_wp
            )
        except Exception as e:
            # Fallback if vision fails (e.g., during initialization)
            vision_features = np.zeros(512)
            if self.current_step == 0:
                print(f"[ActionCoordinatorV2] Vision warning: {e}")
        
        # Combine sensor + vision
        obs = np.concatenate([sensor_obs, vision_features])
        
        return obs.astype(np.float32)
    
    def _addObstacles(self):
        """
        Override parent to add simpler obstacles for faster rendering.
        
        PERFORMANCE OPTIMIZATION:
        BaseAviary adds 4 complex URDF meshes (samurai, duck, cube, sphere)
        which are expensive to render with stereo vision:
            - 4 complex obstacles: ~70 it/s
            - 2 simple obstacles: ~385 it/s (5.5× faster!)
        
        We use only 2 simple geometric shapes (cube + sphere) to maintain
        obstacle avoidance training while keeping rendering fast.
        
        Obstacles are randomized each episode for generalization.
        """
        # Random cube obstacle position (fixed base - no physics simulation)
        cube_pos = [
            np.random.uniform(-4, 4),  # X
            np.random.uniform(-4, 4),  # Y
            np.random.uniform(0.5, 2.0)  # Z (height)
        ]
        cube_id = p.loadURDF("cube_no_rotation.urdf",
                             cube_pos,
                             p.getQuaternionFromEuler([0, 0, 0]),
                             useFixedBase=True,  # Make static
                             physicsClientId=self.CLIENT
                             )
        
        # Random sphere obstacle position (fixed base - no physics simulation)
        sphere_pos = [
            np.random.uniform(-4, 4),  # X
            np.random.uniform(-4, 4),  # Y
            np.random.uniform(0.5, 2.0)  # Z (height)
        ]
        sphere_id = p.loadURDF("sphere2.urdf",
                               sphere_pos,
                               p.getQuaternionFromEuler([0, 0, 0]),
                               useFixedBase=True,  # Make static
                               physicsClientId=self.CLIENT
                               )
    
    def _computeReward(self):
        """
        Reward function combining multiple objectives (all scaled reasonably):
        
        1. Forward velocity (body frame): Encourages flying forward
        2. Progress toward waypoint: Rewards moving closer to target
        3. Waypoint completion: Large bonus for reaching waypoint
        4. Altitude maintenance: Encourages staying at safe altitude
        5. Crash penalties: Large penalties for ground/obstacle collisions
        
        All rewards scaled to prevent value explosion while maintaining learning signal.
        Uses COMMANDED velocity (not actual) to prevent exploiting PID overshoot.
        
        Returns:
            float: Total reward (typically in range [-100, +15] per step)
        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]  # World frame velocity
        quat = state[3:7]
        rpy = state[7:10]
        
        # Initialize reward and component tracking
        reward = 0.0
        self._reward_components = {}  # Track individual reward components for logging
        
        # ============================================================================
        # 1. FORWARD VELOCITY: Keep stereo vision pointed forward
        # ============================================================================
        # Use SMOOTHED velocity target (already in body frame from _preprocessAction)
        # This keeps the stereo cameras pointed forward, which is critical for vision
        forward_velocity_commanded = self.current_velocity_target[0]  # Body frame forward (X-axis)
        
        # ============================================================================
        # 1. DISTANCE TO WAYPOINT: Primary objective (STRONG penalty for being far)
        # ============================================================================
        current_wp = self.waypoints[self.current_waypoint_idx]
        dist = np.linalg.norm(pos - current_wp)
        
        # STRONG distance penalty to make reaching waypoints the clear priority
        # At 5m: -10.0, At 10m: -20.0
        reward_progress = -2.0 * dist
        self._reward_components['distance'] = reward_progress
        reward += reward_progress
        
        # Removed forward velocity and alignment rewards since PID now auto-aligns yaw
        self._reward_components['forward_velocity'] = 0.0
        self._reward_components['alignment'] = 0.0

        # ============================================================================
        # 3. WAYPOINT COMPLETION: Big bonus for success
        # ============================================================================
        waypoint_reached = False
        reward_waypoint = 0.0
        if dist < self.waypoint_radius:
            waypoint_reached = True
            self.current_waypoint_idx += 1
            self.waypoints_reached += 1
            # Scale: 20.0 bonus = equivalent to 10 steps of good progress
            reward_waypoint = 5000.0
            reward += reward_waypoint
        self._reward_components['waypoint_completion'] = reward_waypoint
        
        # ============================================================================
        # 4. ALTITUDE MAINTENANCE: Stay centered around target waypoint altitude
        # ============================================================================
        target_altitude = current_wp[2]  # Use waypoint's Z coordinate as target altitude
        altitude_error = abs(pos[2] - target_altitude)
        
        # Linear penalty: simple and symmetric around target altitude
        # Penalizes being too high OR too low equally
        # At 0.5m error: -0.5, at 1.0m error: -1.0, at 2.0m error: -2.0
        # Linear scaling makes it predictable and doesn't dominate other rewards
        reward_altitude = 3 * (2 - altitude_error)
        self._reward_components['altitude'] = reward_altitude
        reward += reward_altitude
        
        # ============================================================================
        # 5. OBSTACLE AVOIDANCE: Penalty for collisions (but allow recovery)
        # ============================================================================
        
        # Obstacle collision penalty (if obstacles enabled)
        # Use moderate penalty to discourage collisions but allow learning recovery
        # This is better than instant termination - agent learns avoidance gradually
        reward_obstacle = 0.0
        if self.OBSTACLES:
            contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0])
            # Filter out ground plane contacts
            for contact in contact_points:
                # contact[2] is bodyB ID, ground plane typically has ID 0
                if contact[2] > 0:  # Collision with obstacle (not ground)
                    # Moderate penalty: -50.0 (comparable to ~20-30 steps of good progress)
                    # This discourages collisions but doesn't dominate the reward signal
                    reward_obstacle = -100.0
                    reward += reward_obstacle
                    break  # Only penalize once per step
        self._reward_components['obstacle_penalty'] = reward_obstacle
        
        # ============================================================================
        # Update state tracking
        # ============================================================================
        self.previous_pos = pos.copy()
        self.previous_vel = vel.copy()
        
        return reward
    
    def _computeTerminated(self):
        """
        Check if episode should terminate (crash or all waypoints reached).
        
        Returns:
            terminated: bool
        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        
        # Ground crash (below minimum safe altitude)
        if pos[2] < 0.15:
            self._crash_type = 'ground'
            return True
        
        # Flipped over (roll or pitch too extreme)
        if abs(rpy[0]) > np.pi/2 or abs(rpy[1]) > np.pi/2:
            self._crash_type = 'inverted'
            return True
        
        # Obstacle collision - TERMINATE on contact
        if self.OBSTACLES:
            contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0])
            for contact in contact_points:
                # contact[2] is bodyB ID, ground plane typically has ID 0
                if contact[2] > 0:  # Collision with obstacle (not ground)
                    self._crash_type = 'obstacle'
                    return True
        
        # All waypoints reached (success!)
        if self.current_waypoint_idx >= len(self.waypoints):
            self._crash_type = None
            return True
        
        # No termination
        self._crash_type = None
        return False
    
    def _computeTruncated(self):
        """
        Check if episode should truncate (max steps).
        
        Returns:
            truncated: bool
        """
        return self.current_step >= self.max_episode_steps
    
    def _computeInfo(self):
        """
        Compute info dict with useful metrics for callbacks and logging.
        
        Returns:
            info: dict
        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        
        # Current waypoint
        target_wp = self.waypoints[self.current_waypoint_idx]
        dist = np.linalg.norm(target_wp - pos)
        
        # Velocity tracking (for debugging)
        self.actual_velocity = vel.copy()
        
        # Obstacle collision detection (for metrics)
        obstacle_collision = False
        if self.OBSTACLES:
            contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0])
            for contact in contact_points:
                if contact[2] > 0:  # Collision with obstacle (not ground)
                    obstacle_collision = True
                    break
        
        return {
            # Core waypoint metrics (used by WaypointMetricsCallback)
            'waypoint_distance': dist,  # Current distance to waypoint (meters)
            'waypoints_reached': self.waypoints_reached,
            
            # Velocity tracking metrics (used by WaypointMetricsCallback)
            'target_velocity_mag': np.linalg.norm(self.commanded_velocity),
            'actual_velocity_mag': np.linalg.norm(self.actual_velocity),
            'velocity_tracking_error': np.linalg.norm(self.commanded_velocity - self.actual_velocity),
            
            # Obstacle avoidance metrics
            'obstacle_collision': obstacle_collision,  # True if currently touching obstacle
            
            # Reward component breakdown (for TensorBoard analysis)
            'reward_forward_velocity': self._reward_components.get('forward_velocity', 0.0),
            'reward_distance': self._reward_components.get('distance', 0.0),
            'reward_alignment': self._reward_components.get('alignment', 0.0),
            'reward_waypoint_completion': self._reward_components.get('waypoint_completion', 0.0),
            'reward_altitude': self._reward_components.get('altitude', 0.0),
            'reward_obstacle_penalty': self._reward_components.get('obstacle_penalty', 0.0),
            
            # Crash tracking
            'crash_type': self._crash_type,  # 'ground', 'inverted', or None (obstacle removed)
            
            # Additional debugging info
            'waypoint_idx': self.current_waypoint_idx,
            'position': pos.copy(),
            'velocity': vel.copy(),
        }
    
    def step(self, action):
        """
        Execute one timestep.
        
        Args:
            action: np.ndarray of shape (3,) - [vx, vy, vz] (body frame velocity)
        
        Returns:
            obs, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Store commanded velocity for tracking
        # Action format: [vx, vy, vz] where all ∈ [-1, 1]
        self.commanded_velocity = action[0:3] * self.SPEED_LIMIT
        
        # Clip velocity magnitude to SPEED_LIMIT
        vel_mag = np.linalg.norm(self.commanded_velocity)
        if vel_mag > self.SPEED_LIMIT:
            self.commanded_velocity = self.commanded_velocity * (self.SPEED_LIMIT / vel_mag)
        
        # Call parent step - this handles all the physics and PID control
        # Action is passed directly to VelocityAviary._preprocessAction
        obs, reward, terminated, truncated, info = super().step(action.reshape(1, 3))
        
        # Override with our custom observation, reward, terminated, truncated, info
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        
        # Apply crash penalties AFTER computing reward
        # This prevents the agent from exploiting crashes as "free resets"
        if terminated and self._crash_type is not None:
            if self._crash_type == 'ground':
                reward -= 200.0
                info['crash_penalty'] = -200.0
            elif self._crash_type == 'inverted':
                reward -= 600.0
                info['crash_penalty'] = -600.0
            elif self._crash_type == 'obstacle':
                # Obstacle collision penalty (already applied in _computeReward)
                # This is just for tracking
                info['crash_penalty'] = -100.0
            else:
                info['crash_penalty'] = 0.0
        else:
            info['crash_penalty'] = 0.0

        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """
        Reset environment with new waypoints.
        
        Args:
            seed: Random seed
            options: Additional options
        
        Returns:
            obs, info
        """
        # Reset counters
        self.current_step = 0
        self.current_waypoint_idx = 0
        self.waypoints_reached = 0
        
        # Reset velocity tracking
        self.commanded_velocity = np.zeros(3)
        self.actual_velocity = np.zeros(3)
        
        # Reset smoothing state
        self.current_velocity_target = np.zeros(3)
        self.current_yaw_target = 0.0
        
        # Reset crash tracking
        self._crash_type = None
        
        # Generate random initial yaw for curriculum learning
        # This forces the agent to learn heading correction instead of just "go forward"
        random_yaw = np.random.uniform(-np.pi, np.pi)
        
        # Update INIT_RPYS with random yaw BEFORE calling parent reset
        # This ensures the parent reset uses the random orientation
        self.INIT_RPYS = np.array([[0.0, 0.0, random_yaw]])
        
        # Initialize smoothed yaw target to match starting yaw
        self.current_yaw_target = random_yaw
        
        # Generate new waypoints
        self.waypoints = []
        for _ in range(5):
            wp = np.array([
                np.random.uniform(-5, 5),    # X
                np.random.uniform(-5, 5),    # Y
                np.random.uniform(1.0, 2.5)  # Z (safe altitude)
            ])
            self.waypoints.append(wp)
        
        # Call parent reset (will use updated INIT_RPYS with random yaw)
        obs = super().reset(seed=seed, options=options)
        
        # Get initial state
        state = self._getDroneStateVector(0)
        self.previous_pos = state[0:3].copy()
        
        # Return our custom observation
        obs = self._computeObs()
        info = self._computeInfo()
        
        return obs, info


def test_env_v2():
    """Quick test of the V2 environment."""
    print("="*70)
    print("TESTING ACTIONCOORDINATORENV V2")
    print("="*70)
    
    env = ActionCoordinatorEnvV2(gui=False)
    
    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"First waypoint: {env.waypoints[0]}")
    
    print("\nRunning 100 steps with constant forward action...")
    action = np.array([0.8, 0.0, 0.0, 0.5])  # 80% forward, 50% speed
    
    for i in range(100):
        obs, reward, term, trunc, info = env.step(action)
        
        if i % 20 == 0:
            print(f"  Step {i:3d}: "
                  f"cmd_vel={info['target_velocity_mag']:.3f}, "
                  f"act_vel={info['actual_velocity_mag']:.3f}, "
                  f"error={info['velocity_tracking_error']:.3f}, "
                  f"dist={info['waypoint_distance']:.2f}")
        
        if term or trunc:
            print(f"\nEpisode ended at step {i+1}")
            break
    
    env.close()
    
    print("\n✅ V2 Environment test complete!")
    print("="*70)


if __name__ == "__main__":
    test_env_v2()
