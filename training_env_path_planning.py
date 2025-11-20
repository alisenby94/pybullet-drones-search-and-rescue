"""
Stage 1: High-Level Path Planning Environment

This is the UPPER LAYER of a hierarchical RL system.
It uses stereo vision and waypoint sequences to generate velocity commands
that the low-level motor controller (Stage 2) will execute.

REWARD: r_1 = δ(d_prev - d_curr) - c_v||v||₂ - c_a||a||₂ + obstacle_attenuation
    - Progress toward waypoint (with time discount)
    - Penalty for excessive speed
    - Penalty for excessive acceleration
    - Bonus for maintaining safe distance from obstacles

ARCHITECTURE: Transformer for vision attention + MLP
    - Vision Transformer processes 64x32 depth map
    - Attention mechanism focuses on relevant obstacles
    - Outputs desired velocities [vx, vy, vz, ωz]

INPUT:
    - Stereo vision depth map: 64x32 (downsampled with min pooling)
    - Current waypoint sequence: 3 target points [x, y, z]
    - Current state: position, velocity, acceleration, yaw rate
    - Previous velocity command (for smoothness)

OUTPUT:
    - Desired velocities: [vx, vy, vz, ωz] for motor controller
"""

import sys
sys.path.insert(0, '/home/illicit/Repos/pybullet-drones-search-and-rescue/simulation/gym-pybullet-drones')

import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from stereo_vision import StereoVisionSystem, StereoConfig
from depth_downsampler import DepthDownsampler
from typing import List, Tuple


class PathPlanningAviary(BaseRLAviary):
    """
    Stage 1: High-level path planning environment.
    
    Uses vision and waypoints to plan paths, outputs velocity commands
    for low-level motor controller.
    """
    
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 10,  # Lower freq for planning (10Hz vs 30Hz control)
                 gui=False,
                 record=False,
                 obs_type: ObservationType = ObservationType.KIN,
                 act_type: ActionType = ActionType.RPM,  # For now use RPM, will integrate motor controller later
                 motor_controller=None,  # Stage 2 motor controller (frozen)
                 ):
        """Initialize path planning environment."""
        
        # Velocity limits
        self.max_linear_vel = 2.0  # m/s
        self.max_angular_vel = np.pi / 4  # rad/s
        
        # Reward hyperparameters
        self.delta = 1.0  # Progress reward weight
        self.c_v = 0.1  # Velocity penalty coefficient
        self.c_a = 0.05  # Acceleration penalty coefficient
        self.obstacle_threshold = 2.0  # Distance threshold for obstacle attenuation (meters)
        self.gamma = 0.99  # Time discount factor
        
        # Waypoint configuration
        self.num_waypoints = 3  # Look ahead 3 waypoints
        self.waypoint_radius = 0.5  # Reached when within 0.5m
        self.waypoints = []  # List of [x, y, z] waypoints
        self.current_waypoint_idx = 0
        
        # Vision system
        stereo_config = StereoConfig(
            baseline=0.1,
            img_width=128,  # Start at 128x128
            img_height=128,
            fov=90.0  # Wide FOV for better awareness
        )
        self.stereo = StereoVisionSystem(stereo_config)
        self.downsampler = DepthDownsampler(target_width=64, target_height=32)
        
        # Motor controller (Stage 2) - frozen during Stage 1 training
        self.motor_controller = motor_controller
        self.use_motor_controller = motor_controller is not None
        
        # State tracking
        self.previous_position = np.zeros(3)
        self.previous_velocity = np.zeros(3)
        self.previous_distance = np.inf
        self.episode_step = 0
        self.max_steps = 500  # Shorter episodes for planning (50s @ 10Hz)
        self.episode_count = 0
        
        # Previous command (for smoothness)
        self.previous_command = np.zeros(4)  # [vx, vy, vz, ωz]
        
        # Random initial position
        if initial_xyzs is None:
            initial_xyzs = np.array([[
                np.random.uniform(-2.0, 2.0),
                np.random.uniform(-2.0, 2.0),
                np.random.uniform(1.0, 2.0)
            ]])
        
        if initial_rpys is None:
            initial_rpys = np.array([[
                0, 0, np.random.uniform(-np.pi, np.pi)
            ]])
        
        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs_type,
            act=act_type
        )
        
        print("\n" + "="*70)
        print("STAGE 1: HIGH-LEVEL PATH PLANNING")
        print("="*70)
        print(f"Planning frequency: {ctrl_freq} Hz")
        print(f"Max episode length: {self.max_steps} steps")
        print(f"\nREWARD: r_1 = δ(d_prev - d_curr) - c_v||v||₂ - c_a||a||₂ + obstacle_bonus")
        print(f"\nHyperparameters:")
        print(f"  δ (progress weight) = {self.delta}")
        print(f"  c_v (velocity penalty) = {self.c_v}")
        print(f"  c_a (acceleration penalty) = {self.c_a}")
        print(f"  obstacle_threshold = {self.obstacle_threshold}m")
        print(f"  γ (time discount) = {self.gamma}")
        print(f"\nVision:")
        print(f"  Input: 128x128 stereo depth")
        print(f"  Downsampled: 64x32 (min pooling)")
        print(f"  FOV: {stereo_config.fov}°")
        print(f"\nWaypoints:")
        print(f"  Lookahead: {self.num_waypoints} waypoints")
        print(f"  Reached radius: {self.waypoint_radius}m")
        print(f"\nMotor Controller:")
        print(f"  Frozen Stage 2: {'Yes' if self.use_motor_controller else 'No (direct control)'}")
        print("="*70 + "\n")
    
    def _actionSpace(self):
        """
        Action: Desired velocities [-1, 1]^4
        
        These will be scaled to actual velocity limits and sent to motor controller.
        """
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),  # [vx, vy, vz, ωz]
            dtype=np.float32
        )
    
    def _preprocessAction(self, action):
        """
        Convert velocity commands to RPM commands.
        
        For now, this is a placeholder. Later we'll integrate Stage 2 motor controller.
        
        Args:
            action: [-1, 1]^4 normalized velocities
            
        Returns:
            (1, 4) RPM commands
        """
        # For now, just use RPM offset from hover (like motor control env)
        max_offset = 0.3 * self.HOVER_RPM
        offset = np.array(action).reshape(4) * max_offset
        rpm = (self.HOVER_RPM + offset).reshape(1, 4)
        rpm = np.clip(rpm, 0, self.MAX_RPM)
        return rpm
    
    def _observationSpace(self):
        """
        Observation space:
        - Depth map: 64x32 = 2048 values
        - Current position: [x, y, z] (3)
        - Current velocity: [vx, vy, vz] (3)
        - Current acceleration: [ax, ay, az] (3)
        - Current yaw rate: [ωz] (1)
        - Waypoint 1 (relative): [dx, dy, dz] (3)
        - Waypoint 2 (relative): [dx, dy, dz] (3)
        - Waypoint 3 (relative): [dx, dy, dz] (3)
        - Previous command: [vx, vy, vz, ωz] (4)
        
        Total: 2048 + 3 + 3 + 3 + 1 + 3 + 3 + 3 + 4 = 2071 dimensions
        
        Transformer will process the depth map separately.
        """
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2071,),
            dtype=np.float32
        )
    
    def _generate_waypoints(self):
        """Generate random waypoints for the episode."""
        self.waypoints = []
        current_pos = self.INIT_XYZS[0].copy()
        
        for i in range(self.num_waypoints):
            # Generate next waypoint 3-7m away in random direction
            direction = np.random.randn(3)
            direction[2] = np.abs(direction[2])  # Keep z positive (stay airborne)
            direction = direction / np.linalg.norm(direction)
            
            distance = np.random.uniform(3.0, 7.0)
            waypoint = current_pos + direction * distance
            
            # Clamp to arena bounds
            waypoint[0] = np.clip(waypoint[0], -4.0, 4.0)
            waypoint[1] = np.clip(waypoint[1], -4.0, 4.0)
            waypoint[2] = np.clip(waypoint[2], 1.0, 4.0)
            
            self.waypoints.append(waypoint)
            current_pos = waypoint
        
        self.current_waypoint_idx = 0
    
    def _get_current_waypoint(self) -> np.ndarray:
        """Get current target waypoint."""
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        else:
            # If all waypoints reached, return last one
            return self.waypoints[-1]
    
    def _check_waypoint_reached(self, position: np.ndarray) -> bool:
        """Check if current waypoint is reached."""
        if self.current_waypoint_idx >= len(self.waypoints):
            return False
        
        waypoint = self.waypoints[self.current_waypoint_idx]
        distance = np.linalg.norm(position - waypoint)
        
        if distance < self.waypoint_radius:
            self.current_waypoint_idx += 1
            return True
        return False
    
    def _get_depth_map(self) -> np.ndarray:
        """
        Capture and process depth map.
        
        Returns:
            (32, 64) downsampled depth map
        """
        pos = self._getDroneStateVector(0)[0:3]
        quat = self._getDroneStateVector(0)[3:7]
        
        # Get high-res depth from stereo vision
        depth_highres, _ = self.stereo.get_depth_map(
            pos, quat, self.CLIENT, method='pybullet'
        )
        
        # Downsample with min pooling
        depth_lowres = self.downsampler.downsample(depth_highres)
        
        return depth_lowres
    
    def _compute_obstacle_bonus(self, depth_map: np.ndarray) -> float:
        """
        Compute obstacle attenuation bonus.
        
        Rewards staying away from obstacles.
        Uses exponential decay: closer = more penalty.
        
        Args:
            depth_map: (32, 64) depth map
            
        Returns:
            Bonus value (negative if too close)
        """
        # Find minimum distance (closest obstacle)
        min_distance = np.min(depth_map)
        
        if min_distance > self.obstacle_threshold:
            # Safe distance - no penalty/bonus
            return 0.0
        else:
            # Too close - exponential penalty
            # penalty = exp(-k * distance) - 1, where k=2
            penalty = np.exp(-2.0 * min_distance) - 1.0
            return penalty
    
    def _computeObs(self):
        """Compute current observation."""
        # Get state
        pos = self._getDroneStateVector(0)[0:3]
        vel = self._getDroneStateVector(0)[10:13]
        ang_vel = self._getDroneStateVector(0)[13:16]
        
        # Compute acceleration (finite difference)
        acceleration = (vel - self.previous_velocity) / (1.0 / self.CTRL_FREQ)
        
        # Get depth map
        depth_map = self._get_depth_map()
        depth_flat = depth_map.flatten()  # 64*32 = 2048 values
        
        # Get relative waypoint positions
        waypoint_vectors = []
        for i in range(self.num_waypoints):
            if self.current_waypoint_idx + i < len(self.waypoints):
                wp = self.waypoints[self.current_waypoint_idx + i]
                relative = wp - pos
            else:
                # Pad with zeros if no more waypoints
                relative = np.zeros(3)
            waypoint_vectors.append(relative)
        
        waypoints_flat = np.concatenate(waypoint_vectors)  # 9 values
        
        # Construct observation
        obs = np.concatenate([
            depth_flat,           # 2048
            pos,                  # 3
            vel,                  # 3
            acceleration,         # 3
            [ang_vel[2]],         # 1 (yaw rate only)
            waypoints_flat,       # 9
            self.previous_command # 4
        ])
        
        return obs.astype(np.float32)
    
    def _computeReward(self):
        """
        Compute reward using the proposed formula:
        r_1 = δ(d_prev - d_curr) - c_v||v||₂ - c_a||a||₂ + obstacle_bonus
        
        Returns:
            Scalar reward
        """
        pos = self._getDroneStateVector(0)[0:3]
        vel = self._getDroneStateVector(0)[10:13]
        
        # Compute acceleration
        acceleration = (vel - self.previous_velocity) / (1.0 / self.CTRL_FREQ)
        
        # 1. Progress toward waypoint
        current_waypoint = self._get_current_waypoint()
        current_distance = np.linalg.norm(pos - current_waypoint)
        progress = self.delta * (self.previous_distance - current_distance)
        
        # 2. Velocity penalty (encourage slower, controlled flight)
        velocity_penalty = self.c_v * np.linalg.norm(vel)
        
        # 3. Acceleration penalty (encourage smooth flight)
        acceleration_penalty = self.c_a * np.linalg.norm(acceleration)
        
        # 4. Obstacle bonus/penalty
        depth_map = self._get_depth_map()
        obstacle_bonus = self._compute_obstacle_bonus(depth_map)
        
        # Total reward
        reward = progress - velocity_penalty - acceleration_penalty + obstacle_bonus
        
        # Apply time discount to encourage faster completion
        # reward *= (self.gamma ** self.episode_step)
        
        return reward
    
    def _computeTerminated(self):
        """
        Terminate if:
        - All waypoints reached
        - Crashed
        - Out of bounds
        """
        pos = self._getDroneStateVector(0)[0:3]
        rpy = self._getDroneStateVector(0)[7:10]
        
        # Success: all waypoints reached
        if self.current_waypoint_idx >= len(self.waypoints):
            return True
        
        # Crash: ground contact
        if pos[2] < 0.1:
            return True
        
        # Crash: severe tilt
        if abs(rpy[0]) > np.pi/3 or abs(rpy[1]) > np.pi/3:
            return True
        
        # Out of bounds
        if abs(pos[0]) > 5.0 or abs(pos[1]) > 5.0 or pos[2] > 5.0:
            return True
        
        return False
    
    def _computeTruncated(self):
        """Check if episode should truncate (max steps or timeout)."""
        return self.episode_step >= self.max_steps
    
    def _computeInfo(self):
        """Compute debug info."""
        pos = self._getDroneStateVector(0)[0:3]
        current_waypoint = self._get_current_waypoint()
        distance = np.linalg.norm(pos - current_waypoint)
        
        depth_map = self._get_depth_map()
        min_obstacle_distance = np.min(depth_map)
        
        return {
            'waypoint_distance': distance,
            'waypoints_reached': self.current_waypoint_idx,
            'total_waypoints': len(self.waypoints),
            'min_obstacle_distance': min_obstacle_distance,
            'episode_step': self.episode_step
        }
    
    def reset(self, seed=None, options=None):
        """Reset environment with new waypoints."""
        self.episode_step = 0
        self.episode_count += 1
        
        # Random initial position
        self.INIT_XYZS = np.array([[
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(1.0, 2.0)
        ]])
        
        self.INIT_RPYS = np.array([[
            0, 0, np.random.uniform(-np.pi, np.pi)
        ]])
        
        # Generate waypoints
        self._generate_waypoints()
        
        # Initialize tracking
        self.previous_position = self.INIT_XYZS[0].copy()
        self.previous_velocity = np.zeros(3)
        self.previous_command = np.zeros(4)
        
        current_waypoint = self._get_current_waypoint()
        self.previous_distance = np.linalg.norm(self.INIT_XYZS[0] - current_waypoint)
        
        # Call parent reset
        obs, info = super().reset(seed=seed, options=options)
        
        obs = self._computeObs()
        info = self._computeInfo()
        
        return obs, info
    
    def step(self, action):
        """Take one step."""
        self.episode_step += 1
        
        # Store previous state
        pos = self._getDroneStateVector(0)[0:3]
        vel = self._getDroneStateVector(0)[10:13]
        self.previous_position = pos.copy()
        self.previous_velocity = vel.copy()
        
        # Update previous distance
        current_waypoint = self._get_current_waypoint()
        self.previous_distance = np.linalg.norm(pos - current_waypoint)
        
        # Scale action to velocity limits
        action_array = np.array(action).flatten()
        desired_vel = action_array[:3] * self.max_linear_vel
        desired_yaw_rate = action_array[3] * self.max_angular_vel
        
        self.previous_command = np.concatenate([desired_vel, [desired_yaw_rate]])
        
        # If using motor controller (Stage 2), convert velocities to motor commands
        if self.use_motor_controller:
            # TODO: Call frozen motor controller to get RPM commands
            # For now, use direct velocity control
            pass
        
        # Execute (for now, direct velocity control)
        # In hierarchical mode, this would call motor controller
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Check if waypoint reached
        pos_new = self._getDroneStateVector(0)[0:3]
        if self._check_waypoint_reached(pos_new):
            reward += 10.0  # Bonus for reaching waypoint
        
        # Compute actual values
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    """Test path planning environment."""
    print("Testing Path Planning Environment")
    print("="*70)
    
    env = PathPlanningAviary(gui=False)
    obs, info = env.reset()
    
    print(f"\nObservation shape: {obs.shape}")
    print(f"Waypoints: {len(env.waypoints)}")
    print(f"Starting distance: {info['waypoint_distance']:.2f}m")
    
    print("\nRunning 50 steps with random actions...")
    total_reward = 0
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 10 == 0:
            print(f"\nStep {i}:")
            print(f"  Distance to waypoint: {info['waypoint_distance']:.2f}m")
            print(f"  Waypoints reached: {info['waypoints_reached']}/{info['total_waypoints']}")
            print(f"  Min obstacle dist: {info['min_obstacle_distance']:.2f}m")
            print(f"  Reward: {reward:.3f}")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {i}")
            break
    
    print(f"\nTotal reward: {total_reward:.3f}")
    
    env.close()
    print("\n" + "="*70)
    print("Test complete!")
