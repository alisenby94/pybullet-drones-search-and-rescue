"""
Stage 2: Low-Level Motor Control Environment

This is the LOWER LAYER of a hierarchical RL system.
It receives desired velocity commands from a higher-level planner and
learns to generate motor commands that accurately track those velocities.

REWARD: Negative Quadratic Tracking Error
    r_t = -Σ w_i (v_i,actual - v_i,desired)²
    
    Pure compliance-based reward - no bonus for anything except accurate tracking.
    Heavily penalizes deviation from commanded velocities.

ARCHITECTURE: LSTM/RNN for temporal consistency
    - Short-term memory helps smooth control
    - Remembers previous commands to avoid oscillations

INPUT:
    - Desired velocities: [vx, vy, vz, ωz] (from high-level planner)
    - Current state: velocity, angular velocity, orientation
    - Previous motor commands (for temporal consistency)

OUTPUT:
    - Motor RPM offsets from hover [-1, 1]^4
"""

import sys
sys.path.insert(0, '/home/illicit/Repos/pybullet-drones-search-and-rescue/simulation/gym-pybullet-drones')

import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class MotorControlAviary(BaseRLAviary):
    """
    Stage 2: Low-level motor control environment.
    
    Learns to execute velocity commands accurately.
    Uses negative quadratic tracking error for tight control.
    """
    
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs_type: ObservationType = ObservationType.KIN,
                 act_type: ActionType = ActionType.RPM,
                 ):
        """Initialize motor control environment."""
        
        # Velocity limits (m/s and rad/s)
        self.max_linear_vel = 2.0  # 2 m/s max
        self.max_angular_vel = np.pi / 4  # 45°/sec max
        
        # Tracking error weights (can tune these)
        self.weight_vx = 1.0
        self.weight_vy = 1.0
        self.weight_vz = 1.0
        self.weight_omega_z = 1.0
        
        # Random initial position
        if initial_xyzs is None:
            initial_xyzs = np.array([[
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(1.0, 2.0)
            ]])
        
        # Random initial orientation
        if initial_rpys is None:
            initial_rpys = np.array([[
                np.random.uniform(-0.1, 0.1),
                np.random.uniform(-0.1, 0.1),
                np.random.uniform(-np.pi, np.pi)
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
        
        # Desired velocities (commanded by high-level planner)
        self.desired_linear_vel = np.zeros(3)  # [vx, vy, vz] in body frame
        self.desired_yaw_rate = 0.0  # ωz
        
        # Episode tracking
        self.episode_step = 0
        self.max_steps = 1000
        self.episode_count = 0
        
        # Curriculum learning parameters
        self.command_change_interval = 10000  # Will be set in reset()
        self.steps_since_command_change = 0
        
        # Previous motor commands (for observation)
        self.previous_action = np.zeros(4)
        
        print("\n" + "="*70)
        print("STAGE 2: LOW-LEVEL MOTOR CONTROL")
        print("="*70)
        print(f"Control frequency: {ctrl_freq} Hz")
        print(f"Max episode length: {self.max_steps} steps")
        print(f"\nREWARD: Negative Quadratic Tracking Error")
        print(f"  r_t = -Σ w_i (v_i,actual - v_i,desired)²")
        print(f"\nWeights:")
        print(f"  w_vx = {self.weight_vx}")
        print(f"  w_vy = {self.weight_vy}")
        print(f"  w_vz = {self.weight_vz}")
        print(f"  w_ωz = {self.weight_omega_z}")
        print("="*70 + "\n")
    
    def _actionSpace(self):
        """
        Action: RPM offsets from hover [-1, 1]^4
        
        Centered at hover for stability.
        """
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
    
    def _observationSpace(self):
        """
        Observation space (17D):
        - Current linear velocity (body frame): [vx, vy, vz] (3)
        - Current angular velocity (body frame): [ωx, ωy, ωz] (3)
        - Desired linear velocity: [vx_des, vy_des, vz_des] (3)
        - Desired yaw rate: [ωz_des] (1)
        - Orientation: [roll, pitch, yaw] (3)
        - Previous action: [a1, a2, a3, a4] (4)
        
        LSTM/RNN will use temporal sequence of these observations.
        """
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(17,),
            dtype=np.float32
        )
    
    def _preprocessAction(self, action):
        """
        Convert offset action to RPM commands.
        
        Args:
            action: Offset from hover [-1, 1]^4
            
        Returns:
            RPM commands [0, MAX_RPM]^4
        """
        max_offset = 0.3 * self.HOVER_RPM
        offset = np.array(action).reshape(4) * max_offset
        rpm = (self.HOVER_RPM + offset).reshape(1, 4)
        rpm = np.clip(rpm, 0, self.MAX_RPM)
        return rpm
    
    def _computeObs(self):
        """Compute current observation."""
        # Get drone state
        vel = self._getDroneStateVector(0)[10:13]  # World frame
        ang_vel = self._getDroneStateVector(0)[13:16]  # World frame
        rpy = self._getDroneStateVector(0)[7:10]
        
        # Transform to body frame
        yaw = rpy[2]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        vel_body = np.array([
            cos_yaw * vel[0] + sin_yaw * vel[1],
            -sin_yaw * vel[0] + cos_yaw * vel[1],
            vel[2]
        ])
        
        # Transform angular velocity to body frame
        roll, pitch = rpy[0], rpy[1]
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        
        ang_vel_body = np.array([
            ang_vel[0] * cos_pitch + ang_vel[2] * sin_pitch,
            ang_vel[1] * cos_roll - ang_vel[0] * sin_roll * sin_pitch + ang_vel[2] * cos_pitch * sin_roll,
            ang_vel[1] * sin_roll + ang_vel[0] * sin_pitch * cos_roll + ang_vel[2] * cos_pitch * cos_roll
        ])
        
        # Construct observation
        obs = np.concatenate([
            vel_body,                    # Current velocity (3)
            ang_vel_body,                # Current angular velocity (3)
            self.desired_linear_vel,     # Desired velocity (3)
            [self.desired_yaw_rate],     # Desired yaw rate (1)
            rpy,                         # Orientation (3)
            self.previous_action         # Previous action (4)
        ])
        
        return obs.astype(np.float32)
    
    def _computeReward(self):
        """
        Compute reward using NEGATIVE QUADRATIC TRACKING ERROR.
        
        r_t = -Σ w_i (v_i,actual - v_i,desired)²
        
        This heavily penalizes deviation from commanded velocities.
        No bonuses, no complex shaping - pure compliance.
        
        Returns:
            Scalar reward (always ≤ 0)
        """
        # Get current state
        vel = self._getDroneStateVector(0)[10:13]
        ang_vel = self._getDroneStateVector(0)[13:16]
        rpy = self._getDroneStateVector(0)[7:10]
        
        # Transform to body frame
        yaw = rpy[2]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        vel_body = np.array([
            cos_yaw * vel[0] + sin_yaw * vel[1],
            -sin_yaw * vel[0] + cos_yaw * vel[1],
            vel[2]
        ])
        
        # Transform angular velocity to body frame
        roll, pitch = rpy[0], rpy[1]
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        
        ang_vel_body = np.array([
            ang_vel[0] * cos_pitch + ang_vel[2] * sin_pitch,
            ang_vel[1] * cos_roll - ang_vel[0] * sin_roll * sin_pitch + ang_vel[2] * cos_pitch * sin_roll,
            ang_vel[1] * sin_roll + ang_vel[0] * sin_pitch * cos_roll + ang_vel[2] * cos_pitch * cos_roll
        ])
        
        # Compute tracking errors
        error_vx = vel_body[0] - self.desired_linear_vel[0]
        error_vy = vel_body[1] - self.desired_linear_vel[1]
        error_vz = vel_body[2] - self.desired_linear_vel[2]
        error_omega_z = ang_vel_body[2] - self.desired_yaw_rate
        
        # Negative quadratic reward
        reward = -(
            self.weight_vx * error_vx**2 +
            self.weight_vy * error_vy**2 +
            self.weight_vz * error_vz**2 +
            self.weight_omega_z * error_omega_z**2
        )
        
        return reward
    
    def _computeTerminated(self):
        """
        Terminate if drone crashes or flips.
        
        Motor control should never crash - if it does, it's failing.
        """
        pos = self._getDroneStateVector(0)[0:3]
        rpy = self._getDroneStateVector(0)[7:10]
        
        # Terminate on ground contact
        if pos[2] < 0.1:
            return True
        
        # Terminate if severely tilted (>60°)
        if abs(rpy[0]) > np.pi/3 or abs(rpy[1]) > np.pi/3:
            return True
        
        # Terminate if out of bounds
        if abs(pos[0]) > 5.0 or abs(pos[1]) > 5.0 or pos[2] > 5.0:
            return True
        
        return False
    
    def _computeTruncated(self):
        """Check if episode should truncate (max steps)."""
        return self.episode_step >= self.max_steps
    
    def _computeInfo(self):
        """Compute debug info."""
        vel = self._getDroneStateVector(0)[10:13]
        ang_vel = self._getDroneStateVector(0)[13:16]
        rpy = self._getDroneStateVector(0)[7:10]
        
        # Transform to body frame
        yaw = rpy[2]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        vel_body = np.array([
            cos_yaw * vel[0] + sin_yaw * vel[1],
            -sin_yaw * vel[0] + cos_yaw * vel[1],
            vel[2]
        ])
        
        roll, pitch = rpy[0], rpy[1]
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        
        ang_vel_body = np.array([
            ang_vel[0] * cos_pitch + ang_vel[2] * sin_pitch,
            ang_vel[1] * cos_roll - ang_vel[0] * sin_roll * sin_pitch + ang_vel[2] * cos_pitch * sin_roll,
            ang_vel[1] * sin_roll + ang_vel[0] * sin_pitch * cos_roll + ang_vel[2] * cos_pitch * cos_roll
        ])
        
        tracking_error = np.linalg.norm(vel_body - self.desired_linear_vel)
        yaw_error = abs(ang_vel_body[2] - self.desired_yaw_rate)
        
        return {
            'tracking_error': tracking_error,
            'yaw_error': yaw_error,
            'actual_vel': vel_body.copy(),
            'desired_vel': self.desired_linear_vel.copy(),
            'episode_step': self.episode_step
        }
    
    def reset(self, seed=None, options=None):
        """Reset environment with random velocity command."""
        self.episode_step = 0
        self.episode_count += 1
        self.previous_action = np.zeros(4)
        
        # CURRICULUM LEARNING: Gradually increase difficulty
        # Phase 1 (0-100k): Static commands (easy - learn basic control)
        # Phase 2 (100k-300k): Slow changes (moderate - learn adaptation)  
        # Phase 3 (300k+): Fast changes (hard - learn quick response)
        
        if self.episode_count < 20000:
            # Phase 0 (0-20k): HOVER ONLY - learn basic stability
            self.command_change_interval = 10000
            velocity_scale = 0.05  # Almost zero velocity (just tiny movements)
        elif self.episode_count < 100000:
            # Phase 1 (20k-100k): Static small commands
            self.command_change_interval = 10000  # Essentially static
            velocity_scale = 0.2  # Very conservative (20% of max = 0.4 m/s)
        elif self.episode_count < 300000:
            # Phase 2 (100k-300k): Slow changes
            self.command_change_interval = 200
            velocity_scale = 0.5  # Moderate (50% of max = 1.0 m/s)
        else:
            # Phase 3 (300k+): Fast changes
            self.command_change_interval = 50
            velocity_scale = 1.0  # Full range (100% of max = 2.0 m/s)
        
        # Generate random velocity command
        # This simulates what the high-level planner would send
        max_vel = self.max_linear_vel * velocity_scale
        self.desired_linear_vel = np.array([
            np.random.uniform(-max_vel, max_vel),
            np.random.uniform(-max_vel, max_vel),
            np.random.uniform(-max_vel, max_vel)
        ])
        
        # Clamp magnitude
        magnitude = np.linalg.norm(self.desired_linear_vel)
        if magnitude > max_vel:
            self.desired_linear_vel *= (max_vel / magnitude)
        
        self.desired_yaw_rate = np.random.uniform(
            -self.max_angular_vel * velocity_scale,
            self.max_angular_vel * velocity_scale
        )
        
        # Initialize command change counter
        self.steps_since_command_change = 0
        
        # Start from stable hover position (reduced randomness for early training)
        self.INIT_XYZS = np.array([[
            np.random.uniform(-0.2, 0.2),  # Smaller x/y range
            np.random.uniform(-0.2, 0.2),
            1.5  # Fixed height for stability
        ]])
        
        # Start nearly level (easier for motor control)
        self.INIT_RPYS = np.array([[
            np.random.uniform(-0.02, 0.02),  # Almost level
            np.random.uniform(-0.02, 0.02),
            np.random.uniform(-np.pi, np.pi)  # Yaw can vary
        ]])
        
        obs, info = super().reset(seed=seed, options=options)
        obs = self._computeObs()
        info = self._computeInfo()
        
        return obs, info
    
    def step(self, action):
        """Take one step."""
        self.episode_step += 1
        self.steps_since_command_change += 1
        
        # DYNAMIC COMMAND CHANGES (for curriculum Phase 2+)
        # Change velocity command mid-episode to train adaptation
        if self.steps_since_command_change >= self.command_change_interval:
            self.steps_since_command_change = 0
            
            # Generate new velocity command
            if self.episode_count < 100000:
                velocity_scale = 0.3
            elif self.episode_count < 300000:
                velocity_scale = 0.5
            else:
                velocity_scale = 1.0
            
            max_vel = self.max_linear_vel * velocity_scale
            self.desired_linear_vel = np.array([
                np.random.uniform(-max_vel, max_vel),
                np.random.uniform(-max_vel, max_vel),
                np.random.uniform(-max_vel, max_vel)
            ])
            
            magnitude = np.linalg.norm(self.desired_linear_vel)
            if magnitude > max_vel:
                self.desired_linear_vel *= (max_vel / magnitude)
            
            self.desired_yaw_rate = np.random.uniform(
                -self.max_angular_vel * velocity_scale,
                self.max_angular_vel * velocity_scale
            )
        
        # Store action
        self.previous_action = np.array(action).flatten()
        
        # Execute
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Compute actual values
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    """Test motor control environment."""
    print("Testing Motor Control Environment")
    print("="*70)
    
    env = MotorControlAviary(gui=False)
    obs, info = env.reset()
    
    print(f"\nObservation shape: {obs.shape}")
    print(f"Desired velocity: {info['desired_vel']}")
    print(f"Desired yaw rate: {env.desired_yaw_rate:.3f} rad/s")
    
    print("\nRunning 100 steps with random actions...")
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 25 == 0:
            print(f"\nStep {i}:")
            print(f"  Tracking error: {info['tracking_error']:.3f} m/s")
            print(f"  Yaw error: {info['yaw_error']:.3f} rad/s")
            print(f"  Reward: {reward:.3f}")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {i}")
            break
    
    print(f"\nTotal reward: {total_reward:.3f}")
    print(f"Average reward per step: {total_reward/100:.3f}")
    
    env.close()
    print("\n" + "="*70)
    print("Test complete!")
