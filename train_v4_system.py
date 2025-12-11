"""
Train ActionCoordinatorEnv V4 - Car-Like Control (Forward/Backward + Yaw Only)

KEY DIFFERENCES FROM V3:
    - V3: 3DOF control [vx, vy, yaw_delta] - can strafe sideways
    - V4: 2DOF control [vx, yaw_delta] - car-like movement, no lateral control
    
TRAINING STRATEGY:
    - Simpler action space (2D instead of 3D)
    - Forces agent to learn proper turning behavior
    - Should be easier to learn (fewer DOF)
    - More realistic for many applications

ARCHITECTURE:
    - Same as V3 (PPO + custom CNN)
    - Same observation space (517D)
    - Simplified action space (2D)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simulation/gym-pybullet-drones'))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn

from src.envs.action_coordinator_env_v4 import ActionCoordinatorEnvV4


def make_env(rank, seed=0):
    """Create a single environment instance."""
    def _init():
        env = ActionCoordinatorEnvV4(
            gui=False,
            enable_streaming=False,
            enable_obstacles=True,
            max_episode_steps=250
        )
        env = Monitor(env)
        return env
    return _init


class CustomCNN(nn.Module):
    """
    Custom CNN for processing depth maps + sensor fusion.
    
    Same architecture as V3 since observation space is identical.
    
    Architecture:
    - Depth map (512D) → CNN layers → 256D features
    - Sensors (5D: vx, vy, yaw, to_wp_x, to_wp_y) → MLP → 64D features
    - Concatenate → 320D features
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__()
        
        # Vision processing (512D depth map → 256D)
        self.vision_net = nn.Sequential(
            nn.Unflatten(1, (1, 32, 16)),
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # → 16x8
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # → 8x4
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # → 4x2
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 2, 256),
            nn.ReLU()
        )
        
        # Sensor processing (5D → 64D)
        self.sensor_net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # Combined features
        self.features_dim = features_dim
        self.combine_net = nn.Sequential(
            nn.Linear(256 + 64, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        # Split observations
        vision_obs = observations[:, :512]  # Depth map
        sensor_obs = observations[:, 512:]  # 5D sensors
        
        # Process each stream
        vision_features = self.vision_net(vision_obs)
        sensor_features = self.sensor_net(sensor_obs)
        
        # Combine
        combined = torch.cat([vision_features, sensor_features], dim=1)
        return self.combine_net(combined)


def main():
    """Train V4 system."""
    print("="*80)
    print("TRAINING ACTIONCOORDINATORENV V4 - Car-Like Control")
    print("="*80)
    
    # Training parameters
    NUM_ENVS = 8
    TOTAL_TIMESTEPS = 3_000_000
    SAVE_FREQ = 50_000
    EVAL_FREQ = 100_000
    
    # Model parameters
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 256
    N_STEPS = 2048
    N_EPOCHS = 10
    GAMMA = 0.995  # Less harsh discounting (prevent reward hacking)
    GAE_LAMBDA = 0.95
    
    print(f"\n[Training Config]")
    print(f"  Environments: {NUM_ENVS}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gamma: {GAMMA} (patient discount)")
    print(f"  Action space: 2D [vx, yaw_delta] (car-like)")
    print(f"  Save frequency: {SAVE_FREQ:,}")
    print(f"  Eval frequency: {EVAL_FREQ:,}")
    
    # Create output directories
    os.makedirs("models/v4_car_control", exist_ok=True)
    os.makedirs("logs/v4_car_control", exist_ok=True)
    
    # Create vectorized environments
    print(f"\n[Creating Environments]")
    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    eval_env = SubprocVecEnv([make_env(i) for i in range(2)])
    print(f"  ✓ {NUM_ENVS} training environments created")
    print(f"  ✓ 2 evaluation environments created")
    
    # Create custom policy
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )
    
    # Create PPO model
    print(f"\n[Creating PPO Model]")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="logs/v4_car_control"
    )
    print(f"  ✓ PPO model created")
    print(f"  ✓ Custom CNN architecture")
    print(f"  ✓ Policy: [256 features] → [128, 128] → 2D actions")
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ // NUM_ENVS,
        save_path="models/v4_car_control/checkpoints",
        name_prefix="v4_car_control"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/v4_car_control/best",
        log_path="logs/v4_car_control/eval",
        eval_freq=EVAL_FREQ // NUM_ENVS,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    # Train model
    print(f"\n[Starting Training]")
    print(f"  Monitor training with: tensorboard --logdir logs/v4_car_control")
    print("="*80)
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        
        # Save final model
        model.save("models/v4_car_control/v4_car_control_final")
        print(f"\n✓ Training complete!")
        print(f"  Final model saved to: models/v4_car_control/v4_car_control_final.zip")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠ Training interrupted by user")
        model.save("models/v4_car_control/v4_car_control_interrupted")
        print(f"  Model saved to: models/v4_car_control/v4_car_control_interrupted.zip")
    
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
