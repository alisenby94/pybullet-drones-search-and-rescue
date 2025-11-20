"""
Train Stage 2: Low-Level Motor Control

Uses LSTM policy for temporal consistency in motor commands.
Learns to track velocity commands with negative quadratic reward.
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from training_env_motor_control import MotorControlAviary
import torch


def make_env():
    """Create motor control environment."""
    return MotorControlAviary(gui=False)


def train_motor_control(timesteps=500000, name="motor_control_v1"):
    """
    Train the low-level motor controller.
    
    Args:
        timesteps: Total training timesteps
        name: Model name for saving
    """
    print("\n" + "="*70)
    print("TRAINING STAGE 2: LOW-LEVEL MOTOR CONTROL")
    print("="*70)
    print(f"Model name: {name}")
    print(f"Total timesteps: {timesteps:,}")
    print(f"Algorithm: PPO with MLP policy")
    print("\nCURRICULUM LEARNING:")
    print("  Phase 1 (Episodes 0-100k):   Static commands, 30% velocity")
    print("  Phase 2 (Episodes 100k-300k): Slow changes (~6.7s), 50% velocity")
    print("  Phase 3 (Episodes 300k+):     Fast changes (~1.7s), 100% velocity")
    print("\nThis trains on RANDOM commands to prepare for Stage 1 integration.")
    print("="*70 + "\n")
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    # Create eval environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # PPO with MLP policy (includes previous action in obs for temporal consistency)
    model = PPO(
        "MlpPolicy",  # Standard feedforward policy
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Some exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=f"./logs/{name}",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"./models/{name}/checkpoints/",
        name_prefix="motor_control"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{name}/",
        log_path=f"./logs/{name}/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Train
    print("Starting training...")
    print(f"Progress will be logged to: ./logs/{name}")
    print(f"Model will be saved to: ./models/{name}")
    print("="*70 + "\n")
    
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(f"./models/{name}/final_model")
    env.save(f"./models/{name}/vec_normalize.pkl")
    
    print("\n" + "="*70)
    print("Training complete!")
    print(f"Final model saved to: ./models/{name}/final_model.zip")
    print("="*70)
    
    return model, env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 2: Motor Control")
    parser.add_argument("--timesteps", type=int, default=500000,
                        help="Total training timesteps")
    parser.add_argument("--name", type=str, default="motor_control_v1",
                        help="Model name")
    
    args = parser.parse_args()
    
    train_motor_control(timesteps=args.timesteps, name=args.name)
