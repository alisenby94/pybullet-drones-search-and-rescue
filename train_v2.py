"""
Train ActionCoordinatorEnvV2

This script trains the V2 environment which uses VelocityAviary's native velocity control.

Key differences from V1:
- Uses VelocityAviary directly (no custom _preprocessAction override)
- Action space: [vx_dir, vy_dir, vz_dir, speed_fraction]
- Simpler, cleaner reward system focused on waypoint reaching
- No obstacles or vision initially (can add later)

Usage:
    python train_v2.py --timesteps 500000 --name v2_baseline --n-envs 4
"""

import argparse
import os
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

from src.envs.action_coordinator_env_v2 import ActionCoordinatorEnvV2
from src.arch.rl_arch import create_action_coordinator
from src.training.custom_callbacks import WaypointMetricsCallback


def train_v2(timesteps=500000, name="v2_baseline", seed_model=None, n_envs=4):
    """
    Train ActionCoordinatorEnvV2.
    
    Args:
        timesteps: Total training timesteps
        name: Experiment name for saving models
        seed_model: Path to pretrained model to resume from (optional)
        n_envs: Number of parallel environments
    """
    print(f"\n{'='*70}")
    print(f"ACTIONCOORDINATORENV V2 TRAINING")
    print(f"{'='*70}")
    print(f"Experiment name: {name}")
    print(f"Total timesteps: {timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    if seed_model:
        print(f"Resuming from: {seed_model}")
    print(f"{'='*70}\n")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/v2_training", exist_ok=True)
    
    print(f"📊 TensorBoard logs: ./logs/v2_training")
    print(f"   View with: tensorboard --logdir=./logs/v2_training")
    print()
    
    # Create parallel environments
    print(f"Creating {n_envs} parallel environments...")
    
    def make_env():
        """Factory function to create environment instances."""
        def _init():
            return ActionCoordinatorEnvV2(gui=False)
        return _init
    
    # Use SubprocVecEnv for true parallelism
    if n_envs > 1:
        env = SubprocVecEnv([make_env() for _ in range(n_envs)])
        print(f"  ✓ SubprocVecEnv with {n_envs} processes")
    else:
        env = DummyVecEnv([make_env()])
        print(f"  ✓ Single environment (DummyVecEnv)")
    
    # Wrap with VecMonitor for episode statistics
    env = VecMonitor(env, info_keywords=())
    print(f"  ✓ VecMonitor for episode statistics")
    
    # Create or load model
    print("\nCreating model...")
    
    if seed_model:
        # Remove .zip extension if present
        seed_path = seed_model[:-4] if seed_model.endswith('.zip') else seed_model
        
        if os.path.exists(seed_path + ".zip"):
            print(f"  Loading from {seed_path}.zip")
            try:
                model = RecurrentPPO.load(seed_path, env=env)
                print(f"  ✓ Loaded RecurrentPPO")
                model.tensorboard_log = "./logs/v2_training"
                
                # Reset rollout buffer for fresh data
                print(f"  Resetting rollout buffer...")
                model.rollout_buffer.reset()
            except Exception as e:
                print(f"  RecurrentPPO failed, trying PPO: {e}")
                model = PPO.load(seed_path, env=env)
                print(f"  ✓ Loaded PPO")
                model.tensorboard_log = "./logs/v2_training"
                model.rollout_buffer.reset()
        else:
            print(f"  Warning: {seed_path}.zip not found, creating new model")
            model = create_action_coordinator(env, verbose=2)
    else:
        model = create_action_coordinator(env, verbose=2)
        print(f"  ✓ Created RecurrentPPO with GRU architecture")
    
    print(f"  Tensorboard log: {model.tensorboard_log}")
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1000),
        save_path=f"./models/{name}_checkpoints/",
        name_prefix="v2_coordinator",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=0
    )
    
    waypoint_callback = WaypointMetricsCallback(verbose=1)
    
    callbacks = CallbackList([checkpoint_callback, waypoint_callback])
    
    # Train
    print(f"\n{'='*70}")
    print(f"TRAINING START")
    print(f"{'='*70}\n")
    
    should_reset = (seed_model is None)
    print(f"reset_num_timesteps: {should_reset}")
    
    trained_model = model.learn(
        total_timesteps=timesteps,
        reset_num_timesteps=should_reset,
        progress_bar=True,
        callback=callbacks,
        tb_log_name=name
    )
    
    # Validate rewards from recent episodes
    print(f"\n{'='*70}")
    print(f"REWARD VALIDATION")
    print(f"{'='*70}")
    
    if hasattr(trained_model, 'ep_info_buffer') and len(trained_model.ep_info_buffer) > 0:
        recent_episodes = list(trained_model.ep_info_buffer)
        episode_rewards = [ep['r'] for ep in recent_episodes]
        episode_lengths = [ep['l'] for ep in recent_episodes]
        
        print(f"Recent episodes analyzed: {len(episode_rewards)}")
        print(f"\nEpisode Rewards:")
        print(f"  Mean:   {sum(episode_rewards)/len(episode_rewards):+.2f}")
        print(f"  Min:    {min(episode_rewards):+.2f}")
        print(f"  Max:    {max(episode_rewards):+.2f}")
        print(f"  Median: {sorted(episode_rewards)[len(episode_rewards)//2]:+.2f}")
        
        print(f"\nEpisode Lengths:")
        print(f"  Mean:   {sum(episode_lengths)/len(episode_lengths):.1f} steps")
        print(f"  Min:    {min(episode_lengths)} steps")
        print(f"  Max:    {max(episode_lengths)} steps")
        
        # Sanity checks
        print(f"\n🔍 Reward Sanity Checks:")
        
        # Check for obviously broken rewards
        if all(r == 0 for r in episode_rewards):
            print(f"  ⚠️  WARNING: All rewards are ZERO - reward function may be broken!")
        elif all(r > 0 for r in episode_rewards):
            print(f"  ⚠️  WARNING: All rewards are POSITIVE - no negative penalties?")
        elif all(r < -10000 for r in episode_rewards):
            print(f"  ⚠️  WARNING: All rewards extremely negative - may be stuck in crash loops")
        else:
            print(f"  ✓ Reward distribution looks reasonable")
        
        # Check for learning progress
        if len(episode_rewards) >= 20:
            early_rewards = episode_rewards[:10]
            late_rewards = episode_rewards[-10:]
            improvement = (sum(late_rewards)/10) - (sum(early_rewards)/10)
            print(f"  Recent vs Early: {improvement:+.2f} (positive = learning)")
    else:
        print("  No episode data available yet")
    
    print(f"{'='*70}\n")
    
    # Save final model
    print(f"Saving final model...")
    trained_model.save(f"models/{name}_final")
    print(f"  ✓ Saved to models/{name}_final.zip")
    
    # Cleanup
    env.close()
    
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nModel saved to: models/{name}_*")
    print(f"TensorBoard logs: logs/v2_training/{name}_*/")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir=./logs/v2_training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ActionCoordinatorEnvV2")
    parser.add_argument("--timesteps", type=int, default=500000,
                        help="Total training timesteps (default: 500000)")
    parser.add_argument("--name", type=str, default="v2_baseline",
                        help="Experiment name (default: v2_baseline)")
    parser.add_argument("--seed", type=str, default=None,
                        help="Path to pretrained model to resume from")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    
    args = parser.parse_args()
    train_v2(
        timesteps=args.timesteps,
        name=args.name,
        seed_model=args.seed,
        n_envs=args.n_envs
    )
