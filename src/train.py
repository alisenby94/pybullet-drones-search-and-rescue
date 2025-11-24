"""
Action Coordinator Training Script

Train the action coordinator agent. The coordinator issues velocity commands
which are executed by a PID controller (no motor agent needed).

Usage:
    python -m src.train --timesteps 2000000 --name experiment_v1
"""

import argparse
import os
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from tqdm import tqdm

# from src.envs.motor_control_env import MotorControlEnv  # Disabled - not training motor
from src.envs.action_coordinator_env import ActionCoordinatorEnv
# from src.arch.rl_arch import create_motor_controller, create_action_coordinator  # Disabled
from src.arch.rl_arch import create_action_coordinator
from src.training.metrics import TrainingMetrics
from src.training.custom_callbacks import WaypointMetricsCallback


def train(timesteps=1000000, name="coordinator_v1", coord_seed=None, n_envs=8):
    """
    Train action coordinator agent.
    
    The coordinator learns to issue velocity commands that are executed by a PID controller.
    No motor agent is needed - velocity tracking is handled by the PID controller.
    
    Args:
        timesteps: Total training timesteps for coordinator
        name: Experiment name for saving models
        coord_seed: Path to pretrained coordinator model to resume from (optional)
        n_envs: Number of parallel environments (default 8 for multi-core speedup)
    """
    print(f"\n{'='*60}")
    print(f"ACTION COORDINATOR TRAINING: {name}")
    print(f"Total timesteps: {timesteps}")
    print(f"Parallel environments: {n_envs}")
    if coord_seed:
        print(f"Coordinator seed: {coord_seed}")
    print(f"Motor control: PID controller (no agent)")
    print(f"{'='*60}\n")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/action_coordinator_gru", exist_ok=True)
    
    print(f"\n📊 TensorBoard logs will be saved to: ./logs/action_coordinator_gru")
    print(f"   To view: tensorboard --logdir=./logs")
    print()
    
    # Define training phases (name, steps, stage)
    # Only coordinator training - no motor agent
    phases = []
    
    # Skip coordinator preseed if coordinator seed is provided
    if not (coord_seed and os.path.exists(coord_seed + ".zip")):
        phases.append(("Preseed Coordinator", timesteps, "coordinator"))
    else:
        print("Skipping coordinator preseed (using seed model)")
    
    phases.extend([
        ("Train Coordinator", timesteps, "coordinator"),
    ])
    
    # Trim to fit timesteps
    total_planned = sum(s[1] for s in phases)
    if timesteps < total_planned:
        cumsum = 0
        trimmed = []
        for phase in phases:
            if cumsum + phase[1] <= timesteps:
                trimmed.append(phase)
                cumsum += phase[1]
            else:
                break
        phases = trimmed
    
    # Create parallel environments for faster training
    print(f"Creating {n_envs} parallel environments...")
    
    def make_env():
        """Factory function to create environment instances"""
        def _init():
            return ActionCoordinatorEnv(gui=False, enable_vision=True)
        return _init
    
    # Use SubprocVecEnv for true parallelism (separate processes)
    # Falls back to DummyVecEnv if n_envs=1 (single environment, no overhead)
    if n_envs > 1:
        coord_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
        print(f"  Using SubprocVecEnv with {n_envs} parallel processes")
    else:
        coord_env = DummyVecEnv([make_env()])
        print(f"  Using single environment (DummyVecEnv)")
    
    # Wrap with VecMonitor to aggregate episode stats in main process
    # This fixes TensorBoard logging with SubprocVecEnv (prevents file lock race)
    coord_env = VecMonitor(coord_env, info_keywords=())  # Disable info logging to prevent conflicts
    print(f"  Wrapped with VecMonitor for episode statistics aggregation")
    
    # Create or load coordinator model
    print("Creating coordinator model...")
    
    if coord_seed and os.path.exists(coord_seed + ".zip"):
        print(f"  Loading coordinator model from {coord_seed}")
        # Try RecurrentPPO first (current architecture), fall back to PPO
        try:
            coord_model = RecurrentPPO.load(coord_seed, env=coord_env)
            print(f"  Loaded as RecurrentPPO (GRU architecture)")
            # Re-enable TensorBoard logging for loaded model
            coord_model.tensorboard_log = "./logs/action_coordinator_gru"
        except Exception as e:
            print(f"  RecurrentPPO load failed, trying PPO: {e}")
            coord_model = PPO.load(coord_seed, env=coord_env)
            print(f"  Loaded as PPO")
            # Re-enable TensorBoard logging for loaded model
            coord_model.tensorboard_log = "./logs/action_coordinator"
    else:
        if coord_seed:
            print(f"  Warning: Coordinator seed {coord_seed} not found, creating new model")
        coord_model = create_action_coordinator(coord_env, verbose=2)  # Max verbosity for debugging
    
    print(f"  Model tensorboard_log: {coord_model.tensorboard_log}")
    print(f"  Model verbose: {coord_model.verbose}")
    
    # Initialize metrics
    coord_metrics = TrainingMetrics(stage="coordinator")
    
    # Create checkpoint callback to save model periodically
    # Use larger intervals for parallel envs to avoid disk I/O bottleneck
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1000),  # Save every 50k steps (min 1000 to reduce I/O)
        save_path=f"./models/{name}_checkpoints/",
        name_prefix="coordinator",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=0  # Disable verbose to reduce console spam
    )
    
    # Create callbacks: checkpoint saving + waypoint metrics
    # VecMonitor handles episode reward and length automatically
    coord_callbacks = CallbackList([
        checkpoint_callback,
        WaypointMetricsCallback(verbose=0)
    ])
    
    total_steps = 0
    
    # Execute training phases
    for phase_name, steps, stage in phases:
        if steps == 0:
            continue
            
        print(f"\n{'='*60}")
        print(f"Phase: {phase_name} ({steps} steps)")
        print(f"{'='*60}\n")
        
        if stage == "coordinator":
            # Create fresh parallel environments for coordinator training
            if n_envs > 1:
                coord_env_updated = SubprocVecEnv([make_env() for _ in range(n_envs)])
            else:
                coord_env_updated = DummyVecEnv([make_env()])
            
            # Wrap with VecMonitor for proper logging
            coord_env_updated = VecMonitor(coord_env_updated, info_keywords=())  # Disable info logging to prevent conflicts
            coord_model.set_env(coord_env_updated)
            
            # Train coordinator
            print(f"  Training with TensorBoard logging to: {coord_model.tensorboard_log}")
            # Note: reset_num_timesteps must be True for first call to initialize logger
            coord_model.learn(
                total_timesteps=steps,
                reset_num_timesteps=(total_steps == 0),  # True for first phase only
                progress_bar=True,
                callback=coord_callbacks,
                tb_log_name=name  # Add run name for TensorBoard
            )
            total_steps += steps
            
            # Update metrics (episode length tracked in callbacks)
            coord_metrics.update(
                timesteps=total_steps,
                ep_length=0,  # Tracked by callbacks instead
            )
            
            # Save checkpoint
            coord_model.save(f"models/{name}_coordinator_{total_steps}")
            
            coord_env_updated.close()
    
    # Save final model
    print("\nSaving final model...")
    coord_model.save(f"models/{name}_final")
    
    # Cleanup
    coord_env.close()
    
    print(f"\n✅ Training complete! Coordinator model saved to models/{name}_*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train action coordinator agent")
    parser.add_argument("--timesteps", type=int, default=1000000,
                        help="Total training timesteps")
    parser.add_argument("--name", type=str, default="coordinator_v1",
                        help="Experiment name")
    parser.add_argument("--coord-seed", type=str, default=None,
                        help="Path to pretrained coordinator model to resume from (without .zip extension)")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments (default: 8)")
    
    args = parser.parse_args()
    train(timesteps=args.timesteps, name=args.name, coord_seed=args.coord_seed, n_envs=args.n_envs)
