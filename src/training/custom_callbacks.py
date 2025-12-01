"""Custom callbacks for logging detailed metrics to TensorBoard."""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class WaypointMetricsCallback(BaseCallback):
    """
    Log current waypoint metrics every rollout (don't wait for episode completion).
    Also logs velocity tracking metrics.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rollout_count = 0
        self.print_frequency = 10  # Print action stats every N rollouts
        
        # Velocity tracking metrics (collected at every step)
        self.rollout_velocity_errors = []
        self.rollout_target_mags = []
        self.rollout_actual_mags = []
    
    def _on_step(self) -> bool:
        """Collect velocity tracking metrics every step."""
        # Get info from all environments
        infos = self.locals.get("infos", [])
        
        # Handle case where infos is None, a tuple, or not a list
        if infos is None:
            infos = []
        elif isinstance(infos, tuple):
            infos = list(infos)  # Convert tuple to list
        elif not isinstance(infos, list):
            infos = [infos]  # Wrap single item in list
        
        # Collect velocity tracking metrics from each environment
        for info in infos:
            if info is None or not isinstance(info, dict):
                continue
            
            # Collect velocity tracking metrics
            if 'velocity_tracking_error' in info:
                self.rollout_velocity_errors.append(info['velocity_tracking_error'])
                self.rollout_target_mags.append(info['target_velocity_mag'])
                self.rollout_actual_mags.append(info['actual_velocity_mag'])
        
        return True
        
    def _on_rollout_end(self) -> None:
        """Log current waypoint metrics from all environments."""
        self.rollout_count += 1
        
        # Get current info from all environments
        infos = self.locals.get("infos", [])
        
        if not isinstance(infos, list):
            infos = [infos] if infos is not None else []
        
        # Collect current metrics from all envs
        distances = []
        waypoints_reached = []
        
        # Reward component tracking
        reward_forward_velocity = []
        reward_distance = []
        reward_alignment = []
        reward_waypoint_completion = []
        reward_altitude = []
        reward_obstacle_penalty = []
        
        for info in infos:
            if info is None:
                continue
            
            if 'waypoint_distance' in info:
                distances.append(info['waypoint_distance'])
            
            if 'waypoints_reached' in info:
                waypoints_reached.append(info['waypoints_reached'])
            
            # Collect reward components
            if 'reward_forward_velocity' in info:
                reward_forward_velocity.append(info['reward_forward_velocity'])
            if 'reward_distance' in info:
                reward_distance.append(info['reward_distance'])
            if 'reward_alignment' in info:
                reward_alignment.append(info['reward_alignment'])
            if 'reward_waypoint_completion' in info:
                reward_waypoint_completion.append(info['reward_waypoint_completion'])
            if 'reward_altitude' in info:
                reward_altitude.append(info['reward_altitude'])
            if 'reward_obstacle_penalty' in info:
                reward_obstacle_penalty.append(info['reward_obstacle_penalty'])
        
        # Log aggregated metrics
        if distances:
            self.logger.record("waypoint/distance_mean", np.mean(distances))
            self.logger.record("waypoint/distance_min", np.min(distances))
            self.logger.record("waypoint/distance_max", np.max(distances))
        
        if waypoints_reached:
            self.logger.record("waypoint/reached_mean", np.mean(waypoints_reached))
            self.logger.record("waypoint/reached_max", np.max(waypoints_reached))
            self.logger.record("waypoint/reached_total", np.sum(waypoints_reached))
        
        # Log reward component breakdown
        if reward_forward_velocity:
            self.logger.record("reward_components/forward_velocity_mean", np.mean(reward_forward_velocity))
        if reward_distance:
            self.logger.record("reward_components/distance_mean", np.mean(reward_distance))
        if reward_alignment:
            self.logger.record("reward_components/alignment_mean", np.mean(reward_alignment))
        if reward_waypoint_completion:
            # Log the raw reward mean (mostly 0, spiky at 1000)
            self.logger.record("reward_components/waypoint_completion_mean", np.mean(reward_waypoint_completion))
            # Log completion rate: what % of steps captured a waypoint
            completions = [1 if r > 0 else 0 for r in reward_waypoint_completion]
            completion_rate = np.mean(completions) * 100  # As percentage
            self.logger.record("waypoint/completion_rate_percent", completion_rate)
            # Log total completions in this rollout
            total_completions = np.sum(completions)
            self.logger.record("waypoint/completions_this_rollout", total_completions)
        if reward_altitude:
            self.logger.record("reward_components/altitude_mean", np.mean(reward_altitude))
        if reward_obstacle_penalty:
            self.logger.record("reward_components/obstacle_penalty_mean", np.mean(reward_obstacle_penalty))
        
        # Log action statistics from rollout buffer
        # Actions are stored in the buffer during rollout collection
        # Get the raw actions (before scaling) from the rollout buffer
        if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer.full:
            actions = self.model.rollout_buffer.actions
            
            # actions shape: (buffer_size, n_envs, action_dim) or (buffer_size * n_envs, action_dim)
            # Flatten to (N, action_dim) for statistics
            actions_flat = actions.reshape(-1, actions.shape[-1])
            
            # Compute statistics per action dimension
            # Action space: [vx, vy, vz] in range [-1, 1] (3D velocity)
            action_names = ['vx', 'vy', 'vz']
            
            for i, name in enumerate(action_names):
                if i < actions_flat.shape[1]:  # Safety check for dimension
                    action_dim = actions_flat[:, i]
                    self.logger.record(f"action/{name}_mean", np.mean(action_dim))
                    self.logger.record(f"action/{name}_min", np.min(action_dim))
                    self.logger.record(f"action/{name}_max", np.max(action_dim))
                    self.logger.record(f"action/{name}_std", np.std(action_dim))
                    self.logger.record(f"action/{name}_abs_mean", np.mean(np.abs(action_dim)))
            
            # Overall action magnitude (L2 norm of velocity components)
            action_magnitude = np.sqrt(np.sum(actions_flat[:, :3]**2, axis=1))
            self.logger.record("action/magnitude_mean", np.mean(action_magnitude))
            self.logger.record("action/magnitude_min", np.min(action_magnitude))
            self.logger.record("action/magnitude_max", np.max(action_magnitude))
            
            # Action utilization: percentage of max action (1.0) being used
            # This tells us if the model is being conservative (low %) or aggressive (high %)
            action_utilization = np.mean(action_magnitude) / 1.0  # Max magnitude in [-1,1]^3 space
            self.logger.record("action/utilization", action_utilization * 100)  # As percentage
            
            # Print action statistics to console periodically
            if self.verbose > 0 and self.rollout_count % self.print_frequency == 0:
                print(f"\n{'='*80}")
                print(f"ACTION STATISTICS (Rollout {self.rollout_count})")
                print(f"{'='*80}")
                print(f"  vx:      mean={np.mean(actions_flat[:, 0]):+.3f}  min={np.min(actions_flat[:, 0]):+.3f}  max={np.max(actions_flat[:, 0]):+.3f}  std={np.std(actions_flat[:, 0]):.3f}")
                print(f"  vy:      mean={np.mean(actions_flat[:, 1]):+.3f}  min={np.min(actions_flat[:, 1]):+.3f}  max={np.max(actions_flat[:, 1]):+.3f}  std={np.std(actions_flat[:, 1]):.3f}")
                print(f"  vz:      mean={np.mean(actions_flat[:, 2]):+.3f}  min={np.min(actions_flat[:, 2]):+.3f}  max={np.max(actions_flat[:, 2]):+.3f}  std={np.std(actions_flat[:, 2]):.3f}")
                print(f"  Magnitude: mean={np.mean(action_magnitude):.3f}  min={np.min(action_magnitude):.3f}  max={np.max(action_magnitude):.3f}")
                print(f"  Utilization: {action_utilization * 100:.1f}% of maximum")
                print(f"{'='*80}\n")
        
        # Log velocity tracking metrics (collected every step during rollout)
        if self.rollout_velocity_errors:
            self.logger.record("velocity/tracking_error_mean", np.mean(self.rollout_velocity_errors))
            self.logger.record("velocity/tracking_error_max", np.max(self.rollout_velocity_errors))
            self.logger.record("velocity/tracking_error_std", np.std(self.rollout_velocity_errors))
            # Clear for next rollout
            self.rollout_velocity_errors = []
        
        if self.rollout_target_mags:
            self.logger.record("velocity/target_magnitude_mean", np.mean(self.rollout_target_mags))
            self.logger.record("velocity/target_magnitude_max", np.max(self.rollout_target_mags))
            # Clear for next rollout
            self.rollout_target_mags = []
        
        if self.rollout_actual_mags:
            self.logger.record("velocity/actual_magnitude_mean", np.mean(self.rollout_actual_mags))
            self.logger.record("velocity/actual_magnitude_max", np.max(self.rollout_actual_mags))
            self.logger.record("velocity/actual_magnitude_std", np.std(self.rollout_actual_mags))
            # Clear for next rollout
            self.rollout_actual_mags = []


class DetailedMetricsCallback(BaseCallback):
    """
    Callback for logging detailed metrics from info dict to TensorBoard.
    
    This logs the custom error metrics we added to _computeInfo() in our environments.
    Tracks obstacle crashes per batch for monitoring training safety.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_metrics = {}
        # Batch-level tracking
        self.batch_obstacle_crashes = 0
        self.batch_general_crashes = 0
        self.batch_episodes = 0
        self.total_episodes = 0
        self.total_obstacle_crashes = 0
        self.total_general_crashes = 0
        
        # Rollout-level velocity tracking (collected every step)
        self.rollout_velocity_errors = []
        self.rollout_target_mags = []
        self.rollout_actual_mags = []
        
    def _on_step(self) -> bool:
        """
        Called after each step in the environment.
        
        Collects metrics from info dict and logs them when episode ends.
        """
        # Get info from all environments (handles vectorized envs)
        # For SubprocVecEnv, infos may be a tuple or list of dicts
        infos = self.locals.get("infos", [])
        
        # Handle case where infos is None, a tuple, or not a list
        if infos is None:
            infos = []
        elif isinstance(infos, tuple):
            infos = list(infos)  # Convert tuple to list
        elif not isinstance(infos, list):
            infos = [infos]  # Wrap single item in list
        
        for idx, info in enumerate(infos):
            if info is None:
                continue
                
            # Initialize storage for this environment if needed
            if idx not in self.episode_metrics:
                self.episode_metrics[idx] = {}
            
            # Collect metrics from info dict
            # Motor controller metrics
            if 'error_vx' in info:
                for key in ['error_vx', 'error_vy', 'error_vz', 'error_wz']:
                    if key not in self.episode_metrics[idx]:
                        self.episode_metrics[idx][key] = []
                    self.episode_metrics[idx][key].append(info[key])
            
            # Coordinator metrics
            if 'error_progress' in info:
                for key in ['error_progress', 'error_velocity_align', 'error_lateral', 
                           'error_acceleration', 'waypoint_distance', 'waypoints_reached']:
                    if key not in self.episode_metrics[idx]:
                        self.episode_metrics[idx][key] = []
                    if key in info:
                        self.episode_metrics[idx][key].append(info[key])
            
            # Debug: Print what type and content info actually has
            if idx == 0 and self.num_timesteps < 10:  # Only print for first env, first 10 steps
                print(f"[DEBUG] Step {self.num_timesteps}: info type = {type(info)}")
                if isinstance(info, dict):
                    print(f"  info keys = {list(info.keys())[:15]}")
                    if 'velocity_tracking_error' in info:
                        print(f"  ✅ velocity_tracking_error FOUND: {info['velocity_tracking_error']:.3f}")
                    else:
                        print(f"  ❌ velocity_tracking_error NOT in info dict!")
                else:
                    print(f"  ❌ info is not a dict! Value: {info}")
            
            # Velocity tracking metrics (NEW)
            if 'velocity_tracking_error' in info:
                for key in ['target_velocity_mag', 'actual_velocity_mag', 'velocity_tracking_error']:
                    if key not in self.episode_metrics[idx]:
                        self.episode_metrics[idx][key] = []
                    if key in info:
                        self.episode_metrics[idx][key].append(info[key])
                
                # Also track at rollout level (for continuous logging)
                self.rollout_velocity_errors.append(info['velocity_tracking_error'])
                self.rollout_target_mags.append(info['target_velocity_mag'])
                self.rollout_actual_mags.append(info['actual_velocity_mag'])
                
                # Debug: Print first few samples to verify collection
                if len(self.rollout_velocity_errors) <= 3:
                    print(f"[VELOCITY DEBUG] Collected sample {len(self.rollout_velocity_errors)}: "
                          f"target={info['target_velocity_mag']:.3f}, actual={info['actual_velocity_mag']:.3f}, "
                          f"error={info['velocity_tracking_error']:.3f}")
                
                # Debug: Print velocity metrics collection
                if self.num_timesteps % 500 == 0:  # Print every 500 steps
                    print(f"[VELOCITY DEBUG] Step {self.num_timesteps}: Target={info['target_velocity_mag']:.3f} m/s, "
                          f"Actual={info['actual_velocity_mag']:.3f} m/s, Error={info['velocity_tracking_error']:.3f} m/s")
            
            # Track obstacle crashes
            if 'obstacle_crash' in info and info['obstacle_crash']:
                if 'obstacle_crash' not in self.episode_metrics[idx]:
                    self.episode_metrics[idx]['obstacle_crash'] = False
                self.episode_metrics[idx]['obstacle_crash'] = True
            
            # Track general crashes
            if 'crashed' in info and info['crashed']:
                if 'crashed' not in self.episode_metrics[idx]:
                    self.episode_metrics[idx]['crashed'] = False
                self.episode_metrics[idx]['crashed'] = True
            
            # Check if episode ended
            dones = self.locals.get("dones", [False])
            if idx < len(dones) and dones[idx]:
                # Episode ended - update batch counters
                self.batch_episodes += 1
                if self.episode_metrics[idx].get('obstacle_crash', False):
                    self.batch_obstacle_crashes += 1
                if self.episode_metrics[idx].get('crashed', False):
                    self.batch_general_crashes += 1
                
                # Log episode-level metrics
                if self.episode_metrics[idx]:
                    self._log_episode_metrics(idx)
                    # Reset metrics for this environment
                    self.episode_metrics[idx] = {}
        
        return True
    
    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout (batch collection).
        Log batch-level crash statistics and velocity tracking stats.
        """
        # Always log, even if batch_episodes is 0 (helps debug SubprocVecEnv issues)
        self.logger.record("debug/callback_active", 1)
        self.logger.record("debug/batch_episodes_seen", self.batch_episodes)
        
        # Debug: Print rollout statistics
        print(f"\n[ROLLOUT END DEBUG] Collected {len(self.rollout_velocity_errors)} velocity samples")
        if self.rollout_velocity_errors:
            print(f"  Target velocity: mean={np.mean(self.rollout_target_mags):.3f} m/s, max={np.max(self.rollout_target_mags):.3f} m/s")
            print(f"  Actual velocity: mean={np.mean(self.rollout_actual_mags):.3f} m/s, max={np.max(self.rollout_actual_mags):.3f} m/s")
            print(f"  Tracking error:  mean={np.mean(self.rollout_velocity_errors):.3f} m/s, max={np.max(self.rollout_velocity_errors):.3f} m/s")
        
        # Log rollout-level velocity statistics (collected every step)
        if self.rollout_velocity_errors:
            self.logger.record("velocity/tracking_error_mean", np.mean(self.rollout_velocity_errors))
            self.logger.record("velocity/tracking_error_max", np.max(self.rollout_velocity_errors))
            self.logger.record("velocity/tracking_error_std", np.std(self.rollout_velocity_errors))
            # Clear for next rollout
            self.rollout_velocity_errors = []
        
        if self.rollout_target_mags:
            self.logger.record("velocity/target_magnitude_mean", np.mean(self.rollout_target_mags))
            self.logger.record("velocity/target_magnitude_max", np.max(self.rollout_target_mags))
            # Clear for next rollout
            self.rollout_target_mags = []
        
        if self.rollout_actual_mags:
            self.logger.record("velocity/actual_magnitude_mean", np.mean(self.rollout_actual_mags))
            self.logger.record("velocity/actual_magnitude_max", np.max(self.rollout_actual_mags))
            self.logger.record("velocity/actual_magnitude_std", np.std(self.rollout_actual_mags))
            # Clear for next rollout
            self.rollout_actual_mags = []
        
        if self.batch_episodes > 0:
            # Update totals
            self.total_episodes += self.batch_episodes
            self.total_obstacle_crashes += self.batch_obstacle_crashes
            self.total_general_crashes += self.batch_general_crashes
            
            # Log batch statistics
            obstacle_crash_rate = self.batch_obstacle_crashes / self.batch_episodes
            general_crash_rate = self.batch_general_crashes / self.batch_episodes
            
            self.logger.record("crashes/batch_obstacle_crashes", self.batch_obstacle_crashes)
            self.logger.record("crashes/batch_general_crashes", self.batch_general_crashes)
            self.logger.record("crashes/batch_episodes", self.batch_episodes)
            self.logger.record("crashes/batch_obstacle_crash_rate", obstacle_crash_rate)
            self.logger.record("crashes/batch_general_crash_rate", general_crash_rate)
            
            # Log cumulative statistics
            total_obstacle_rate = self.total_obstacle_crashes / self.total_episodes
            total_general_rate = self.total_general_crashes / self.total_episodes
            
            self.logger.record("crashes/total_obstacle_crashes", self.total_obstacle_crashes)
            self.logger.record("crashes/total_general_crashes", self.total_general_crashes)
            self.logger.record("crashes/total_episodes", self.total_episodes)
            self.logger.record("crashes/total_obstacle_crash_rate", total_obstacle_rate)
            self.logger.record("crashes/total_general_crash_rate", total_general_rate)
            
            # Reset batch counters
            self.batch_obstacle_crashes = 0
            self.batch_general_crashes = 0
            self.batch_episodes = 0
    
    def _log_episode_metrics(self, env_idx):
        """Log the collected episode metrics to TensorBoard."""
        metrics = self.episode_metrics[env_idx]
        
        # Log motor controller errors
        if 'error_vx' in metrics:
            self.logger.record("motor/error_vx_mean", np.mean(np.abs(metrics['error_vx'])))
            self.logger.record("motor/error_vy_mean", np.mean(np.abs(metrics['error_vy'])))
            self.logger.record("motor/error_vz_mean", np.mean(np.abs(metrics['error_vz'])))
            self.logger.record("motor/error_wz_mean", np.mean(np.abs(metrics['error_wz'])))
            
            # Total tracking error
            total_error = np.mean([
                np.abs(metrics['error_vx']),
                np.abs(metrics['error_vy']),
                np.abs(metrics['error_vz']),
                np.abs(metrics['error_wz'])
            ])
            self.logger.record("motor/total_tracking_error", np.mean(total_error))
            
            # Max errors (worst case)
            self.logger.record("motor/error_vx_max", np.max(np.abs(metrics['error_vx'])))
            self.logger.record("motor/error_vy_max", np.max(np.abs(metrics['error_vy'])))
            self.logger.record("motor/error_vz_max", np.max(np.abs(metrics['error_vz'])))
            self.logger.record("motor/error_wz_max", np.max(np.abs(metrics['error_wz'])))
        
        # Log coordinator errors
        if 'error_progress' in metrics:
            self.logger.record("coordinator/error_progress_mean", np.mean(np.abs(metrics['error_progress'])))
            
        if 'error_velocity_align' in metrics:
            self.logger.record("coordinator/error_velocity_align_mean", np.mean(metrics['error_velocity_align']))
            
        if 'error_lateral' in metrics:
            self.logger.record("coordinator/error_lateral_mean", np.mean(metrics['error_lateral']))
            
        if 'error_acceleration' in metrics:
            self.logger.record("coordinator/error_acceleration_mean", np.mean(metrics['error_acceleration']))
            
        if 'waypoint_distance' in metrics:
            self.logger.record("coordinator/waypoint_distance_final", metrics['waypoint_distance'][-1])
            self.logger.record("coordinator/waypoint_distance_mean", np.mean(metrics['waypoint_distance']))
            
        if 'waypoints_reached' in metrics:
            self.logger.record("coordinator/waypoints_reached", metrics['waypoints_reached'][-1])
        
        # Log velocity tracking metrics (NEW)
        if 'velocity_tracking_error' in metrics:
            self.logger.record("velocity/tracking_error_mean", np.mean(metrics['velocity_tracking_error']))
            self.logger.record("velocity/tracking_error_max", np.max(metrics['velocity_tracking_error']))
            self.logger.record("velocity/tracking_error_min", np.min(metrics['velocity_tracking_error']))
            
        if 'target_velocity_mag' in metrics:
            self.logger.record("velocity/target_magnitude_mean", np.mean(metrics['target_velocity_mag']))
            self.logger.record("velocity/target_magnitude_max", np.max(metrics['target_velocity_mag']))
            
        if 'actual_velocity_mag' in metrics:
            self.logger.record("velocity/actual_magnitude_mean", np.mean(metrics['actual_velocity_mag']))
            self.logger.record("velocity/actual_magnitude_max", np.max(metrics['actual_velocity_mag']))
        
        # Log crash information for this episode
        if metrics.get('obstacle_crash', False):
            self.logger.record("crashes/obstacle_crash", 1)
        else:
            self.logger.record("crashes/obstacle_crash", 0)
            
        if metrics.get('crashed', False):
            self.logger.record("crashes/general_crash", 1)
        else:
            self.logger.record("crashes/general_crash", 0)


class PeriodicMetricsCallback(BaseCallback):
    """
    Callback that logs current error metrics every N steps (not just at episode end).
    
    Useful for seeing real-time convergence during long episodes.
    """
    
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.step_count = 0
        
    def _on_step(self) -> bool:
        """Log current metrics every log_freq steps."""
        self.step_count += 1
        
        if self.step_count % self.log_freq == 0:
            # Log that we're actually being called
            self.logger.record("debug/periodic_callback_steps", self.step_count)
            
            infos = self.locals.get("infos", [])
            
            # Handle case where infos is None or not a list
            if not isinstance(infos, list):
                infos = [infos] if infos is not None else []
            
            # Collect current step metrics
            for info in infos:
                if info is None:
                    continue
                
                # Motor controller instant metrics
                if 'error_vx' in info:
                    self.logger.record("motor_instant/error_vx", abs(info['error_vx']))
                    self.logger.record("motor_instant/error_vy", abs(info['error_vy']))
                    self.logger.record("motor_instant/error_vz", abs(info['error_vz']))
                    self.logger.record("motor_instant/error_wz", abs(info['error_wz']))
                
                # Coordinator instant metrics
                if 'waypoint_distance' in info:
                    self.logger.record("coordinator_instant/waypoint_distance", info['waypoint_distance'])
                
                if 'waypoints_reached' in info:
                    self.logger.record("coordinator_instant/waypoints_reached", info['waypoints_reached'])
                
                # Only log first environment to avoid spam
                break
        
        return True
