"""Custom callbacks for logging detailed metrics to TensorBoard."""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


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
        
    def _on_step(self) -> bool:
        """
        Called after each step in the environment.
        
        Collects metrics from info dict and logs them when episode ends.
        """
        # Get info from all environments (handles vectorized envs)
        infos = self.locals.get("infos", [])
        
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
        Log batch-level crash statistics.
        """
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
            infos = self.locals.get("infos", [])
            
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
