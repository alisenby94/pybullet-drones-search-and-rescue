"""
Standardized Training Metrics

PURPOSE:
    Provide consistent metric tracking across all training stages.
    
USAGE:
    metrics = TrainingMetrics(stage="preseed")
    metrics.update(ep_length=100, ep_reward=50.5, tracking_error=0.5)
    metrics.log()  # Print formatted metrics
    history = metrics.get_history()  # Get all recorded metrics
"""

import numpy as np
from typing import Dict, Optional, List


class TrainingMetrics:
    """
    Standardized metrics for hierarchical training.
    
    Tracks episode stats, training progress, and stage-specific metrics.
    """
    
    def __init__(self, stage: str):
        """
        Initialize metrics tracker.
        
        Args:
            stage: Training stage name ("preseed", "motor", "coordinator", "co-train")
        """
        self.stage = stage
        self.history = {
            # Episode metrics
            'episode/length': [],
            'episode/reward': [],
            'episode/count': [],
            
            # Training metrics
            'train/timesteps': [],
            'train/fps': [],
            
            # Stage-specific metrics
            'stage/name': [],
        }
        
        # Add stage-specific metric keys
        if stage in ["preseed", "motor"]:
            self.history.update({
                'tracking/error': [],
                'tracking/vx_error': [],
                'tracking/vy_error': [],
                'tracking/vz_error': [],
                'tracking/omega_error': [],
            })
        elif stage in ["coordinator", "co-train"]:
            self.history.update({
                'waypoint/reached': [],
                'waypoint/distance': [],
                'command/extreme_penalty': [],
                'command/mean_magnitude': [],
                'compliance/mean': [],
            })
        
        # Running averages (for logging)
        self.episode_lengths = []
        self.episode_rewards = []
        self.episode_count = 0
        self.total_timesteps = 0
        
        # Stage-specific running data
        self.tracking_errors = []
        self.vx_errors = []
        self.vy_errors = []
        self.vz_errors = []
        self.omega_errors = []
        self.waypoints_reached = []
        self.waypoint_distances = []
        self.command_magnitudes = []
        self.compliance_values = []
    
    def update(self, **kwargs):
        """
        Update metrics with new data.
        
        Args:
            ep_length: Episode length
            ep_reward: Episode reward
            timesteps: Current total timesteps
            fps: Training speed (frames per second)
            tracking_error: Total tracking error (stage 2)
            vx_error, vy_error, vz_error, omega_error: Component errors (stage 2)
            waypoints_reached: Number of waypoints reached (stage 1)
            waypoint_distance: Distance to current waypoint (stage 1)
            command_magnitude: Mean action magnitude (stage 1)
            compliance: Tracking compliance (stage 1)
        """
        # Episode metrics
        if 'ep_length' in kwargs:
            self.episode_lengths.append(kwargs['ep_length'])
        
        if 'ep_reward' in kwargs:
            self.episode_rewards.append(kwargs['ep_reward'])
            self.episode_count += 1
        
        if 'timesteps' in kwargs:
            self.total_timesteps = kwargs['timesteps']
        
        # Stage 2 (motor control) metrics
        if 'tracking_error' in kwargs:
            self.tracking_errors.append(kwargs['tracking_error'])
        
        if 'vx_error' in kwargs:
            self.vx_errors.append(kwargs['vx_error'])
        
        if 'vy_error' in kwargs:
            self.vy_errors.append(kwargs['vy_error'])
        
        if 'vz_error' in kwargs:
            self.vz_errors.append(kwargs['vz_error'])
        
        if 'omega_error' in kwargs:
            self.omega_errors.append(kwargs['omega_error'])
        
        # Stage 1 (coordinator) metrics
        if 'waypoints_reached' in kwargs:
            self.waypoints_reached.append(kwargs['waypoints_reached'])
        
        if 'waypoint_distance' in kwargs:
            self.waypoint_distances.append(kwargs['waypoint_distance'])
        
        if 'command_magnitude' in kwargs:
            self.command_magnitudes.append(kwargs['command_magnitude'])
        
        if 'compliance' in kwargs:
            self.compliance_values.append(kwargs['compliance'])
    
    def log(self, window: int = 100) -> Dict[str, float]:
        """
        Log current metrics (moving average over window).
        
        Args:
            window: Number of episodes to average over
            
        Returns:
            Dict of metric name -> value
        """
        metrics = {}
        
        # Episode metrics
        if len(self.episode_lengths) > 0:
            recent_lengths = self.episode_lengths[-window:]
            metrics['episode/length'] = np.mean(recent_lengths)
            metrics['episode/length_std'] = np.std(recent_lengths)
        
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-window:]
            metrics['episode/reward'] = np.mean(recent_rewards)
            metrics['episode/reward_std'] = np.std(recent_rewards)
        
        metrics['episode/count'] = self.episode_count
        metrics['train/timesteps'] = self.total_timesteps
        
        # Stage-specific metrics
        if self.stage in ["preseed", "motor"] and len(self.tracking_errors) > 0:
            recent_errors = self.tracking_errors[-window:]
            metrics['tracking/error'] = np.mean(recent_errors)
            
            if len(self.vx_errors) > 0:
                metrics['tracking/vx_error'] = np.mean(self.vx_errors[-window:])
            if len(self.vy_errors) > 0:
                metrics['tracking/vy_error'] = np.mean(self.vy_errors[-window:])
            if len(self.vz_errors) > 0:
                metrics['tracking/vz_error'] = np.mean(self.vz_errors[-window:])
            if len(self.omega_errors) > 0:
                metrics['tracking/omega_error'] = np.mean(self.omega_errors[-window:])
        
        elif self.stage in ["coordinator", "co-train"]:
            if len(self.waypoints_reached) > 0:
                metrics['waypoint/reached'] = np.mean(self.waypoints_reached[-window:])
            
            if len(self.waypoint_distances) > 0:
                metrics['waypoint/distance'] = np.mean(self.waypoint_distances[-window:])
            
            if len(self.command_magnitudes) > 0:
                metrics['command/mean_magnitude'] = np.mean(self.command_magnitudes[-window:])
            
            if len(self.compliance_values) > 0:
                metrics['compliance/mean'] = np.mean(self.compliance_values[-window:])
        
        return metrics
    
    def save_to_history(self, metrics: Dict[str, float]):
        """
        Save current metrics to history.
        
        Args:
            metrics: Dict of metric name -> value from log()
        """
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_history(self) -> Dict[str, List[float]]:
        """
        Get complete training history.
        
        Returns:
            Dict of metric name -> list of values
        """
        return self.history
    
    def print_metrics(self, metrics: Dict[str, float]):
        """
        Pretty-print metrics.
        
        Args:
            metrics: Dict from log()
        """
        print(f"\n[{self.stage.upper()}] Step {metrics.get('train/timesteps', 0)}")
        print(f"  Episodes: {metrics.get('episode/count', 0)}")
        
        if 'episode/length' in metrics:
            print(f"  Length: {metrics['episode/length']:.1f} ± {metrics.get('episode/length_std', 0):.1f}")
        
        if 'episode/reward' in metrics:
            print(f"  Reward: {metrics['episode/reward']:.2f} ± {metrics.get('episode/reward_std', 0):.2f}")
        
        # Stage-specific
        if 'tracking/error' in metrics:
            print(f"  Tracking Error: {metrics['tracking/error']:.4f}")
            if 'tracking/vx_error' in metrics:
                print(f"    vx: {metrics['tracking/vx_error']:.4f}, "
                      f"vy: {metrics['tracking/vy_error']:.4f}, "
                      f"vz: {metrics['tracking/vz_error']:.4f}, "
                      f"ω: {metrics['tracking/omega_error']:.4f}")
        
        if 'waypoint/reached' in metrics:
            print(f"  Waypoints Reached: {metrics['waypoint/reached']:.2f}")
        
        if 'waypoint/distance' in metrics:
            print(f"  Waypoint Distance: {metrics['waypoint/distance']:.3f}")
        
        if 'compliance/mean' in metrics:
            print(f"  Compliance: {metrics['compliance/mean']:.3f}")
    
    def reset_running_stats(self):
        """Clear running statistics (call at end of logging window)."""
        # Keep history, but clear running averages
        pass  # We use [-window:] slicing, so no need to clear
