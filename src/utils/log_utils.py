"""
Progress Bar Utilities

PURPOSE:
    Standardized progress bars for all training operations.
    
USAGE:
    from src.utils.log_utils import create_training_progress_bar
    
    pbar = create_training_progress_bar(50000, "Preseed Motor Control")
    for step in range(50000):
        # ... training ...
        pbar.update(1)
        pbar.set_postfix({'reward': mean_reward, 'length': mean_length})
    pbar.close()
"""

from tqdm import tqdm
from typing import Optional, Dict


def create_training_progress_bar(
    total_steps: int,
    description: str,
    position: int = 0,
    leave: bool = True
) -> tqdm:
    """
    Create a standardized progress bar for training.
    
    Args:
        total_steps: Total number of steps
        description: Progress bar description
        position: Line position (for multiple bars)
        leave: Keep bar after completion
        
    Returns:
        tqdm progress bar
    """
    return tqdm(
        total=total_steps,
        desc=description,
        unit='steps',
        ncols=100,
        position=position,
        leave=leave,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )


def create_episode_progress_bar(
    total_episodes: int,
    description: str,
    position: int = 0,
    leave: bool = True
) -> tqdm:
    """
    Create a progress bar for episode counting.
    
    Args:
        total_episodes: Total number of episodes
        description: Progress bar description
        position: Line position
        leave: Keep bar after completion
        
    Returns:
        tqdm progress bar
    """
    return tqdm(
        total=total_episodes,
        desc=description,
        unit='ep',
        ncols=100,
        position=position,
        leave=leave,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )


class TrainingProgressTracker:
    """
    Multi-stage progress tracker with nested bars.
    
    Example:
        tracker = TrainingProgressTracker([
            ("Preseed", 50000),
            ("Cycle 1 Motor", 20000),
            ("Cycle 1 Coordinator", 20000),
        ])
        
        for stage_name, steps in stages:
            pbar = tracker.start_stage(stage_name, steps)
            for step in range(steps):
                # ... training ...
                tracker.update(1, reward=..., length=...)
            tracker.finish_stage()
    """
    
    def __init__(self, stages: list):
        """
        Initialize multi-stage tracker.
        
        Args:
            stages: List of (name, steps) tuples
        """
        self.stages = stages
        self.total_steps = sum(steps for _, steps in stages)
        
        # Overall progress bar
        self.overall_pbar = tqdm(
            total=self.total_steps,
            desc="Total Training",
            unit='steps',
            position=0,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        # Current stage bar
        self.stage_pbar: Optional[tqdm] = None
        self.current_stage = 0
    
    def start_stage(self, stage_name: str, steps: int) -> tqdm:
        """
        Start a new training stage.
        
        Args:
            stage_name: Name of stage
            steps: Number of steps in stage
            
        Returns:
            Progress bar for this stage
        """
        if self.stage_pbar is not None:
            self.stage_pbar.close()
        
        self.stage_pbar = tqdm(
            total=steps,
            desc=f"  {stage_name}",
            unit='steps',
            position=1,
            ncols=100,
            leave=False,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        return self.stage_pbar
    
    def update(self, n: int = 1, **postfix):
        """
        Update progress bars.
        
        Args:
            n: Number of steps
            **postfix: Additional info to display
        """
        self.overall_pbar.update(n)
        
        if self.stage_pbar is not None:
            self.stage_pbar.update(n)
            if postfix:
                self.stage_pbar.set_postfix(postfix)
    
    def finish_stage(self):
        """Finish current stage."""
        if self.stage_pbar is not None:
            self.stage_pbar.close()
            self.stage_pbar = None
        
        self.current_stage += 1
    
    def finish(self):
        """Finish all training."""
        if self.stage_pbar is not None:
            self.stage_pbar.close()
        
        self.overall_pbar.close()
        print("\n✅ Training complete!")


def format_metrics_for_postfix(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Format metrics dict for progress bar postfix display.
    
    Args:
        metrics: Raw metrics dict
        
    Returns:
        Formatted metrics for display
    """
    formatted = {}
    
    for key, value in metrics.items():
        # Short names for display
        short_key = key.split('/')[-1]
        
        # Format based on magnitude
        if abs(value) < 0.01:
            formatted[short_key] = f"{value:.4f}"
        elif abs(value) < 1.0:
            formatted[short_key] = f"{value:.3f}"
        elif abs(value) < 100:
            formatted[short_key] = f"{value:.1f}"
        else:
            formatted[short_key] = f"{int(value)}"
    
    return formatted
