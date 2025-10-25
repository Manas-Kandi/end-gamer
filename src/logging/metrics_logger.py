"""Metrics logging for training progress and evaluation results."""

import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class MetricsLogger:
    """Logger for training metrics with TensorBoard and file-based logging.
    
    Provides unified interface for logging scalars, losses, evaluation metrics,
    and other training statistics to both TensorBoard and structured log files.
    """
    
    def __init__(self, log_dir: str, experiment_name: Optional[str] = None):
        """Initialize metrics logger.
        
        Args:
            log_dir: Directory to store logs
            experiment_name: Optional experiment name for organization
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create log directory structure
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer if available
        if TENSORBOARD_AVAILABLE:
            self.tb_writer = SummaryWriter(log_dir=str(self.experiment_dir / "tensorboard"))
        else:
            self.tb_writer = None
            print("Warning: TensorBoard not available. Install with: pip install tensorboard")
        
        # Initialize file-based logging
        self.metrics_file = self.experiment_dir / "metrics.jsonl"
        self.training_log = self.experiment_dir / "training.log"
        
        # Initialize log files
        self._init_log_files()
        
        # Track step counters
        self.global_step = 0
        self.training_step = 0
        self.evaluation_step = 0
    
    def _init_log_files(self) -> None:
        """Initialize log files with headers."""
        # Create training log with header
        with open(self.training_log, 'w') as f:
            f.write(f"Training Log - {self.experiment_name}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n\n")
        
        # Metrics file will be written in JSONL format (one JSON object per line)
        if not self.metrics_file.exists():
            self.metrics_file.touch()
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        """Log single scalar metric.
        
        Args:
            tag: Metric name/tag
            value: Scalar value to log
            step: Step number (uses global_step if None)
        """
        if step is None:
            step = self.global_step
        
        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)
        
        # Log to structured file
        self._log_to_file({
            'type': 'scalar',
            'tag': tag,
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_losses(self, losses: Dict[str, float], step: Optional[int] = None) -> None:
        """Log training loss metrics.
        
        Args:
            losses: Dictionary of loss names to values
            step: Step number (uses training_step if None)
        """
        if step is None:
            step = self.training_step
        self.training_step += 1
        
        # Log each loss individually
        for loss_name, loss_value in losses.items():
            self.log_scalar(f"loss/{loss_name}", loss_value, step)
        
        # Log combined losses entry
        self._log_to_file({
            'type': 'losses',
            'losses': losses,
            'step': step,
            'timestamp': datetime.now().isoformat()
        })
        
        # Log to training file
        self._log_to_training_file(f"Step {step}: Losses - {losses}")
    
    def log_evaluation(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log evaluation metrics.
        
        Args:
            metrics: Dictionary of evaluation metric names to values
            step: Step number (uses evaluation_step if None)
        """
        if step is None:
            step = self.evaluation_step
        self.evaluation_step += 1
        
        # Log each metric individually
        for metric_name, metric_value in metrics.items():
            self.log_scalar(f"eval/{metric_name}", metric_value, step)
        
        # Log combined evaluation entry
        self._log_to_file({
            'type': 'evaluation',
            'metrics': metrics,
            'step': step,
            'timestamp': datetime.now().isoformat()
        })
        
        # Log to training file
        self._log_to_training_file(f"Evaluation {step}: {metrics}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """Log hyperparameters.
        
        Args:
            hparams: Dictionary of hyperparameter names to values
        """
        # Log to TensorBoard
        if self.tb_writer:
            # Convert all values to scalars for TensorBoard
            scalar_hparams = {}
            for key, value in hparams.items():
                if isinstance(value, (int, float)):
                    scalar_hparams[key] = value
                else:
                    scalar_hparams[key] = str(value)
            
            self.tb_writer.add_hparams(scalar_hparams, {})
        
        # Log to structured file
        self._log_to_file({
            'type': 'hyperparameters',
            'hparams': hparams,
            'timestamp': datetime.now().isoformat()
        })
        
        # Log to training file
        self._log_to_training_file(f"Hyperparameters: {hparams}")
    
    def log_training_progress(self, iteration: int, total_games: int, 
                            games_per_iteration: int, message: str = "") -> None:
        """Log training progress information.
        
        Args:
            iteration: Current training iteration
            total_games: Total games played so far
            games_per_iteration: Games per iteration
            message: Optional additional message
        """
        progress_info = {
            'iteration': iteration,
            'total_games': total_games,
            'games_per_iteration': games_per_iteration,
            'progress_percent': (total_games / 100000) * 100 if total_games > 0 else 0
        }
        
        # Log progress metrics
        self.log_scalar("training/iteration", iteration)
        self.log_scalar("training/total_games", total_games)
        self.log_scalar("training/progress_percent", progress_info['progress_percent'])
        
        # Log to structured file
        self._log_to_file({
            'type': 'training_progress',
            'progress': progress_info,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Log to training file
        progress_msg = (f"Iteration {iteration}: {total_games} games played "
                       f"({progress_info['progress_percent']:.1f}%)")
        if message:
            progress_msg += f" - {message}"
        self._log_to_training_file(progress_msg)
    
    def log_resource_usage(self, cpu_percent: float, memory_mb: float, 
                          gpu_memory_mb: Optional[float] = None) -> None:
        """Log system resource usage.
        
        Args:
            cpu_percent: CPU usage percentage
            memory_mb: Memory usage in MB
            gpu_memory_mb: GPU memory usage in MB (optional)
        """
        self.log_scalar("resources/cpu_percent", cpu_percent)
        self.log_scalar("resources/memory_mb", memory_mb)
        
        if gpu_memory_mb is not None:
            self.log_scalar("resources/gpu_memory_mb", gpu_memory_mb)
        
        # Log to structured file
        resource_info = {
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb
        }
        if gpu_memory_mb is not None:
            resource_info['gpu_memory_mb'] = gpu_memory_mb
        
        self._log_to_file({
            'type': 'resource_usage',
            'resources': resource_info,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_mcts_statistics(self, nodes_per_second: float, avg_depth: float,
                           simulations: int, step: Optional[int] = None) -> None:
        """Log MCTS performance statistics.
        
        Args:
            nodes_per_second: MCTS nodes evaluated per second
            avg_depth: Average search depth
            simulations: Number of simulations per search
            step: Step number
        """
        if step is None:
            step = self.global_step
        
        self.log_scalar("mcts/nodes_per_second", nodes_per_second, step)
        self.log_scalar("mcts/avg_depth", avg_depth, step)
        self.log_scalar("mcts/simulations", simulations, step)
        
        # Log to structured file
        self._log_to_file({
            'type': 'mcts_statistics',
            'stats': {
                'nodes_per_second': nodes_per_second,
                'avg_depth': avg_depth,
                'simulations': simulations
            },
            'step': step,
            'timestamp': datetime.now().isoformat()
        })
    
    def _log_to_file(self, data: Dict[str, Any]) -> None:
        """Log structured data to JSONL file.
        
        Args:
            data: Dictionary to log as JSON
        """
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
    
    def _log_to_training_file(self, message: str) -> None:
        """Log message to training log file.
        
        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.training_log, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def increment_global_step(self) -> None:
        """Increment global step counter."""
        self.global_step += 1
    
    def set_global_step(self, step: int) -> None:
        """Set global step counter.
        
        Args:
            step: New global step value
        """
        self.global_step = step
    
    def get_log_directory(self) -> Path:
        """Get the experiment log directory.
        
        Returns:
            Path to experiment directory
        """
        return self.experiment_dir
    
    def close(self) -> None:
        """Close logger and cleanup resources."""
        if self.tb_writer:
            self.tb_writer.close()
        
        # Write final log entry
        self._log_to_training_file("Logging session ended")
        self._log_to_file({
            'type': 'session_end',
            'timestamp': datetime.now().isoformat()
        })
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()