"""Training orchestrator for coordinating the complete training pipeline."""

import os
import time
from typing import Dict, List, Optional, Callable
from pathlib import Path
import torch
import torch.nn as nn
from datetime import datetime

from .trainer import Trainer
from .replay_buffer import ReplayBuffer
from .parallel_self_play import ParallelSelfPlay
from .training_example import TrainingExample
from ..neural_net.chess_net import ChessNet
from ..config.config import Config


class TrainingOrchestrator:
    """Coordinates the complete training pipeline.
    
    Manages the training loop including self-play generation, neural network
    training, evaluation, and checkpointing. Integrates all training components
    and handles the overall training schedule.
    """
    
    def __init__(self, config: Config):
        """Initialize training orchestrator.
        
        Args:
            config: Configuration object with all hyperparameters
        """
        self.config = config
        
        # Initialize neural network
        self.neural_net = ChessNet(
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters
        ).to(config.device)
        
        # Initialize trainer
        self.trainer = Trainer(self.neural_net, config)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            max_size=config.buffer_size,
            seed=config.random_seed
        )
        
        # Initialize parallel self-play
        self.parallel_self_play = ParallelSelfPlay(self.neural_net, config)
        
        # Training state
        self.iteration = 0
        self.total_games = 0
        self.total_training_steps = 0
        self.start_time = None
        
        # Create directories
        self._create_directories()
        
        # Metrics tracking
        self.metrics_history = []
        
    def _create_directories(self) -> None:
        """Create necessary directories for training."""
        directories = [
            self.config.checkpoint_dir,
            self.config.tensorboard_log_dir,
            self.config.data_dir,
            Path(self.config.checkpoint_dir) / "best_models"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def train(self, resume_from_checkpoint: Optional[str] = None,
              progress_callback: Optional[Callable[[Dict], None]] = None) -> None:
        """Main training loop.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
            progress_callback: Optional callback for progress updates
        """
        print("Starting training...")
        self.start_time = time.time()
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
            print(f"Resumed from checkpoint: {resume_from_checkpoint}")
            print(f"Starting from iteration {self.iteration}, {self.total_games} games")
        
        try:
            while self.total_games < self.config.target_games:
                iteration_start_time = time.time()
                
                print(f"\n=== Iteration {self.iteration} ===")
                print(f"Games played: {self.total_games}/{self.config.target_games}")
                
                # Update curriculum level
                current_curriculum = self.config.get_current_curriculum_level(self.total_games)
                current_temperature = self.config.get_exploration_temperature(self.total_games)
                
                print(f"Curriculum level: {current_curriculum}")
                print(f"Temperature: {current_temperature:.2f}")
                
                # Self-play phase
                print("Generating self-play games...")
                new_examples = self._generate_self_play_games(
                    num_games=self.config.games_per_iteration,
                    temperature=current_temperature
                )
                
                # Add examples to replay buffer
                self.replay_buffer.add_examples(new_examples)
                self.total_games += len(new_examples) // 20  # Approximate games (20 examples per game)
                
                print(f"Generated {len(new_examples)} training examples")
                print(f"Replay buffer size: {len(self.replay_buffer)}")
                
                # Training phase
                if len(self.replay_buffer) >= self.config.batch_size:
                    print("Training neural network...")
                    training_metrics = self._train_network(
                        num_steps=self.config.training_steps_per_iteration
                    )
                    
                    # Update parallel self-play with new model
                    self.parallel_self_play.update_neural_net(self.neural_net)
                    
                    print(f"Training loss: {training_metrics['avg_total_loss']:.4f}")
                
                # Evaluation phase
                if self.total_games % self.config.evaluation_frequency == 0 and self.total_games > 0:
                    print("Evaluating model...")
                    eval_metrics = self._evaluate_model()
                    print(f"Evaluation metrics: {eval_metrics}")
                
                # Checkpoint saving
                if self.total_games % self.config.checkpoint_frequency == 0 and self.total_games > 0:
                    checkpoint_path = self._save_checkpoint()
                    print(f"Saved checkpoint: {checkpoint_path}")
                
                # Progress callback
                if progress_callback:
                    iteration_time = time.time() - iteration_start_time
                    progress_info = {
                        'iteration': self.iteration,
                        'total_games': self.total_games,
                        'target_games': self.config.target_games,
                        'progress_pct': (self.total_games / self.config.target_games) * 100,
                        'iteration_time': iteration_time,
                        'buffer_size': len(self.replay_buffer),
                        'curriculum_level': current_curriculum,
                        'temperature': current_temperature
                    }
                    progress_callback(progress_info)
                
                self.iteration += 1
                
                # Print iteration summary
                iteration_time = time.time() - iteration_start_time
                print(f"Iteration {self.iteration-1} completed in {iteration_time:.1f}s")
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            checkpoint_path = self._save_checkpoint(suffix="_interrupted")
            print(f"Saved checkpoint: {checkpoint_path}")
        
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            checkpoint_path = self._save_checkpoint(suffix="_error")
            print(f"Saved checkpoint: {checkpoint_path}")
            raise
        
        finally:
            # Cleanup
            self.parallel_self_play.cleanup()
            
            if self.start_time:
                total_time = time.time() - self.start_time
                print(f"\nTotal training time: {total_time/3600:.1f} hours")
        
        print(f"\nTraining completed! Final games: {self.total_games}")
        
        # Save final checkpoint
        final_checkpoint = self._save_checkpoint(suffix="_final")
        print(f"Final checkpoint saved: {final_checkpoint}")
    
    def _generate_self_play_games(self, num_games: int, 
                                 temperature: float) -> List[TrainingExample]:
        """Generate self-play games using parallel workers.
        
        Args:
            num_games: Number of games to generate
            temperature: Temperature for move sampling
            
        Returns:
            List of training examples from all games
        """
        # Update curriculum level in config for workers
        current_curriculum = self.config.get_current_curriculum_level(self.total_games)
        self.config.curriculum_level = current_curriculum
        
        # Generate games
        examples = self.parallel_self_play.generate_games(
            total_games=num_games,
            temperature=temperature
        )
        
        return examples
    
    def _train_network(self, num_steps: int) -> Dict[str, float]:
        """Train neural network for specified number of steps.
        
        Args:
            num_steps: Number of training steps to perform
            
        Returns:
            Dictionary with average training metrics
        """
        total_losses = []
        policy_losses = []
        value_losses = []
        
        for step in range(num_steps):
            # Sample batch from replay buffer
            try:
                positions, policies, values = self.replay_buffer.sample_batch(
                    self.config.batch_size
                )
            except ValueError as e:
                print(f"Warning: Could not sample batch: {e}")
                break
            
            # Training step
            losses = self.trainer.train_step(positions, policies, values)
            
            # Collect metrics
            total_losses.append(losses['total_loss'])
            policy_losses.append(losses['policy_loss'])
            value_losses.append(losses['value_loss'])
            
            self.total_training_steps += 1
            
            # Log progress
            if step % self.config.log_frequency == 0:
                print(f"  Step {step}/{num_steps}: Loss = {losses['total_loss']:.4f}")
        
        # Return average metrics
        if total_losses:
            return {
                'avg_total_loss': sum(total_losses) / len(total_losses),
                'avg_policy_loss': sum(policy_losses) / len(policy_losses),
                'avg_value_loss': sum(value_losses) / len(value_losses),
                'num_steps': len(total_losses)
            }
        else:
            return {
                'avg_total_loss': 0.0,
                'avg_policy_loss': 0.0,
                'avg_value_loss': 0.0,
                'num_steps': 0
            }
    
    def _evaluate_model(self) -> Dict[str, float]:
        """Evaluate current model performance.
        
        Returns:
            Dictionary with evaluation metrics
        """
        # For now, return basic metrics
        # TODO: Implement comprehensive evaluation when Evaluator class is available
        
        buffer_stats = self.replay_buffer.get_statistics()
        
        return {
            'buffer_utilization': buffer_stats['utilization'],
            'avg_value': buffer_stats['avg_value'],
            'value_std': buffer_stats['value_std'],
            'learning_rate': self.trainer.get_learning_rate(),
            'total_training_steps': self.total_training_steps
        }
    
    def _save_checkpoint(self, suffix: str = "") -> str:
        """Save training checkpoint.
        
        Args:
            suffix: Optional suffix for checkpoint filename
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{self.total_games}_{timestamp}{suffix}.pt"
        checkpoint_path = Path(self.config.checkpoint_dir) / filename
        
        # Prepare checkpoint data
        checkpoint = {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'total_training_steps': self.total_training_steps,
            'model_state_dict': self.neural_net.state_dict(),
            'trainer_state_dict': self.trainer.get_state_dict(),
            'config': self.config,
            'metrics_history': self.metrics_history,
            'timestamp': timestamp,
            
            # Model architecture info
            'num_res_blocks': self.config.num_res_blocks,
            'num_filters': self.config.num_filters,
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest checkpoint
        latest_path = Path(self.config.checkpoint_dir) / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
        
        return str(checkpoint_path)
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
            
            # Load training state
            self.iteration = checkpoint.get('iteration', 0)
            self.total_games = checkpoint.get('total_games', 0)
            self.total_training_steps = checkpoint.get('total_training_steps', 0)
            self.metrics_history = checkpoint.get('metrics_history', [])
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.neural_net.load_state_dict(checkpoint['model_state_dict'])
            
            # Load trainer state
            if 'trainer_state_dict' in checkpoint:
                self.trainer.load_state_dict(checkpoint['trainer_state_dict'])
            
            # Update parallel self-play with loaded model
            self.parallel_self_play.update_neural_net(self.neural_net)
            
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    def get_training_info(self) -> Dict:
        """Get current training information.
        
        Returns:
            Dictionary with training status and metrics
        """
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'target_games': self.config.target_games,
            'progress_pct': (self.total_games / self.config.target_games) * 100,
            'total_training_steps': self.total_training_steps,
            'elapsed_time_hours': elapsed_time / 3600,
            'buffer_size': len(self.replay_buffer),
            'buffer_utilization': len(self.replay_buffer) / self.config.buffer_size,
            'current_lr': self.trainer.get_learning_rate(),
            'curriculum_level': self.config.get_current_curriculum_level(self.total_games),
            'temperature': self.config.get_exploration_temperature(self.total_games)
        }
    
    def save_model(self, filepath: str) -> None:
        """Save just the neural network model.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'model_state_dict': self.neural_net.state_dict(),
            'num_res_blocks': self.config.num_res_blocks,
            'num_filters': self.config.num_filters,
            'total_games': self.total_games,
            'config': self.config
        }
        
        torch.save(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load neural network model.
        
        Args:
            filepath: Path to model file
        """
        model_data = torch.load(filepath, map_location=self.config.device)
        
        # Load model weights
        if 'model_state_dict' in model_data:
            self.neural_net.load_state_dict(model_data['model_state_dict'])
        
        # Update parallel self-play
        self.parallel_self_play.update_neural_net(self.neural_net)
        
        print(f"Model loaded from {filepath}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'parallel_self_play'):
            self.parallel_self_play.cleanup()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.cleanup()