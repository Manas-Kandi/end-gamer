"""Parallel self-play game generation using multiprocessing."""

import multiprocessing as mp
from typing import List, Optional, Callable
import time
import os
import pickle
import tempfile
from pathlib import Path

from .training_example import TrainingExample
from .self_play_worker import SelfPlayWorker
from ..config.config import Config


def _worker_play_games(args) -> List[TrainingExample]:
    """Worker function for multiprocessing.
    
    This function runs in a separate process and generates self-play games.
    
    Args:
        args: Tuple of (neural_net_path, config, num_games, temperature, worker_id)
        
    Returns:
        List of training examples from generated games
    """
    neural_net_path, config, num_games, temperature, worker_id = args
    
    try:
        # Import torch inside worker to avoid serialization issues
        import torch
        from ..neural_net.chess_net import ChessNet
        
        # Load model checkpoint
        if os.path.exists(neural_net_path):
            try:
                checkpoint = torch.load(neural_net_path, map_location=config.device)
                
                # Get architecture parameters from checkpoint if available
                num_res_blocks = checkpoint.get('num_res_blocks', config.num_res_blocks)
                num_filters = checkpoint.get('num_filters', config.num_filters)
                
                # Create neural network with correct architecture
                neural_net = ChessNet(
                    num_res_blocks=num_res_blocks,
                    num_filters=num_filters
                )
                
                # Load model weights
                if 'model_state_dict' in checkpoint:
                    neural_net.load_state_dict(checkpoint['model_state_dict'])
                else:
                    neural_net.load_state_dict(checkpoint)
                    
            except Exception as e:
                print(f"Warning: Could not load model in worker {worker_id}: {e}")
                # Fallback to default architecture with random weights
                neural_net = ChessNet(
                    num_res_blocks=config.num_res_blocks,
                    num_filters=config.num_filters
                )
        else:
            # No model file - use random weights
            neural_net = ChessNet(
                num_res_blocks=config.num_res_blocks,
                num_filters=config.num_filters
            )
        
        neural_net.to(config.device)
        neural_net.eval()
        
        # Create worker and generate games
        worker = SelfPlayWorker(neural_net, config)
        examples = worker.play_multiple_games(num_games, temperature)
        
        return examples
        
    except Exception as e:
        print(f"Error in worker {worker_id}: {e}")
        return []


class ParallelSelfPlay:
    """Manages parallel self-play game generation.
    
    Uses multiprocessing to run multiple SelfPlayWorker instances in parallel,
    aggregating results from all workers.
    """
    
    def __init__(self, neural_net, config: Config):
        """Initialize parallel self-play manager.
        
        Args:
            neural_net: Neural network for self-play
            config: Configuration object
        """
        self.neural_net = neural_net
        self.config = config
        self.temp_dir = Path(tempfile.mkdtemp())
        self.model_path = self.temp_dir / "temp_model.pt"
        
        # Save model for workers to load
        self._save_model_for_workers()
    
    def _save_model_for_workers(self) -> None:
        """Save neural network state for worker processes."""
        import torch
        
        # Save model state dict along with architecture info
        checkpoint = {
            'model_state_dict': self.neural_net.state_dict(),
            'num_res_blocks': self.config.num_res_blocks,
            'num_filters': self.config.num_filters
        }
        torch.save(checkpoint, self.model_path)
    
    def generate_games(self, total_games: int, 
                      temperature: Optional[float] = None,
                      progress_callback: Optional[Callable[[int, int], None]] = None) -> List[TrainingExample]:
        """Generate games using parallel workers.
        
        Args:
            total_games: Total number of games to generate
            temperature: Temperature for move sampling
            progress_callback: Optional callback for progress updates (current, total)
            
        Returns:
            List of all training examples from all games
        """
        if temperature is None:
            temperature = self.config.temperature
        
        num_workers = min(self.config.num_workers, total_games)
        games_per_worker = total_games // num_workers
        remaining_games = total_games % num_workers
        
        # Prepare arguments for each worker
        worker_args = []
        for worker_id in range(num_workers):
            worker_games = games_per_worker
            if worker_id < remaining_games:
                worker_games += 1
            
            if worker_games > 0:
                worker_args.append((
                    str(self.model_path),
                    self.config,
                    worker_games,
                    temperature,
                    worker_id
                ))
        
        # Generate games in parallel
        all_examples = []
        
        if len(worker_args) == 1:
            # Single worker - run directly to avoid multiprocessing overhead
            examples = _worker_play_games(worker_args[0])
            all_examples.extend(examples)
        else:
            # Multiple workers - use multiprocessing
            with mp.Pool(processes=num_workers) as pool:
                # Start async jobs
                results = pool.map_async(_worker_play_games, worker_args)
                
                # Wait for completion with progress updates
                start_time = time.time()
                while not results.ready():
                    if progress_callback:
                        # Estimate progress (rough approximation)
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            estimated_total_time = elapsed * total_games / max(1, len(all_examples))
                            progress = min(elapsed / estimated_total_time, 0.95)
                            progress_callback(int(progress * total_games), total_games)
                    
                    time.sleep(1.0)
                
                # Get results
                worker_results = results.get()
                
                # Aggregate results from all workers
                for examples in worker_results:
                    all_examples.extend(examples)
        
        if progress_callback:
            progress_callback(total_games, total_games)
        
        return all_examples
    
    def generate_games_batch(self, batch_size: int, num_batches: int,
                           temperature: Optional[float] = None,
                           progress_callback: Optional[Callable[[int, int], None]] = None) -> List[List[TrainingExample]]:
        """Generate games in batches for memory management.
        
        Args:
            batch_size: Number of games per batch
            num_batches: Number of batches to generate
            temperature: Temperature for move sampling
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of batches, where each batch is a list of training examples
        """
        batches = []
        
        for batch_idx in range(num_batches):
            # Update model for workers (in case it changed)
            self._save_model_for_workers()
            
            # Generate batch
            batch_examples = self.generate_games(
                batch_size, 
                temperature,
                lambda current, total: progress_callback(
                    batch_idx * batch_size + current,
                    num_batches * batch_size
                ) if progress_callback else None
            )
            
            batches.append(batch_examples)
        
        return batches
    
    def update_neural_net(self, new_neural_net) -> None:
        """Update neural network for future game generation.
        
        Args:
            new_neural_net: Updated neural network
        """
        self.neural_net = new_neural_net
        self._save_model_for_workers()
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        try:
            if self.model_path.exists():
                self.model_path.unlink()
            self.temp_dir.rmdir()
        except Exception:
            pass  # Ignore cleanup errors
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.cleanup()
    
    def get_worker_info(self) -> dict:
        """Get information about worker configuration.
        
        Returns:
            Dictionary with worker information
        """
        return {
            "num_workers": self.config.num_workers,
            "available_cpus": mp.cpu_count(),
            "temp_dir": str(self.temp_dir),
            "model_path": str(self.model_path)
        }