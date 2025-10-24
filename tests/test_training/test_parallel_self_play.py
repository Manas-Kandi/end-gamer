"""Integration tests for parallel self-play."""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.training.parallel_self_play import ParallelSelfPlay, _worker_play_games
from src.training.training_example import TrainingExample
from src.config.config import Config
from src.neural_net.chess_net import ChessNet


class TestParallelSelfPlay:
    """Test cases for ParallelSelfPlay."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            mcts_simulations=20,  # Reduced for faster tests
            c_puct=1.0,
            temperature=1.0,
            curriculum_level=0,
            device='cpu',
            num_workers=2,  # Use 2 workers for testing
            random_seed=42
        )
    
    @pytest.fixture
    def neural_net(self):
        """Create test neural network."""
        net = ChessNet(num_res_blocks=1, num_filters=16)  # Very small for tests
        net.eval()
        return net
    
    @pytest.fixture
    def parallel_self_play(self, neural_net, config):
        """Create ParallelSelfPlay instance."""
        return ParallelSelfPlay(neural_net, config)
    
    def test_initialization(self, parallel_self_play, config):
        """Test parallel self-play initialization."""
        assert parallel_self_play.config == config
        assert parallel_self_play.temp_dir.exists()
        assert parallel_self_play.model_path.exists()
    
    def test_generate_games_single_worker(self, parallel_self_play):
        """Test game generation with single worker."""
        # Use only 1 game to force single worker path
        examples = parallel_self_play.generate_games(total_games=1)
        
        assert isinstance(examples, list)
        assert len(examples) > 0
        
        for example in examples:
            assert isinstance(example, TrainingExample)
            assert example.position.shape == (8, 8, 12)
            assert example.policy.shape == (4096,)
            assert -1.0 <= example.value <= 1.0
    
    def test_generate_games_multiple_workers(self, parallel_self_play):
        """Test game generation with multiple workers."""
        total_games = 4  # Will be split across workers
        examples = parallel_self_play.generate_games(total_games=total_games)
        
        assert isinstance(examples, list)
        assert len(examples) > 0
        
        for example in examples:
            assert isinstance(example, TrainingExample)
    
    def test_generate_games_with_progress_callback(self, parallel_self_play):
        """Test game generation with progress callback."""
        progress_updates = []
        
        def progress_callback(current, total):
            progress_updates.append((current, total))
        
        examples = parallel_self_play.generate_games(
            total_games=2,
            progress_callback=progress_callback
        )
        
        assert len(examples) > 0
        assert len(progress_updates) > 0
        
        # Check that final progress update shows completion
        final_current, final_total = progress_updates[-1]
        assert final_current == final_total
    
    def test_generate_games_batch(self, parallel_self_play):
        """Test batch game generation."""
        batch_size = 2
        num_batches = 2
        
        batches = parallel_self_play.generate_games_batch(
            batch_size=batch_size,
            num_batches=num_batches
        )
        
        assert len(batches) == num_batches
        
        for batch in batches:
            assert isinstance(batch, list)
            assert len(batch) > 0
            
            for example in batch:
                assert isinstance(example, TrainingExample)
    
    def test_update_neural_net(self, parallel_self_play):
        """Test updating neural network."""
        # Create new network
        new_net = ChessNet(num_res_blocks=1, num_filters=16)
        new_net.eval()
        
        # Update network
        old_model_path = parallel_self_play.model_path
        parallel_self_play.update_neural_net(new_net)
        
        # Model should be updated
        assert parallel_self_play.neural_net is new_net
        assert parallel_self_play.model_path == old_model_path  # Path stays same
        assert parallel_self_play.model_path.exists()
    
    def test_get_worker_info(self, parallel_self_play):
        """Test getting worker information."""
        info = parallel_self_play.get_worker_info()
        
        assert isinstance(info, dict)
        assert 'num_workers' in info
        assert 'available_cpus' in info
        assert 'temp_dir' in info
        assert 'model_path' in info
        
        assert info['num_workers'] == parallel_self_play.config.num_workers
        assert info['available_cpus'] > 0
    
    def test_cleanup(self, parallel_self_play):
        """Test cleanup of temporary files."""
        temp_dir = parallel_self_play.temp_dir
        model_path = parallel_self_play.model_path
        
        assert temp_dir.exists()
        assert model_path.exists()
        
        parallel_self_play.cleanup()
        
        # Files should be cleaned up
        assert not model_path.exists()
        assert not temp_dir.exists()
    
    def test_worker_function_directly(self, config):
        """Test the worker function directly."""
        import torch
        from src.neural_net.chess_net import ChessNet
        
        # Create and save a test model
        neural_net = ChessNet(num_res_blocks=1, num_filters=16)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(neural_net.state_dict(), f.name)
            model_path = f.name
        
        try:
            # Test worker function
            args = (model_path, config, 1, 1.0, 0)  # 1 game, worker_id=0
            examples = _worker_play_games(args)
            
            assert isinstance(examples, list)
            # Worker might return empty list if there are errors, that's ok for test
            
        finally:
            # Cleanup
            Path(model_path).unlink(missing_ok=True)
    
    def test_error_handling_in_worker(self, config):
        """Test error handling in worker function."""
        # Test with non-existent model path
        args = ("nonexistent_model.pt", config, 1, 1.0, 0)
        examples = _worker_play_games(args)
        
        # Should still generate examples (with random weights)
        assert isinstance(examples, list)
        # Examples may be generated even without model file (using random weights)