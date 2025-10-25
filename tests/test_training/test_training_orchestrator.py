"""Unit tests for TrainingOrchestrator class."""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import os

from src.training.training_orchestrator import TrainingOrchestrator
from src.training.training_example import TrainingExample
from src.config.config import Config
import numpy as np


class TestTrainingOrchestrator:
    """Test cases for TrainingOrchestrator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.config = Config(
            # Small values for fast testing
            batch_size=4,
            learning_rate=0.001,
            weight_decay=1e-4,
            target_games=100,
            evaluation_frequency=20,
            games_per_iteration=10,
            training_steps_per_iteration=5,
            num_workers=1,  # Single worker for testing
            buffer_size=50,
            checkpoint_frequency=25,
            device='cpu',
            random_seed=42,
            
            # Use temp directory
            checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints'),
            tensorboard_log_dir=os.path.join(self.temp_dir, 'logs'),
            data_dir=os.path.join(self.temp_dir, 'data'),
            
            # Small network for testing
            num_res_blocks=1,
            num_filters=16
        )
        
        self.orchestrator = TrainingOrchestrator(self.config)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up orchestrator
        if hasattr(self, 'orchestrator'):
            self.orchestrator.cleanup()
        
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator.config == self.config
        assert self.orchestrator.iteration == 0
        assert self.orchestrator.total_games == 0
        assert self.orchestrator.total_training_steps == 0
        assert self.orchestrator.start_time is None
        
        # Check components are initialized
        assert self.orchestrator.neural_net is not None
        assert self.orchestrator.trainer is not None
        assert self.orchestrator.replay_buffer is not None
        assert self.orchestrator.parallel_self_play is not None
        
        # Check directories were created
        assert os.path.exists(self.config.checkpoint_dir)
        assert os.path.exists(self.config.tensorboard_log_dir)
        assert os.path.exists(self.config.data_dir)
    
    def test_create_directories(self):
        """Test directory creation."""
        # Remove directories
        shutil.rmtree(self.temp_dir)
        
        # Create new orchestrator (should recreate directories)
        new_orchestrator = TrainingOrchestrator(self.config)
        
        # Check directories exist
        assert os.path.exists(self.config.checkpoint_dir)
        assert os.path.exists(self.config.tensorboard_log_dir)
        assert os.path.exists(self.config.data_dir)
        assert os.path.exists(os.path.join(self.config.checkpoint_dir, 'best_models'))
        
        new_orchestrator.cleanup()
    
    @patch('src.training.training_orchestrator.ParallelSelfPlay')
    def test_generate_self_play_games(self, mock_parallel_self_play):
        """Test self-play game generation."""
        # Mock parallel self-play
        mock_examples = [
            TrainingExample(
                position=np.random.randn(8, 8, 12),
                policy=np.random.rand(4096),
                value=0.5
            ) for _ in range(20)
        ]
        
        mock_parallel_self_play.return_value.generate_games.return_value = mock_examples
        
        # Create new orchestrator with mocked parallel self-play
        orchestrator = TrainingOrchestrator(self.config)
        
        # Test game generation
        examples = orchestrator._generate_self_play_games(num_games=10, temperature=1.0)
        
        # Verify results
        assert len(examples) == 20
        mock_parallel_self_play.return_value.generate_games.assert_called_once_with(
            total_games=10, temperature=1.0
        )
        
        orchestrator.cleanup()
    
    def test_train_network(self):
        """Test neural network training."""
        # Add some examples to replay buffer
        examples = []
        for _ in range(20):
            example = TrainingExample(
                position=np.random.randn(8, 8, 12),
                policy=np.random.rand(4096),
                value=np.random.uniform(-1, 1)
            )
            examples.append(example)
        
        self.orchestrator.replay_buffer.add_examples(examples)
        
        # Test training
        metrics = self.orchestrator._train_network(num_steps=3)
        
        # Check metrics
        assert isinstance(metrics, dict)
        assert 'avg_total_loss' in metrics
        assert 'avg_policy_loss' in metrics
        assert 'avg_value_loss' in metrics
        assert 'num_steps' in metrics
        
        assert metrics['num_steps'] == 3
        assert metrics['avg_total_loss'] >= 0
        assert metrics['avg_policy_loss'] >= 0
        assert metrics['avg_value_loss'] >= 0
        
        # Check training steps were counted
        assert self.orchestrator.total_training_steps == 3
    
    def test_train_network_insufficient_data(self):
        """Test training with insufficient data in buffer."""
        # Empty buffer - should not be able to sample
        metrics = self.orchestrator._train_network(num_steps=5)
        
        # Should return zero metrics
        assert metrics['num_steps'] == 0
        assert metrics['avg_total_loss'] == 0.0
        assert self.orchestrator.total_training_steps == 0
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Add some examples to buffer for statistics
        examples = []
        for _ in range(10):
            example = TrainingExample(
                position=np.random.randn(8, 8, 12),
                policy=np.random.rand(4096),
                value=np.random.uniform(-1, 1)
            )
            examples.append(example)
        
        self.orchestrator.replay_buffer.add_examples(examples)
        
        # Test evaluation
        metrics = self.orchestrator._evaluate_model()
        
        # Check metrics
        assert isinstance(metrics, dict)
        assert 'buffer_utilization' in metrics
        assert 'avg_value' in metrics
        assert 'value_std' in metrics
        assert 'learning_rate' in metrics
        assert 'total_training_steps' in metrics
        
        assert 0 <= metrics['buffer_utilization'] <= 1
        assert metrics['learning_rate'] > 0
    
    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        # Set some training state
        self.orchestrator.iteration = 5
        self.orchestrator.total_games = 50
        self.orchestrator.total_training_steps = 100
        
        # Save checkpoint
        checkpoint_path = self.orchestrator._save_checkpoint()
        
        # Check file was created
        assert os.path.exists(checkpoint_path)
        
        # Check latest checkpoint was also created
        latest_path = os.path.join(self.config.checkpoint_dir, "latest_checkpoint.pt")
        assert os.path.exists(latest_path)
        
        # Load and verify checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        assert checkpoint['iteration'] == 5
        assert checkpoint['total_games'] == 50
        assert checkpoint['total_training_steps'] == 100
        assert 'model_state_dict' in checkpoint
        assert 'trainer_state_dict' in checkpoint
        assert 'config' in checkpoint
    
    def test_load_checkpoint(self):
        """Test checkpoint loading."""
        # Save initial checkpoint
        self.orchestrator.iteration = 3
        self.orchestrator.total_games = 30
        self.orchestrator.total_training_steps = 60
        checkpoint_path = self.orchestrator._save_checkpoint()
        
        # Create new orchestrator
        new_orchestrator = TrainingOrchestrator(self.config)
        
        # Load checkpoint
        new_orchestrator._load_checkpoint(checkpoint_path)
        
        # Verify state was loaded
        assert new_orchestrator.iteration == 3
        assert new_orchestrator.total_games == 30
        assert new_orchestrator.total_training_steps == 60
        
        new_orchestrator.cleanup()
    
    def test_load_checkpoint_file_not_found(self):
        """Test loading non-existent checkpoint."""
        with pytest.raises(FileNotFoundError):
            self.orchestrator._load_checkpoint("nonexistent_checkpoint.pt")
    
    def test_get_training_info(self):
        """Test getting training information."""
        # Set some state
        self.orchestrator.iteration = 2
        self.orchestrator.total_games = 20
        self.orchestrator.total_training_steps = 40
        self.orchestrator.start_time = 1000.0  # Mock start time
        
        # Add some examples to buffer
        examples = [
            TrainingExample(
                position=np.random.randn(8, 8, 12),
                policy=np.random.rand(4096),
                value=0.5
            ) for _ in range(10)
        ]
        self.orchestrator.replay_buffer.add_examples(examples)
        
        # Get training info
        with patch('time.time', return_value=2000.0):  # Mock current time
            info = self.orchestrator.get_training_info()
        
        # Check info
        assert info['iteration'] == 2
        assert info['total_games'] == 20
        assert info['target_games'] == self.config.target_games
        assert info['progress_pct'] == 20.0  # 20/100 * 100
        assert info['total_training_steps'] == 40
        assert info['elapsed_time_hours'] == (2000.0 - 1000.0) / 3600
        assert info['buffer_size'] == 10
        assert info['buffer_utilization'] == 10 / self.config.buffer_size
        assert 'current_lr' in info
        assert 'curriculum_level' in info
        assert 'temperature' in info
    
    def test_save_and_load_model(self):
        """Test saving and loading just the model."""
        model_path = os.path.join(self.temp_dir, "test_model.pt")
        
        # Set some training state
        self.orchestrator.total_games = 42
        
        # Save model
        self.orchestrator.save_model(model_path)
        
        # Check file exists
        assert os.path.exists(model_path)
        
        # Create new orchestrator and load model
        new_orchestrator = TrainingOrchestrator(self.config)
        new_orchestrator.load_model(model_path)
        
        # Verify model was loaded (check that weights are the same)
        original_params = list(self.orchestrator.neural_net.parameters())
        loaded_params = list(new_orchestrator.neural_net.parameters())
        
        for orig, loaded in zip(original_params, loaded_params):
            assert torch.allclose(orig, loaded)
        
        new_orchestrator.cleanup()
    
    @patch('src.training.training_orchestrator.ParallelSelfPlay')
    def test_train_single_iteration(self, mock_parallel_self_play):
        """Test a single training iteration."""
        # Mock self-play to return examples
        mock_examples = []
        for _ in range(50):  # Enough examples for training
            example = TrainingExample(
                position=np.random.randn(8, 8, 12),
                policy=np.random.rand(4096),
                value=np.random.uniform(-1, 1)
            )
            mock_examples.append(example)
        
        mock_parallel_self_play.return_value.generate_games.return_value = mock_examples
        
        # Create orchestrator with mocked parallel self-play
        orchestrator = TrainingOrchestrator(self.config)
        
        # Mock the training to stop after one iteration
        original_target = orchestrator.config.target_games
        orchestrator.config.target_games = 1  # Will stop after first iteration
        
        # Run training
        orchestrator.train()
        
        # Verify training occurred
        assert orchestrator.iteration > 0
        assert orchestrator.total_training_steps > 0
        assert len(orchestrator.replay_buffer) > 0
        
        # Restore original target
        orchestrator.config.target_games = original_target
        orchestrator.cleanup()
    
    def test_train_with_progress_callback(self):
        """Test training with progress callback."""
        callback_calls = []
        
        def progress_callback(info):
            callback_calls.append(info)
        
        # Mock to stop quickly
        self.orchestrator.config.target_games = 1
        
        with patch.object(self.orchestrator, '_generate_self_play_games') as mock_generate:
            mock_generate.return_value = []  # No examples
            
            # Run training with callback
            self.orchestrator.train(progress_callback=progress_callback)
        
        # Verify callback was called
        assert len(callback_calls) > 0
        
        # Check callback info structure
        info = callback_calls[0]
        assert 'iteration' in info
        assert 'total_games' in info
        assert 'target_games' in info
        assert 'progress_pct' in info
        assert 'iteration_time' in info
    
    def test_train_keyboard_interrupt(self):
        """Test training interruption handling."""
        # Mock to raise KeyboardInterrupt
        with patch.object(self.orchestrator, '_generate_self_play_games') as mock_generate:
            mock_generate.side_effect = KeyboardInterrupt("User interrupt")
            
            # Should handle interrupt gracefully
            self.orchestrator.train()
        
        # Should have saved an interrupted checkpoint
        interrupted_files = [f for f in os.listdir(self.config.checkpoint_dir) 
                           if 'interrupted' in f]
        assert len(interrupted_files) > 0
    
    def test_train_with_exception(self):
        """Test training error handling."""
        # Mock to raise exception
        with patch.object(self.orchestrator, '_generate_self_play_games') as mock_generate:
            mock_generate.side_effect = RuntimeError("Test error")
            
            # Should handle error and re-raise
            with pytest.raises(RuntimeError, match="Test error"):
                self.orchestrator.train()
        
        # Should have saved an error checkpoint
        error_files = [f for f in os.listdir(self.config.checkpoint_dir) 
                      if 'error' in f]
        assert len(error_files) > 0
    
    def test_cleanup(self):
        """Test resource cleanup."""
        # Mock parallel self-play cleanup
        with patch.object(self.orchestrator.parallel_self_play, 'cleanup') as mock_cleanup:
            self.orchestrator.cleanup()
            mock_cleanup.assert_called_once()
    
    def test_curriculum_and_temperature_updates(self):
        """Test curriculum level and temperature updates during training."""
        # Test different game counts
        test_cases = [
            (0, 0, 1.2),      # Phase 1: simple, high exploration
            (10000, 0, 1.2),  # Still phase 1
            (30000, 1, 1.0),  # Phase 2: medium, balanced
            (50000, 1, 1.0),  # Still phase 2
            (80000, 2, 0.8),  # Phase 3: complex, low exploration
        ]
        
        for games, expected_curriculum, expected_temp in test_cases:
            self.orchestrator.total_games = games
            
            curriculum = self.config.get_current_curriculum_level(games)
            temperature = self.config.get_exploration_temperature(games)
            
            assert curriculum == expected_curriculum, f"Games {games}: expected curriculum {expected_curriculum}, got {curriculum}"
            assert temperature == expected_temp, f"Games {games}: expected temperature {expected_temp}, got {temperature}"
    
    def test_metrics_tracking(self):
        """Test metrics history tracking."""
        # Initially empty
        assert len(self.orchestrator.metrics_history) == 0
        
        # Add some mock metrics
        self.orchestrator.metrics_history.append({'loss': 0.5, 'iteration': 1})
        self.orchestrator.metrics_history.append({'loss': 0.4, 'iteration': 2})
        
        # Save and load checkpoint
        checkpoint_path = self.orchestrator._save_checkpoint()
        
        new_orchestrator = TrainingOrchestrator(self.config)
        new_orchestrator._load_checkpoint(checkpoint_path)
        
        # Verify metrics were preserved
        assert len(new_orchestrator.metrics_history) == 2
        assert new_orchestrator.metrics_history[0]['loss'] == 0.5
        assert new_orchestrator.metrics_history[1]['loss'] == 0.4
        
        new_orchestrator.cleanup()
    
    def test_different_config_parameters(self):
        """Test orchestrator with different configuration parameters."""
        # Test with different config
        alt_config = Config(
            batch_size=8,
            learning_rate=0.01,
            num_res_blocks=2,
            num_filters=32,
            device='cpu',
            checkpoint_dir=os.path.join(self.temp_dir, 'alt_checkpoints'),
            tensorboard_log_dir=os.path.join(self.temp_dir, 'alt_logs'),
            data_dir=os.path.join(self.temp_dir, 'alt_data')
        )
        
        alt_orchestrator = TrainingOrchestrator(alt_config)
        
        # Check configuration was applied
        assert alt_orchestrator.config.batch_size == 8
        assert alt_orchestrator.config.learning_rate == 0.01
        assert alt_orchestrator.config.num_res_blocks == 2
        assert alt_orchestrator.config.num_filters == 32
        
        # Check directories were created
        assert os.path.exists(alt_config.checkpoint_dir)
        assert os.path.exists(alt_config.tensorboard_log_dir)
        assert os.path.exists(alt_config.data_dir)
        
        alt_orchestrator.cleanup()