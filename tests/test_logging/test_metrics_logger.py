"""Unit tests for MetricsLogger class."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.logging.metrics_logger import MetricsLogger


class TestMetricsLogger:
    """Test cases for MetricsLogger class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for logs
        self.temp_dir = Path(tempfile.mkdtemp())
        self.log_dir = self.temp_dir / "logs"
        self.experiment_name = "test_experiment"
        
    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization_with_tensorboard(self):
        """Test MetricsLogger initialization with TensorBoard available."""
        with patch('src.logging.metrics_logger.TENSORBOARD_AVAILABLE', True):
            with patch('src.logging.metrics_logger.SummaryWriter') as mock_writer:
                logger = MetricsLogger(str(self.log_dir), self.experiment_name)
                
                # Check directory structure
                assert logger.experiment_dir.exists()
                assert logger.experiment_dir == self.log_dir / self.experiment_name
                
                # Check TensorBoard writer initialization
                mock_writer.assert_called_once()
                
                # Check file creation
                assert logger.metrics_file.exists()
                assert logger.training_log.exists()
                
                # Check step counters
                assert logger.global_step == 0
                assert logger.training_step == 0
                assert logger.evaluation_step == 0
                
                logger.close()
    
    def test_initialization_without_tensorboard(self):
        """Test MetricsLogger initialization without TensorBoard."""
        with patch('src.logging.metrics_logger.TENSORBOARD_AVAILABLE', False):
            logger = MetricsLogger(str(self.log_dir), self.experiment_name)
            
            assert logger.tb_writer is None
            assert logger.experiment_dir.exists()
            
            logger.close()
    
    def test_initialization_auto_experiment_name(self):
        """Test initialization with auto-generated experiment name."""
        with patch('src.logging.metrics_logger.TENSORBOARD_AVAILABLE', False):
            logger = MetricsLogger(str(self.log_dir))
            
            # Should have auto-generated name with timestamp
            assert "experiment_" in logger.experiment_name
            assert logger.experiment_dir.exists()
            
            logger.close()
    
    def test_log_scalar(self):
        """Test logging scalar metrics."""
        with patch('src.logging.metrics_logger.TENSORBOARD_AVAILABLE', True):
            with patch('src.logging.metrics_logger.SummaryWriter') as mock_writer_class:
                mock_writer = Mock()
                mock_writer_class.return_value = mock_writer
                
                logger = MetricsLogger(str(self.log_dir), self.experiment_name)
                
                # Log scalar with explicit step
                logger.log_scalar("test_metric", 0.5, step=10)
                mock_writer.add_scalar.assert_called_with("test_metric", 0.5, 10)
                
                # Log scalar with default step
                logger.log_scalar("test_metric2", 0.8)
                mock_writer.add_scalar.assert_called_with("test_metric2", 0.8, 0)
                
                # Check file logging
                with open(logger.metrics_file, 'r') as f:
                    lines = f.readlines()
                    assert len(lines) == 2
                    
                    # Check first entry
                    entry1 = json.loads(lines[0])
                    assert entry1['type'] == 'scalar'
                    assert entry1['tag'] == 'test_metric'
                    assert entry1['value'] == 0.5
                    assert entry1['step'] == 10
                    
                    # Check second entry
                    entry2 = json.loads(lines[1])
                    assert entry2['type'] == 'scalar'
                    assert entry2['tag'] == 'test_metric2'
                    assert entry2['value'] == 0.8
                    assert entry2['step'] == 0
                
                logger.close()
    
    def test_log_losses(self):
        """Test logging training losses."""
        with patch('src.logging.metrics_logger.TENSORBOARD_AVAILABLE', True):
            with patch('src.logging.metrics_logger.SummaryWriter') as mock_writer_class:
                mock_writer = Mock()
                mock_writer_class.return_value = mock_writer
                
                logger = MetricsLogger(str(self.log_dir), self.experiment_name)
                
                losses = {
                    'total_loss': 1.5,
                    'policy_loss': 0.8,
                    'value_loss': 0.7
                }
                
                # Log losses
                logger.log_losses(losses, step=5)
                
                # Check TensorBoard calls
                expected_calls = [
                    (("loss/total_loss", 1.5, 5),),
                    (("loss/policy_loss", 0.8, 5),),
                    (("loss/value_loss", 0.7, 5),)
                ]
                
                for expected_call in expected_calls:
                    assert expected_call in mock_writer.add_scalar.call_args_list
                
                # Check training step increment
                assert logger.training_step == 1
                
                # Check file logging
                with open(logger.metrics_file, 'r') as f:
                    lines = f.readlines()
                    # Should have 4 lines: 3 individual scalars + 1 combined losses entry
                    assert len(lines) >= 4
                    
                    # Find the losses entry
                    losses_entry = None
                    for line in lines:
                        entry = json.loads(line)
                        if entry['type'] == 'losses':
                            losses_entry = entry
                            break
                    
                    assert losses_entry is not None
                    assert losses_entry['losses'] == losses
                    assert losses_entry['step'] == 5
                
                # Check training log
                with open(logger.training_log, 'r') as f:
                    content = f.read()
                    assert "Step 5: Losses" in content
                
                logger.close()
    
    def test_log_evaluation(self):
        """Test logging evaluation metrics."""
        with patch('src.logging.metrics_logger.TENSORBOARD_AVAILABLE', True):
            with patch('src.logging.metrics_logger.SummaryWriter') as mock_writer_class:
                mock_writer = Mock()
                mock_writer_class.return_value = mock_writer
                
                logger = MetricsLogger(str(self.log_dir), self.experiment_name)
                
                metrics = {
                    'win_rate': 0.85,
                    'draw_rate': 0.90,
                    'elo_estimate': 1800.0
                }
                
                # Log evaluation
                logger.log_evaluation(metrics, step=3)
                
                # Check TensorBoard calls
                expected_calls = [
                    (("eval/win_rate", 0.85, 3),),
                    (("eval/draw_rate", 0.90, 3),),
                    (("eval/elo_estimate", 1800.0, 3),)
                ]
                
                for expected_call in expected_calls:
                    assert expected_call in mock_writer.add_scalar.call_args_list
                
                # Check evaluation step increment
                assert logger.evaluation_step == 1
                
                # Check file logging
                with open(logger.metrics_file, 'r') as f:
                    lines = f.readlines()
                    
                    # Find the evaluation entry
                    eval_entry = None
                    for line in lines:
                        entry = json.loads(line)
                        if entry['type'] == 'evaluation':
                            eval_entry = entry
                            break
                    
                    assert eval_entry is not None
                    assert eval_entry['metrics'] == metrics
                    assert eval_entry['step'] == 3
                
                logger.close()
    
    def test_log_hyperparameters(self):
        """Test logging hyperparameters."""
        with patch('src.logging.metrics_logger.TENSORBOARD_AVAILABLE', True):
            with patch('src.logging.metrics_logger.SummaryWriter') as mock_writer_class:
                mock_writer = Mock()
                mock_writer_class.return_value = mock_writer
                
                logger = MetricsLogger(str(self.log_dir), self.experiment_name)
                
                hparams = {
                    'learning_rate': 0.001,
                    'batch_size': 512,
                    'num_res_blocks': 3,
                    'device': 'cuda'
                }
                
                # Log hyperparameters
                logger.log_hyperparameters(hparams)
                
                # Check TensorBoard call
                mock_writer.add_hparams.assert_called_once()
                call_args = mock_writer.add_hparams.call_args[0]
                
                # Check that numeric values are preserved and strings are converted
                expected_scalar_hparams = {
                    'learning_rate': 0.001,
                    'batch_size': 512,
                    'num_res_blocks': 3,
                    'device': 'cuda'  # String values are kept as strings
                }
                assert call_args[0] == expected_scalar_hparams
                
                # Check file logging
                with open(logger.metrics_file, 'r') as f:
                    lines = f.readlines()
                    
                    hparams_entry = None
                    for line in lines:
                        entry = json.loads(line)
                        if entry['type'] == 'hyperparameters':
                            hparams_entry = entry
                            break
                    
                    assert hparams_entry is not None
                    assert hparams_entry['hparams'] == hparams
                
                logger.close()
    
    def test_log_training_progress(self):
        """Test logging training progress."""
        with patch('src.logging.metrics_logger.TENSORBOARD_AVAILABLE', False):
            logger = MetricsLogger(str(self.log_dir), self.experiment_name)
            
            # Log training progress
            logger.log_training_progress(
                iteration=5,
                total_games=5000,
                games_per_iteration=1000,
                message="Checkpoint saved"
            )
            
            # Check file logging
            with open(logger.metrics_file, 'r') as f:
                lines = f.readlines()
                
                # Find progress entry
                progress_entry = None
                for line in lines:
                    entry = json.loads(line)
                    if entry['type'] == 'training_progress':
                        progress_entry = entry
                        break
                
                assert progress_entry is not None
                assert progress_entry['progress']['iteration'] == 5
                assert progress_entry['progress']['total_games'] == 5000
                assert progress_entry['progress']['progress_percent'] == 5.0
                assert progress_entry['message'] == "Checkpoint saved"
            
            # Check training log
            with open(logger.training_log, 'r') as f:
                content = f.read()
                assert "Iteration 5: 5000 games played (5.0%)" in content
                assert "Checkpoint saved" in content
            
            logger.close()
    
    def test_log_resource_usage(self):
        """Test logging resource usage."""
        with patch('src.logging.metrics_logger.TENSORBOARD_AVAILABLE', False):
            logger = MetricsLogger(str(self.log_dir), self.experiment_name)
            
            # Log resource usage with GPU
            logger.log_resource_usage(
                cpu_percent=75.5,
                memory_mb=8192.0,
                gpu_memory_mb=4096.0
            )
            
            # Check file logging
            with open(logger.metrics_file, 'r') as f:
                lines = f.readlines()
                
                resource_entry = None
                for line in lines:
                    entry = json.loads(line)
                    if entry['type'] == 'resource_usage':
                        resource_entry = entry
                        break
                
                assert resource_entry is not None
                assert resource_entry['resources']['cpu_percent'] == 75.5
                assert resource_entry['resources']['memory_mb'] == 8192.0
                assert resource_entry['resources']['gpu_memory_mb'] == 4096.0
            
            # Log resource usage without GPU
            logger.log_resource_usage(cpu_percent=50.0, memory_mb=4096.0)
            
            with open(logger.metrics_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) >= 2
            
            logger.close()
    
    def test_log_mcts_statistics(self):
        """Test logging MCTS statistics."""
        with patch('src.logging.metrics_logger.TENSORBOARD_AVAILABLE', False):
            logger = MetricsLogger(str(self.log_dir), self.experiment_name)
            
            # Log MCTS stats
            logger.log_mcts_statistics(
                nodes_per_second=1500.0,
                avg_depth=12.5,
                simulations=400
            )
            
            # Check file logging
            with open(logger.metrics_file, 'r') as f:
                lines = f.readlines()
                
                mcts_entry = None
                for line in lines:
                    entry = json.loads(line)
                    if entry['type'] == 'mcts_statistics':
                        mcts_entry = entry
                        break
                
                assert mcts_entry is not None
                assert mcts_entry['stats']['nodes_per_second'] == 1500.0
                assert mcts_entry['stats']['avg_depth'] == 12.5
                assert mcts_entry['stats']['simulations'] == 400
            
            logger.close()
    
    def test_step_management(self):
        """Test step counter management."""
        with patch('src.logging.metrics_logger.TENSORBOARD_AVAILABLE', False):
            logger = MetricsLogger(str(self.log_dir), self.experiment_name)
            
            # Test initial values
            assert logger.global_step == 0
            assert logger.training_step == 0
            assert logger.evaluation_step == 0
            
            # Test increment
            logger.increment_global_step()
            assert logger.global_step == 1
            
            # Test set
            logger.set_global_step(100)
            assert logger.global_step == 100
            
            # Test automatic increment in log_losses
            logger.log_losses({'loss': 1.0})
            assert logger.training_step == 1
            
            # Test automatic increment in log_evaluation
            logger.log_evaluation({'metric': 0.5})
            assert logger.evaluation_step == 1
            
            logger.close()
    
    def test_context_manager(self):
        """Test using logger as context manager."""
        with patch('src.logging.metrics_logger.TENSORBOARD_AVAILABLE', True):
            with patch('src.logging.metrics_logger.SummaryWriter') as mock_writer_class:
                mock_writer = Mock()
                mock_writer_class.return_value = mock_writer
                
                with MetricsLogger(str(self.log_dir), self.experiment_name) as logger:
                    logger.log_scalar("test", 1.0)
                    assert logger.experiment_dir.exists()
                
                # Should have called close
                mock_writer.close.assert_called_once()
    
    def test_get_log_directory(self):
        """Test getting log directory."""
        with patch('src.logging.metrics_logger.TENSORBOARD_AVAILABLE', False):
            logger = MetricsLogger(str(self.log_dir), self.experiment_name)
            
            log_dir = logger.get_log_directory()
            assert log_dir == logger.experiment_dir
            assert log_dir.exists()
            
            logger.close()
    
    def test_file_logging_format(self):
        """Test that file logging produces valid JSON."""
        with patch('src.logging.metrics_logger.TENSORBOARD_AVAILABLE', False):
            logger = MetricsLogger(str(self.log_dir), self.experiment_name)
            
            # Log various types of data
            logger.log_scalar("metric1", 1.0)
            logger.log_losses({'loss1': 0.5, 'loss2': 0.3})
            logger.log_evaluation({'eval1': 0.8})
            logger.log_hyperparameters({'param1': 'value1'})
            
            # Read and validate all entries
            with open(logger.metrics_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        entry = json.loads(line)  # Should not raise exception
                        assert 'type' in entry
                        assert 'timestamp' in entry
            
            logger.close()
    
    def test_training_log_format(self):
        """Test training log format."""
        with patch('src.logging.metrics_logger.TENSORBOARD_AVAILABLE', False):
            logger = MetricsLogger(str(self.log_dir), self.experiment_name)
            
            # Check initial header
            with open(logger.training_log, 'r') as f:
                content = f.read()
                assert f"Training Log - {self.experiment_name}" in content
                assert "Started:" in content
            
            # Log some data
            logger.log_losses({'loss': 1.0})
            logger.log_training_progress(1, 1000, 1000)
            
            # Check entries have timestamps
            with open(logger.training_log, 'r') as f:
                lines = f.readlines()
                
                # Find log entries (skip header)
                log_entries = [line for line in lines if line.startswith('[')]
                assert len(log_entries) >= 2
                
                # Check timestamp format
                for entry in log_entries:
                    assert entry.startswith('[20')  # Year starts with 20
                    assert ']' in entry
            
            logger.close()
    
    def test_without_tensorboard_import_error(self):
        """Test behavior when TensorBoard import fails."""
        # This test simulates the case where tensorboard is not installed
        with patch('src.logging.metrics_logger.TENSORBOARD_AVAILABLE', False):
            with patch('src.logging.metrics_logger.SummaryWriter', None):
                logger = MetricsLogger(str(self.log_dir), self.experiment_name)
                
                assert logger.tb_writer is None
                
                # Should still work for file logging
                logger.log_scalar("test", 1.0)
                
                with open(logger.metrics_file, 'r') as f:
                    line = f.readline().strip()
                    entry = json.loads(line)
                    assert entry['tag'] == 'test'
                    assert entry['value'] == 1.0
                
                logger.close()