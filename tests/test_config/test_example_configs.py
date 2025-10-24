"""Tests for example configuration files."""

import pytest
from pathlib import Path

from src.config.config import Config


class TestExampleConfigs:
    """Test that example configuration files load correctly."""
    
    def test_default_config_loads(self):
        """Test that default.yaml loads without errors."""
        config_path = Path('configs/default.yaml')
        assert config_path.exists(), "default.yaml should exist"
        
        config = Config.from_yaml(str(config_path))
        
        # Verify key values match requirements
        assert config.batch_size == 512
        assert config.learning_rate == 0.001
        assert config.weight_decay == 1e-4
        assert config.mcts_simulations == 400
        assert config.c_puct == 1.0
        assert config.target_games == 100000
        
        # Validate configuration
        config.validate()
    
    def test_quick_test_config_loads(self):
        """Test that quick_test.yaml loads without errors."""
        config_path = Path('configs/quick_test.yaml')
        assert config_path.exists(), "quick_test.yaml should exist"
        
        config = Config.from_yaml(str(config_path))
        
        # Verify reduced parameters for testing
        assert config.batch_size == 64
        assert config.target_games == 1000
        assert config.mcts_simulations == 50
        assert config.num_res_blocks == 2
        assert config.num_filters == 128
        
        # Validate configuration
        config.validate()
    
    def test_full_training_config_loads(self):
        """Test that full_training.yaml loads without errors."""
        config_path = Path('configs/full_training.yaml')
        assert config_path.exists(), "full_training.yaml should exist"
        
        config = Config.from_yaml(str(config_path))
        
        # Verify enhanced parameters for full training
        assert config.batch_size == 1024
        assert config.target_games == 100000
        assert config.mcts_simulations == 800
        assert config.num_res_blocks == 4
        assert config.num_filters == 512
        assert config.mixed_precision == True
        
        # Validate configuration
        config.validate()
    
    def test_configs_have_different_parameters(self):
        """Test that different configs have appropriately different parameters."""
        default_config = Config.from_yaml('configs/default.yaml')
        quick_config = Config.from_yaml('configs/quick_test.yaml')
        full_config = Config.from_yaml('configs/full_training.yaml')
        
        # Quick test should have smaller parameters
        assert quick_config.batch_size < default_config.batch_size
        assert quick_config.target_games < default_config.target_games
        assert quick_config.mcts_simulations < default_config.mcts_simulations
        
        # Full training should have larger parameters
        assert full_config.batch_size > default_config.batch_size
        assert full_config.mcts_simulations > default_config.mcts_simulations
        assert full_config.num_res_blocks >= default_config.num_res_blocks
        assert full_config.num_filters > default_config.num_filters
    
    def test_all_configs_validate(self):
        """Test that all example configs pass validation."""
        config_files = [
            'configs/default.yaml',
            'configs/quick_test.yaml', 
            'configs/full_training.yaml'
        ]
        
        for config_file in config_files:
            config = Config.from_yaml(config_file)
            config.validate()  # Should not raise any exceptions