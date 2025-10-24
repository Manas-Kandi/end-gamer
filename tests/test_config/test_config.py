"""Tests for Config dataclass."""

import pytest
import tempfile
import os
from pathlib import Path
import yaml

from src.config.config import Config


class TestConfig:
    """Test Config dataclass functionality."""
    
    def test_default_values(self):
        """Test that default values match requirements."""
        config = Config()
        
        # Test requirement 7.1-7.6 defaults
        assert config.batch_size == 512
        assert config.learning_rate == 0.001
        assert config.weight_decay == 1e-4
        assert config.mcts_simulations == 400
        assert config.c_puct == 1.0
        assert config.target_games == 100000
        assert config.evaluation_frequency == 1000
        
        # Test neural network defaults
        assert config.num_res_blocks == 3
        assert config.num_filters == 256
        
        # Test training schedule
        assert config.phase_1_games == 25000
        assert config.phase_2_games == 75000
        assert config.phase_3_games == 100000
    
    def test_curriculum_schedule(self):
        """Test curriculum learning schedule."""
        config = Config()
        
        # Test curriculum level progression
        assert config.get_current_curriculum_level(0) == 0
        assert config.get_current_curriculum_level(10000) == 0
        assert config.get_current_curriculum_level(25000) == 1
        assert config.get_current_curriculum_level(50000) == 1
        assert config.get_current_curriculum_level(75000) == 2
        assert config.get_current_curriculum_level(90000) == 2
    
    def test_exploration_temperature(self):
        """Test exploration temperature based on training phase."""
        config = Config()
        
        # Phase 1: High exploration
        assert config.get_exploration_temperature(10000) == 1.2
        
        # Phase 2: Balanced
        assert config.get_exploration_temperature(50000) == 1.0
        
        # Phase 3: Low exploration
        assert config.get_exploration_temperature(80000) == 0.8
    
    def test_to_yaml(self):
        """Test saving configuration to YAML file."""
        config = Config(
            batch_size=256,
            learning_rate=0.002,
            mcts_simulations=200
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config.to_yaml(temp_path)
            
            # Verify file was created
            assert os.path.exists(temp_path)
            
            # Verify content
            with open(temp_path, 'r') as f:
                saved_data = yaml.safe_load(f)
            
            assert saved_data['batch_size'] == 256
            assert saved_data['learning_rate'] == 0.002
            assert saved_data['mcts_simulations'] == 200
            
        finally:
            os.unlink(temp_path)
    
    def test_from_yaml(self):
        """Test loading configuration from YAML file."""
        config_data = {
            'batch_size': 256,
            'learning_rate': 0.002,
            'mcts_simulations': 200,
            'c_puct': 1.5,
            'target_games': 50000
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = Config.from_yaml(temp_path)
            
            # Test loaded values
            assert config.batch_size == 256
            assert config.learning_rate == 0.002
            assert config.mcts_simulations == 200
            assert config.c_puct == 1.5
            assert config.target_games == 50000
            
            # Test that unspecified values use defaults
            assert config.weight_decay == 1e-4
            assert config.num_res_blocks == 3
            
        finally:
            os.unlink(temp_path)
    
    def test_from_yaml_empty_file(self):
        """Test loading from empty YAML file uses defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Write empty file
            temp_path = f.name
        
        try:
            config = Config.from_yaml(temp_path)
            
            # Should use all defaults
            assert config.batch_size == 512
            assert config.learning_rate == 0.001
            assert config.mcts_simulations == 400
            
        finally:
            os.unlink(temp_path)
    
    def test_from_yaml_nonexistent_file(self):
        """Test loading from nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml('nonexistent_file.yaml')
    
    def test_from_yaml_invalid_yaml(self):
        """Test loading invalid YAML raises YAMLError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content: [')
            temp_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                Config.from_yaml(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_to_yaml_creates_directory(self):
        """Test that to_yaml creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, 'subdir', 'config.yaml')
            config = Config()
            
            config.to_yaml(config_path)
            
            assert os.path.exists(config_path)
            assert os.path.exists(os.path.dirname(config_path))
    
    def test_validate_valid_config(self):
        """Test validation passes for valid configuration."""
        config = Config()
        config.validate()  # Should not raise
    
    def test_validate_invalid_batch_size(self):
        """Test validation fails for invalid batch_size."""
        config = Config(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()
        
        config = Config(batch_size=-1)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()
    
    def test_validate_invalid_learning_rate(self):
        """Test validation fails for invalid learning_rate."""
        config = Config(learning_rate=0)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            config.validate()
        
        config = Config(learning_rate=-0.001)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            config.validate()
    
    def test_validate_invalid_weight_decay(self):
        """Test validation fails for invalid weight_decay."""
        config = Config(weight_decay=-0.1)
        with pytest.raises(ValueError, match="weight_decay must be non-negative"):
            config.validate()
    
    def test_validate_invalid_mcts_simulations(self):
        """Test validation fails for invalid mcts_simulations."""
        config = Config(mcts_simulations=0)
        with pytest.raises(ValueError, match="mcts_simulations must be positive"):
            config.validate()
    
    def test_validate_invalid_c_puct(self):
        """Test validation fails for invalid c_puct."""
        config = Config(c_puct=0)
        with pytest.raises(ValueError, match="c_puct must be positive"):
            config.validate()
    
    def test_validate_invalid_target_games(self):
        """Test validation fails for invalid target_games."""
        config = Config(target_games=0)
        with pytest.raises(ValueError, match="target_games must be positive"):
            config.validate()
    
    def test_validate_invalid_num_workers(self):
        """Test validation fails for invalid num_workers."""
        config = Config(num_workers=0)
        with pytest.raises(ValueError, match="num_workers must be positive"):
            config.validate()
    
    def test_validate_invalid_temperature(self):
        """Test validation fails for invalid temperature."""
        config = Config(temperature=-0.1)
        with pytest.raises(ValueError, match="temperature should be between 0 and 2"):
            config.validate()
        
        config = Config(temperature=2.1)
        with pytest.raises(ValueError, match="temperature should be between 0 and 2"):
            config.validate()
    
    def test_validate_invalid_device(self):
        """Test validation fails for invalid device."""
        config = Config(device='invalid')
        with pytest.raises(ValueError, match="device must be 'cpu', 'cuda', or 'mps'"):
            config.validate()
    
    def test_roundtrip_yaml(self):
        """Test that saving and loading preserves configuration."""
        original_config = Config(
            batch_size=256,
            learning_rate=0.002,
            mcts_simulations=200,
            c_puct=1.5,
            target_games=50000,
            curriculum_schedule={0: 0, 10000: 1, 20000: 2}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save and load
            original_config.to_yaml(temp_path)
            loaded_config = Config.from_yaml(temp_path)
            
            # Compare key fields
            assert loaded_config.batch_size == original_config.batch_size
            assert loaded_config.learning_rate == original_config.learning_rate
            assert loaded_config.mcts_simulations == original_config.mcts_simulations
            assert loaded_config.c_puct == original_config.c_puct
            assert loaded_config.target_games == original_config.target_games
            assert loaded_config.curriculum_schedule == original_config.curriculum_schedule
            
        finally:
            os.unlink(temp_path)