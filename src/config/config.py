"""Configuration dataclass for chess engine training and inference."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml
import os
from pathlib import Path


@dataclass
class Config:
    """Configuration for chess engine training and inference.
    
    Contains all hyperparameters for neural network, MCTS, training,
    self-play, and scheduling with sensible defaults matching requirements.
    """
    
    # Neural Network Architecture
    num_res_blocks: int = 3
    num_filters: int = 256
    
    # Training Hyperparameters (Requirement 7.1-7.6)
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    target_games: int = 100000
    evaluation_frequency: int = 1000
    
    # MCTS Configuration (Requirement 7.4-7.5)
    mcts_simulations: int = 400
    c_puct: float = 1.0
    
    # Self-Play Configuration
    games_per_iteration: int = 100
    training_steps_per_iteration: int = 100
    num_workers: int = 4
    temperature: float = 1.0
    
    # Training Schedule (Requirement 7.8-7.10)
    phase_1_games: int = 25000  # High exploration
    phase_2_games: int = 75000  # Balanced exploration/exploitation
    phase_3_games: int = 100000  # Low exploration for fine-tuning
    
    # Curriculum Learning
    curriculum_level: int = 0  # 0: simple, 1: medium, 2: complex
    curriculum_schedule: Dict[int, int] = field(default_factory=lambda: {
        0: 0,      # 0-25K games: simple positions
        25000: 1,  # 25K-75K games: medium positions
        75000: 2   # 75K+ games: complex positions
    })
    
    # Buffer and Storage
    buffer_size: int = 100000
    checkpoint_frequency: int = 5000
    
    # Hardware and Performance
    device: str = 'cuda'
    mixed_precision: bool = False
    num_data_workers: int = 4
    pin_memory: bool = True
    
    # Optimizer Configuration
    optimizer: str = 'adam'
    lr_scheduler: str = 'step'
    lr_step_size: int = 25000
    lr_gamma: float = 0.1
    gradient_clip_norm: float = 1.0
    
    # Evaluation Configuration
    eval_games: int = 100
    eval_mcts_simulations: int = 400
    benchmark_opponents: list = field(default_factory=lambda: [
        'random', 'minimax_d3', 'minimax_d5', 'stockfish'
    ])
    
    # Logging and Monitoring
    log_frequency: int = 100
    tensorboard_log_dir: str = 'logs/tensorboard'
    checkpoint_dir: str = 'checkpoints'
    
    # Reproducibility (Requirement 14.2)
    random_seed: Optional[int] = 42
    
    # File Paths
    config_dir: str = 'configs'
    data_dir: str = 'data'
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config instance with loaded parameters
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Handle None case (empty file)
            if config_dict is None:
                config_dict = {}
                
            return cls(**config_dict)
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            config_path: Path where to save YAML configuration file
            
        Raises:
            OSError: If file cannot be written
        """
        config_path = Path(config_path)
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclass to dictionary
        config_dict = self._to_dict()
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        except OSError as e:
            raise OSError(f"Error writing YAML file {config_path}: {e}")
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary for YAML serialization."""
        config_dict = {}
        
        for field_name, field_value in self.__dict__.items():
            # Handle special cases for serialization
            if isinstance(field_value, Path):
                config_dict[field_name] = str(field_value)
            else:
                config_dict[field_name] = field_value
                
        return config_dict
    
    def get_current_curriculum_level(self, games_played: int) -> int:
        """Get curriculum level based on games played.
        
        Args:
            games_played: Number of games played so far
            
        Returns:
            Current curriculum level (0, 1, or 2)
        """
        current_level = 0
        for threshold, level in sorted(self.curriculum_schedule.items()):
            if games_played >= threshold:
                current_level = level
            else:
                break
        return current_level
    
    def get_exploration_temperature(self, games_played: int) -> float:
        """Get exploration temperature based on training phase.
        
        Args:
            games_played: Number of games played so far
            
        Returns:
            Temperature for move sampling
        """
        if games_played < self.phase_1_games:
            return 1.2  # High exploration
        elif games_played < self.phase_2_games:
            return 1.0  # Balanced
        else:
            return 0.8  # Low exploration for fine-tuning
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        
        if self.mcts_simulations <= 0:
            raise ValueError("mcts_simulations must be positive")
        
        if self.c_puct <= 0:
            raise ValueError("c_puct must be positive")
        
        if self.target_games <= 0:
            raise ValueError("target_games must be positive")
        
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature should be between 0 and 2")
        
        if self.device not in ['cpu', 'cuda', 'mps']:
            raise ValueError("device must be 'cpu', 'cuda', or 'mps'")