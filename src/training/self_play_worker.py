"""Self-play worker for generating training games."""

import random
from typing import List, Optional
import numpy as np
import chess

from .training_example import TrainingExample
from ..mcts.mcts import MCTS
from ..chess_env.position import Position
from ..chess_env.position_generator import PositionGenerator
from ..chess_env.move_encoder import MoveEncoder
from ..config.config import Config


class SelfPlayWorker:
    """Worker for generating self-play games.
    
    Generates complete games by having the neural network play against itself
    using MCTS for move selection. Collects training examples during play
    and assigns final game results to all positions.
    """
    
    def __init__(self, neural_net, config: Config):
        """Initialize self-play worker.
        
        Args:
            neural_net: Neural network for position evaluation
            config: Configuration object with hyperparameters
        """
        self.neural_net = neural_net
        self.config = config
        
        # Initialize MCTS with neural network
        self.mcts = MCTS(
            neural_net=neural_net,
            num_simulations=config.mcts_simulations,
            c_puct=config.c_puct,
            device=config.device
        )
        
        # Initialize position generator with current curriculum level
        self.position_generator = PositionGenerator(
            curriculum_level=config.curriculum_level
        )
        
        # Set random seed if specified
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
    
    def play_game(self, temperature: Optional[float] = None) -> List[TrainingExample]:
        """Play one complete self-play game and return training examples.
        
        Args:
            temperature: Temperature for move sampling. If None, uses config default
            
        Returns:
            List of training examples from the game
        """
        if temperature is None:
            temperature = self.config.temperature
        
        examples = []
        
        # Generate starting position
        position = self.position_generator.generate_position()
        
        move_count = 0
        max_moves = 200  # Prevent infinite games
        
        while not position.is_terminal() and move_count < max_moves:
            # Get canonical form (always from current player's perspective)
            canonical_position = position.get_canonical_form()
            
            # Run MCTS to get improved policy
            policy = self.mcts.search(canonical_position)
            
            # Store training example (value will be filled in after game ends)
            example = TrainingExample(
                position=canonical_position.to_tensor(),
                policy=policy,
                value=0.0  # Placeholder, will be updated with game result
            )
            examples.append(example)
            
            # Sample move from policy with temperature
            move = self._sample_move(policy, position, temperature)
            
            # Make the move
            position = position.make_move(move)
            move_count += 1
        
        # Get final game result
        if position.is_terminal():
            final_result = position.get_result()
        else:
            # Game exceeded max moves - treat as draw
            final_result = 0.0
        
        # Assign game result to all examples
        # Result alternates for each player (zero-sum game)
        for i, example in enumerate(examples):
            # Even indices (0, 2, 4, ...) are from the perspective of the first player
            # Odd indices (1, 3, 5, ...) are from the perspective of the second player
            if i % 2 == 0:
                example.value = final_result
            else:
                example.value = -final_result
        
        return examples
    
    def _sample_move(self, policy: np.ndarray, position: Position, 
                     temperature: float) -> chess.Move:
        """Sample move from policy with temperature.
        
        Args:
            policy: Policy vector from MCTS (4096,)
            position: Current position
            temperature: Temperature for sampling (0 = greedy, >1 = more random)
            
        Returns:
            Selected chess move
        """
        legal_moves = position.get_legal_moves()
        
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Get probabilities for legal moves
        move_probs = []
        for move in legal_moves:
            move_idx = MoveEncoder.encode_move(move)
            move_probs.append(policy[move_idx])
        
        move_probs = np.array(move_probs)
        
        # Apply temperature
        if temperature == 0.0:
            # Greedy selection - choose move with highest probability
            best_idx = np.argmax(move_probs)
            return legal_moves[best_idx]
        else:
            # Temperature sampling
            # Apply temperature and renormalize
            move_probs = move_probs ** (1.0 / temperature)
            
            # Handle numerical issues
            if np.sum(move_probs) == 0:
                # If all probabilities are zero, sample uniformly
                move_probs = np.ones_like(move_probs)
            
            move_probs = move_probs / np.sum(move_probs)
            
            # Sample from distribution
            try:
                move_idx = np.random.choice(len(legal_moves), p=move_probs)
                return legal_moves[move_idx]
            except ValueError:
                # Fallback to uniform sampling if probabilities are invalid
                return random.choice(legal_moves)
    
    def play_multiple_games(self, num_games: int, 
                           temperature: Optional[float] = None) -> List[TrainingExample]:
        """Play multiple self-play games.
        
        Args:
            num_games: Number of games to play
            temperature: Temperature for move sampling
            
        Returns:
            List of all training examples from all games
        """
        all_examples = []
        
        for game_idx in range(num_games):
            try:
                game_examples = self.play_game(temperature)
                all_examples.extend(game_examples)
            except Exception as e:
                # Log error but continue with other games
                print(f"Error in game {game_idx}: {e}")
                continue
        
        return all_examples
    
    def update_curriculum_level(self, new_level: int) -> None:
        """Update curriculum level for position generation.
        
        Args:
            new_level: New curriculum level (0, 1, or 2)
        """
        if not (0 <= new_level <= 2):
            raise ValueError("Curriculum level must be 0, 1, or 2")
        
        self.position_generator = PositionGenerator(curriculum_level=new_level)
        self.config.curriculum_level = new_level
    
    def set_temperature(self, temperature: float) -> None:
        """Set temperature for move sampling.
        
        Args:
            temperature: New temperature value
        """
        if temperature < 0:
            raise ValueError("Temperature must be non-negative")
        
        self.config.temperature = temperature
    
    def get_game_statistics(self) -> dict:
        """Get statistics about recent games.
        
        Returns:
            Dictionary with game statistics
        """
        return {
            "curriculum_level": self.config.curriculum_level,
            "temperature": self.config.temperature,
            "mcts_simulations": self.config.mcts_simulations,
            "c_puct": self.config.c_puct
        }