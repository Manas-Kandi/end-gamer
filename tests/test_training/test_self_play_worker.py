"""Tests for SelfPlayWorker class."""

import pytest
import numpy as np
import torch
import chess

from src.training.self_play_worker import SelfPlayWorker
from src.training.training_example import TrainingExample
from src.config.config import Config
from src.neural_net.chess_net import ChessNet
from src.chess_env.position import Position


class TestSelfPlayWorker:
    """Test cases for SelfPlayWorker."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            mcts_simulations=50,  # Reduced for faster tests
            c_puct=1.0,
            temperature=1.0,
            curriculum_level=0,
            device='cpu',
            random_seed=42
        )
    
    @pytest.fixture
    def neural_net(self):
        """Create test neural network."""
        net = ChessNet(num_res_blocks=1, num_filters=32)  # Smaller for tests
        net.eval()
        return net
    
    @pytest.fixture
    def worker(self, neural_net, config):
        """Create SelfPlayWorker instance."""
        return SelfPlayWorker(neural_net, config)
    
    def test_initialization(self, worker, config):
        """Test worker initialization."""
        assert worker.config == config
        assert worker.mcts.num_simulations == config.mcts_simulations
        assert worker.mcts.c_puct == config.c_puct
        assert worker.position_generator.curriculum_level == config.curriculum_level
    
    def test_play_game_returns_examples(self, worker):
        """Test that play_game returns training examples."""
        examples = worker.play_game()
        
        assert isinstance(examples, list)
        assert len(examples) > 0
        
        for example in examples:
            assert isinstance(example, TrainingExample)
            assert example.position.shape == (8, 8, 12)
            assert example.policy.shape == (4096,)
            assert -1.0 <= example.value <= 1.0
    
    def test_play_game_with_temperature(self, worker):
        """Test play_game with different temperatures."""
        # Test with greedy selection (temperature = 0)
        examples_greedy = worker.play_game(temperature=0.0)
        assert len(examples_greedy) > 0
        
        # Test with high temperature
        examples_random = worker.play_game(temperature=2.0)
        assert len(examples_random) > 0
    
    def test_sample_move_greedy(self, worker):
        """Test greedy move sampling."""
        # Create a simple position
        board = chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1")
        position = Position(board)
        
        # Create mock policy with clear best move
        policy = np.zeros(4096)
        legal_moves = position.get_legal_moves()
        if legal_moves:
            from src.chess_env.move_encoder import MoveEncoder
            best_move_idx = MoveEncoder.encode_move(legal_moves[0])
            policy[best_move_idx] = 1.0
        
        # Test greedy sampling
        selected_move = worker._sample_move(policy, position, temperature=0.0)
        assert selected_move in legal_moves
    
    def test_sample_move_with_temperature(self, worker):
        """Test move sampling with temperature."""
        board = chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1")
        position = Position(board)
        
        # Create uniform policy
        policy = np.zeros(4096)
        legal_moves = position.get_legal_moves()
        from src.chess_env.move_encoder import MoveEncoder
        for move in legal_moves:
            move_idx = MoveEncoder.encode_move(move)
            policy[move_idx] = 1.0 / len(legal_moves)
        
        # Test temperature sampling
        selected_move = worker._sample_move(policy, position, temperature=1.0)
        assert selected_move in legal_moves
    
    def test_play_multiple_games(self, worker):
        """Test playing multiple games."""
        num_games = 3
        all_examples = worker.play_multiple_games(num_games)
        
        assert isinstance(all_examples, list)
        assert len(all_examples) > 0
        
        # Should have examples from multiple games
        # (exact count depends on game length)
        for example in all_examples:
            assert isinstance(example, TrainingExample)
    
    def test_update_curriculum_level(self, worker):
        """Test updating curriculum level."""
        # Test valid levels
        for level in [0, 1, 2]:
            worker.update_curriculum_level(level)
            assert worker.config.curriculum_level == level
            assert worker.position_generator.curriculum_level == level
        
        # Test invalid level
        with pytest.raises(ValueError):
            worker.update_curriculum_level(3)
        
        with pytest.raises(ValueError):
            worker.update_curriculum_level(-1)
    
    def test_set_temperature(self, worker):
        """Test setting temperature."""
        # Test valid temperatures
        for temp in [0.0, 0.5, 1.0, 2.0]:
            worker.set_temperature(temp)
            assert worker.config.temperature == temp
        
        # Test invalid temperature
        with pytest.raises(ValueError):
            worker.set_temperature(-0.1)
    
    def test_get_game_statistics(self, worker):
        """Test getting game statistics."""
        stats = worker.get_game_statistics()
        
        assert isinstance(stats, dict)
        assert 'curriculum_level' in stats
        assert 'temperature' in stats
        assert 'mcts_simulations' in stats
        assert 'c_puct' in stats
        
        assert stats['curriculum_level'] == worker.config.curriculum_level
        assert stats['temperature'] == worker.config.temperature
    
    def test_game_result_assignment(self, worker):
        """Test that game results are correctly assigned to examples."""
        examples = worker.play_game()
        
        if len(examples) > 1:
            # Check that values alternate for different players
            for i in range(len(examples) - 1):
                # Values should be opposite for consecutive moves
                # (since they're from different players' perspectives)
                assert examples[i].value == -examples[i + 1].value or \
                       (examples[i].value == 0.0 and examples[i + 1].value == 0.0)
    
    def test_error_handling_in_multiple_games(self, worker):
        """Test error handling when playing multiple games."""
        # This test ensures the worker continues even if individual games fail
        # We can't easily force an error, so we just test the method works
        examples = worker.play_multiple_games(2)
        assert isinstance(examples, list)
    
    def test_sample_move_edge_cases(self, worker):
        """Test edge cases in move sampling."""
        board = chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1")
        position = Position(board)
        
        # Test with all-zero policy (should fallback to uniform)
        policy = np.zeros(4096)
        legal_moves = position.get_legal_moves()
        
        selected_move = worker._sample_move(policy, position, temperature=1.0)
        assert selected_move in legal_moves
        
        # Test with very small probabilities
        policy = np.full(4096, 1e-10)
        from src.chess_env.move_encoder import MoveEncoder
        for move in legal_moves:
            move_idx = MoveEncoder.encode_move(move)
            policy[move_idx] = 1e-9
        
        selected_move = worker._sample_move(policy, position, temperature=1.0)
        assert selected_move in legal_moves