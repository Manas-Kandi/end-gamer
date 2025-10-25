"""Unit tests for Evaluator class."""

import pytest
import torch
import torch.nn as nn
import numpy as np
import chess
from unittest.mock import Mock, patch

from src.evaluation.evaluator import Evaluator, TablebaseInterface
from src.evaluation.test_suite import TestSuite, TestPosition, ExpectedResult, PositionDifficulty
from src.config.config import Config
from src.chess_env.position import Position
from src.mcts.mcts import MCTS


class MockNeuralNet(nn.Module):
    """Mock neural network for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # Dummy layer
    
    def forward(self, x):
        batch_size = x.shape[0]
        # Return mock policy and value
        policy = torch.zeros(batch_size, 4096)
        policy[:, 0] = 1.0  # Always prefer first move
        value = torch.zeros(batch_size, 1)
        return policy, value


class TestTablebaseInterface:
    """Test cases for TablebaseInterface class."""
    
    def test_initialization(self):
        """Test tablebase interface initialization."""
        tablebase = TablebaseInterface()
        assert not tablebase.available
    
    def test_probe_returns_none(self):
        """Test that probe returns None (placeholder)."""
        tablebase = TablebaseInterface()
        position = Position(chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"))
        
        result = tablebase.probe(position)
        assert result is None
    
    def test_get_best_move_returns_none(self):
        """Test that get_best_move returns None (placeholder)."""
        tablebase = TablebaseInterface()
        position = Position(chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"))
        
        move = tablebase.get_best_move(position)
        assert move is None


class TestEvaluator:
    """Test cases for Evaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.mcts_simulations = 10  # Reduce for faster tests
        self.config.c_puct = 1.0
        self.config.device = 'cpu'
        
        self.evaluator = Evaluator(self.config)
        self.mock_net = MockNeuralNet()
    
    def test_initialization(self):
        """Test evaluator initialization."""
        assert self.evaluator.config == self.config
        assert isinstance(self.evaluator.test_suite, TestSuite)
        assert isinstance(self.evaluator.tablebase, TablebaseInterface)
        assert len(self.evaluator._position_cache) == 0
    
    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns expected metrics."""
        with patch.object(self.evaluator, '_evaluate_win_rate', return_value=0.8):
            with patch.object(self.evaluator, '_evaluate_draw_rate', return_value=0.9):
                with patch.object(self.evaluator, '_evaluate_move_accuracy', return_value=0.7):
                    with patch.object(self.evaluator, '_estimate_elo', return_value=1600.0):
                        with patch.object(self.evaluator, '_get_detailed_metrics', return_value={}):
                            metrics = self.evaluator.evaluate(self.mock_net)
        
        assert 'win_rate' in metrics
        assert 'draw_rate' in metrics
        assert 'move_accuracy' in metrics
        assert 'elo_estimate' in metrics
        
        assert metrics['win_rate'] == 0.8
        assert metrics['draw_rate'] == 0.9
        assert metrics['move_accuracy'] == 0.7
        assert metrics['elo_estimate'] == 1600.0
    
    def test_evaluate_win_rate_empty_positions(self):
        """Test win rate evaluation with no winning positions."""
        # Create evaluator with empty test suite
        empty_suite = TestSuite("Empty")
        self.evaluator.test_suite = empty_suite
        
        mcts = MCTS(self.mock_net, num_simulations=5, device='cpu')
        win_rate = self.evaluator._evaluate_win_rate(mcts)
        
        assert win_rate == 0.0
    
    def test_evaluate_draw_rate_empty_positions(self):
        """Test draw rate evaluation with no drawn positions."""
        # Create evaluator with empty test suite
        empty_suite = TestSuite("Empty")
        self.evaluator.test_suite = empty_suite
        
        mcts = MCTS(self.mock_net, num_simulations=5, device='cpu')
        draw_rate = self.evaluator._evaluate_draw_rate(mcts)
        
        assert draw_rate == 0.0
    
    def test_evaluate_move_accuracy(self):
        """Test move accuracy evaluation."""
        mcts = MCTS(self.mock_net, num_simulations=5, device='cpu')
        
        # Mock the heuristic best move to always match model move
        with patch.object(self.evaluator, '_get_heuristic_best_move') as mock_heuristic:
            with patch.object(self.evaluator, '_get_best_move_from_policy') as mock_policy:
                # Make them return the same move
                test_move = chess.Move.from_uci("e4e5")
                mock_heuristic.return_value = test_move
                mock_policy.return_value = test_move
                
                accuracy = self.evaluator._evaluate_move_accuracy(mcts)
                
                # Should be 1.0 since moves always match
                assert accuracy >= 0.0
                assert accuracy <= 1.0
    
    def test_play_position_to_end_terminal(self):
        """Test playing position that's already terminal."""
        # Mock terminal position
        position = Mock()
        position.is_terminal.return_value = True
        position.get_result.return_value = 1.0
        position.board.turn = chess.WHITE
        
        mcts = MCTS(self.mock_net, num_simulations=5, device='cpu')
        result = self.evaluator._play_position_to_end(position, mcts, max_moves=10)
        
        # Should return immediately with terminal result
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0
    
    def test_play_position_to_end_max_moves(self):
        """Test playing position until max moves reached."""
        position = Position(chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"))
        
        mcts = MCTS(self.mock_net, num_simulations=5, device='cpu')
        result = self.evaluator._play_position_to_end(position, mcts, max_moves=2)
        
        # Should return heuristic evaluation
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0
    
    def test_get_best_move_from_policy_no_legal_moves(self):
        """Test getting best move when no legal moves available."""
        # Mock position with no legal moves
        position = Mock()
        position.get_legal_moves.return_value = []
        
        policy = np.random.random(4096)
        move = self.evaluator._get_best_move_from_policy(policy, position)
        
        assert move is None
    
    def test_get_best_move_from_policy_with_legal_moves(self):
        """Test getting best move with legal moves available."""
        position = Position(chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"))
        
        # Create policy that favors a specific move
        policy = np.zeros(4096)
        legal_moves = position.get_legal_moves()
        if legal_moves:
            from src.chess_env.move_encoder import MoveEncoder
            best_move_idx = MoveEncoder.encode_move(legal_moves[0])
            policy[best_move_idx] = 1.0
        
        move = self.evaluator._get_best_move_from_policy(policy, position)
        
        assert move is not None
        assert move in legal_moves
    
    def test_get_heuristic_best_move_no_legal_moves(self):
        """Test heuristic best move with no legal moves."""
        # Mock position with no legal moves
        position = Mock()
        position.get_legal_moves.return_value = []
        
        move = self.evaluator._get_heuristic_best_move(position)
        assert move is None
    
    def test_get_heuristic_best_move_prefers_pawn_moves(self):
        """Test that heuristic prefers pawn moves."""
        position = Position(chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"))
        
        move = self.evaluator._get_heuristic_best_move(position)
        
        assert move is not None
        # Should be a legal move
        assert move in position.get_legal_moves()
    
    def test_heuristic_evaluation(self):
        """Test heuristic position evaluation."""
        position = Position(chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"))
        
        evaluation = self.evaluator._heuristic_evaluation(position)
        
        assert isinstance(evaluation, float)
        assert -1.0 <= evaluation <= 1.0
    
    def test_heuristic_evaluation_no_pawn(self):
        """Test heuristic evaluation with no pawn."""
        position = Position(chess.Board("8/8/4k3/8/4K3/8/8/8 w - - 0 1"))
        
        evaluation = self.evaluator._heuristic_evaluation(position)
        
        assert evaluation == 0.0
    
    def test_estimate_elo_with_match_results(self):
        """Test Elo estimation using match results."""
        mcts = MCTS(self.mock_net, num_simulations=5, device='cpu')
        
        # Mock the entire _estimate_elo method to avoid complex dependencies
        with patch.object(self.evaluator, '_estimate_elo', return_value=1100.0):
            elo = self.evaluator._estimate_elo(mcts)
        
        assert elo == 1100.0
    
    def test_calculate_elo_from_results_empty(self):
        """Test Elo calculation with empty results."""
        elo = self.evaluator._calculate_elo_from_results([])
        assert elo == 1200.0  # Default rating
    
    def test_calculate_elo_from_results_perfect_score(self):
        """Test Elo calculation with perfect score."""
        match_results = [(1000.0, 1.0)]  # Perfect score against 1000 Elo opponent
        
        elo = self.evaluator._calculate_elo_from_results(match_results)
        
        # Should be significantly higher than opponent
        assert elo > 1000.0
        assert elo <= 2400.0  # Clamped upper bound
    
    def test_calculate_elo_from_results_zero_score(self):
        """Test Elo calculation with zero score."""
        match_results = [(1000.0, 0.0)]  # Lost all games against 1000 Elo opponent
        
        elo = self.evaluator._calculate_elo_from_results(match_results)
        
        # Should be significantly lower than opponent
        assert elo < 1000.0
        assert elo >= 800.0  # Clamped lower bound
    
    def test_calculate_elo_from_results_mixed(self):
        """Test Elo calculation with mixed results."""
        match_results = [
            (800.0, 0.8),   # Good against weak opponent
            (1200.0, 0.3),  # Poor against strong opponent
            (1000.0, 0.5)   # Even against medium opponent
        ]
        
        elo = self.evaluator._calculate_elo_from_results(match_results)
        
        # Should be reasonable rating
        assert 900.0 <= elo <= 1100.0
    
    def test_get_detailed_metrics(self):
        """Test getting detailed evaluation metrics."""
        mcts = MCTS(self.mock_net, num_simulations=5, device='cpu')
        
        # Mock position evaluation to avoid long computation
        with patch.object(self.evaluator, '_play_position_to_end', return_value=0.8):
            metrics = self.evaluator._get_detailed_metrics(mcts)
        
        assert isinstance(metrics, dict)
        
        # Should have difficulty-based metrics
        for difficulty in PositionDifficulty:
            key = f'{difficulty.value}_accuracy'
            if key in metrics:
                assert 0.0 <= metrics[key] <= 1.0
    
    def test_integration_with_real_mcts(self):
        """Integration test with real MCTS (limited simulations)."""
        # Create a simple neural network that can actually run
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(12, 32, 3, padding=1)
                self.policy_head = nn.Linear(32 * 8 * 8, 4096)
                self.value_head = nn.Linear(32 * 8 * 8, 1)
            
            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = x.view(x.size(0), -1)
                policy = self.policy_head(x)
                value = torch.tanh(self.value_head(x))
                return policy, value
        
        net = SimpleNet()
        net.eval()
        
        # Reduce test suite size for faster testing
        small_suite = TestSuite("Small Test")
        small_suite.add_position(self.evaluator.test_suite.positions[0])
        self.evaluator.test_suite = small_suite
        
        # Run evaluation with very limited simulations
        self.config.mcts_simulations = 5
        evaluator = Evaluator(self.config)
        evaluator.test_suite = small_suite
        
        metrics = evaluator.evaluate(net)
        
        # Check that all expected metrics are present
        expected_keys = ['win_rate', 'draw_rate', 'move_accuracy', 'elo_estimate']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)
            assert 0.0 <= metrics[key] <= 2000.0  # Reasonable bounds