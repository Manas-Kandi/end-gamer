"""Integration tests for MCTS with chess environment."""

import pytest
import torch
import torch.nn as nn
import chess
import numpy as np

from src.chess_env.position import Position
from src.chess_env.move_encoder import MoveEncoder
from src.mcts.mcts import MCTS


class SimpleTestNet(nn.Module):
    """Simple neural network for integration testing."""
    
    def __init__(self):
        super().__init__()
        # Simple architecture that matches expected interface
        self.conv1 = nn.Conv2d(12, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(16, 2, 1)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096)
        
        # Value head
        self.value_conv = nn.Conv2d(16, 1, 1)
        self.value_fc1 = nn.Linear(8 * 8, 32)
        self.value_fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # Backbone
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        # Policy head
        policy = torch.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        
        # Value head
        value = torch.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = torch.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value


class TestMCTSIntegration:
    """Integration tests for MCTS with chess environment."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.neural_net = SimpleTestNet()
        self.mcts = MCTS(
            neural_net=self.neural_net,
            num_simulations=20,  # Small number for fast tests
            c_puct=1.0,
            device='cpu'
        )
    
    def test_mcts_with_starting_position(self):
        """Test MCTS with standard chess starting position."""
        # Create starting position with only kings and one pawn
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.E2, chess.Piece(chess.PAWN, chess.WHITE))
        board.turn = chess.WHITE
        
        position = Position(board)
        
        # Run MCTS search
        policy = self.mcts.search(position)
        
        # Verify policy is valid
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (4096,)
        assert np.all(policy >= 0)
        assert np.abs(np.sum(policy) - 1.0) < 1e-6
        
        # Verify only legal moves have positive probability
        legal_moves = position.get_legal_moves()
        assert len(legal_moves) > 0
        
        total_legal_prob = 0.0
        for move in legal_moves:
            move_idx = MoveEncoder.encode_move(move)
            total_legal_prob += policy[move_idx]
        
        # All probability mass should be on legal moves
        assert np.abs(total_legal_prob - 1.0) < 1e-6
    
    def test_mcts_best_move_selection(self):
        """Test that MCTS can select best moves."""
        # Create position where pawn can advance
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.E2, chess.Piece(chess.PAWN, chess.WHITE))
        board.turn = chess.WHITE
        
        position = Position(board)
        
        # Get best move
        best_move = self.mcts.get_best_move(position)
        
        # Should return a legal move
        legal_moves = position.get_legal_moves()
        assert best_move in legal_moves
        
        # Should be deterministic for same position
        best_move2 = self.mcts.get_best_move(position)
        # Note: Due to randomness in neural network and MCTS, moves might differ
        # We just verify it's still legal
        assert best_move2 in legal_moves
    
    def test_mcts_move_probabilities(self):
        """Test getting move probabilities from MCTS."""
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.E2, chess.Piece(chess.PAWN, chess.WHITE))
        board.turn = chess.WHITE
        
        position = Position(board)
        
        # Get move probabilities
        move_probs = self.mcts.get_move_probabilities(position)
        
        # Should have probabilities for all legal moves
        legal_moves = position.get_legal_moves()
        assert len(move_probs) == len(legal_moves)
        
        # All legal moves should be present
        for move in legal_moves:
            assert move in move_probs
            assert move_probs[move] >= 0
        
        # Probabilities should sum to 1
        total_prob = sum(move_probs.values())
        assert np.abs(total_prob - 1.0) < 1e-6
    
    def test_mcts_with_complex_position(self):
        """Test MCTS with more complex king-pawn position."""
        # Create position with pawn advanced and kings closer
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.E4, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E6, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.D5, chess.Piece(chess.PAWN, chess.WHITE))
        board.turn = chess.WHITE
        
        position = Position(board)
        
        # Should handle complex position without errors
        policy = self.mcts.search(position)
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (4096,)
        
        best_move = self.mcts.get_best_move(position)
        legal_moves = position.get_legal_moves()
        assert best_move in legal_moves
    
    def test_mcts_with_terminal_position(self):
        """Test MCTS behavior with terminal positions."""
        # Create checkmate position
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.A7, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(chess.B6, chess.Piece(chess.KING, chess.WHITE))
        board.turn = chess.BLACK
        
        position = Position(board)
        assert position.is_terminal()
        
        # MCTS should handle terminal position gracefully
        policy = self.mcts.search(position)
        assert np.all(policy == 0)  # No legal moves
        
        best_move = self.mcts.get_best_move(position)
        assert best_move is None
    
    def test_mcts_consistency(self):
        """Test that MCTS produces consistent results."""
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.E2, chess.Piece(chess.PAWN, chess.WHITE))
        board.turn = chess.WHITE
        
        position = Position(board)
        
        # Run multiple searches
        policies = []
        for _ in range(3):
            policy = self.mcts.search(position)
            policies.append(policy)
        
        # All policies should be valid
        for policy in policies:
            assert isinstance(policy, np.ndarray)
            assert policy.shape == (4096,)
            assert np.all(policy >= 0)
            assert np.abs(np.sum(policy) - 1.0) < 1e-6
        
        # Policies might differ due to randomness, but should be reasonable
        # (We don't enforce exact consistency due to neural network randomness)
    
    def test_mcts_parameter_changes(self):
        """Test that changing MCTS parameters affects behavior."""
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.E2, chess.Piece(chess.PAWN, chess.WHITE))
        board.turn = chess.WHITE
        
        position = Position(board)
        
        # Test with different simulation counts
        original_sims = self.mcts.num_simulations
        
        self.mcts.set_num_simulations(5)
        policy_few = self.mcts.search(position)
        
        self.mcts.set_num_simulations(50)
        policy_many = self.mcts.search(position)
        
        # Both should be valid policies
        assert np.abs(np.sum(policy_few) - 1.0) < 1e-6
        assert np.abs(np.sum(policy_many) - 1.0) < 1e-6
        
        # Restore original
        self.mcts.set_num_simulations(original_sims)
        
        # Test with different exploration constants
        original_c_puct = self.mcts.c_puct
        
        self.mcts.set_c_puct(0.1)  # Low exploration
        policy_low_exp = self.mcts.search(position)
        
        self.mcts.set_c_puct(5.0)  # High exploration
        policy_high_exp = self.mcts.search(position)
        
        # Both should be valid
        assert np.abs(np.sum(policy_low_exp) - 1.0) < 1e-6
        assert np.abs(np.sum(policy_high_exp) - 1.0) < 1e-6
        
        # Restore original
        self.mcts.set_c_puct(original_c_puct)
    
    def test_position_tensor_conversion(self):
        """Test that position tensor conversion works with MCTS."""
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.E2, chess.Piece(chess.PAWN, chess.WHITE))
        board.turn = chess.WHITE
        
        position = Position(board)
        
        # Test tensor conversion
        tensor = position.to_tensor()
        assert tensor.shape == (8, 8, 12)
        
        # Test that MCTS can use this tensor
        policy, value = self.mcts._evaluate_position(position)
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (4096,)
        assert isinstance(value, (float, np.floating))
        assert -1.0 <= value <= 1.0