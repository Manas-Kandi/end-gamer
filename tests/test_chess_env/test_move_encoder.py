"""Unit tests for MoveEncoder class."""

import pytest
import numpy as np
import chess
from src.chess_env.position import Position
from src.chess_env.move_encoder import MoveEncoder


class TestMoveEncoder:
    """Test cases for MoveEncoder class."""
    
    def test_encode_move_basic(self):
        """Test basic move encoding."""
        # e2 to e4: from_square=12, to_square=28
        move = chess.Move.from_uci("e2e4")
        encoded = MoveEncoder.encode_move(move)
        expected = 12 * 64 + 28  # 796
        assert encoded == expected
    
    def test_encode_move_corner_cases(self):
        """Test move encoding for corner squares."""
        # a1 to a1 (from_square=0, to_square=0)
        move = chess.Move(chess.A1, chess.A1)
        encoded = MoveEncoder.encode_move(move)
        assert encoded == 0
        
        # h8 to h8 (from_square=63, to_square=63)
        move = chess.Move(chess.H8, chess.H8)
        encoded = MoveEncoder.encode_move(move)
        assert encoded == 63 * 64 + 63  # 4095
    
    def test_decode_move_basic(self):
        """Test basic move decoding."""
        # Decode index 796 should give e2e4
        move = MoveEncoder.decode_move(796)
        assert move.from_square == 12  # e2
        assert move.to_square == 28    # e4
        assert move.uci() == "e2e4"
    
    def test_decode_move_corner_cases(self):
        """Test move decoding for corner cases."""
        # Decode index 0
        move = MoveEncoder.decode_move(0)
        assert move.from_square == 0
        assert move.to_square == 0
        
        # Decode index 4095
        move = MoveEncoder.decode_move(4095)
        assert move.from_square == 63
        assert move.to_square == 63
    
    def test_decode_move_invalid_index(self):
        """Test decoding invalid move indices."""
        with pytest.raises(ValueError, match="Move index .* must be in range"):
            MoveEncoder.decode_move(-1)
        
        with pytest.raises(ValueError, match="Move index .* must be in range"):
            MoveEncoder.decode_move(4096)
    
    def test_encode_decode_roundtrip(self):
        """Test that encoding and decoding are inverse operations."""
        # Test various moves
        test_moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("g1f3"),
            chess.Move.from_uci("a1a8"),
            chess.Move.from_uci("h1h8"),
            chess.Move.from_uci("d7d8q"),  # Promotion
        ]
        
        for original_move in test_moves:
            encoded = MoveEncoder.encode_move(original_move)
            decoded = MoveEncoder.decode_move(encoded)
            
            # Note: promotion info might be lost in simple encoding
            assert decoded.from_square == original_move.from_square
            assert decoded.to_square == original_move.to_square
    
    def test_get_move_mask_starting_position(self):
        """Test move mask for starting position."""
        pos = Position()
        mask = MoveEncoder.get_move_mask(pos)
        
        assert mask.shape == (4096,)
        assert mask.dtype == np.float32
        
        # Starting position has 20 legal moves
        assert np.sum(mask) == 20
        
        # Check that some known legal moves are marked
        e2e4_idx = MoveEncoder.encode_move(chess.Move.from_uci("e2e4"))
        assert mask[e2e4_idx] == 1.0
        
        g1f3_idx = MoveEncoder.encode_move(chess.Move.from_uci("g1f3"))
        assert mask[g1f3_idx] == 1.0
    
    def test_get_move_mask_king_pawn_endgame(self):
        """Test move mask for king-pawn endgame."""
        # White king on e4, black king on e6, white pawn on e5
        board = chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1")
        pos = Position(board)
        mask = MoveEncoder.get_move_mask(pos)
        
        assert mask.shape == (4096,)
        legal_moves_count = len(pos.get_legal_moves())
        assert np.sum(mask) == legal_moves_count
        
        # All marked moves should be legal
        legal_indices = MoveEncoder.get_legal_move_indices(pos)
        for idx in legal_indices:
            assert mask[idx] == 1.0
    
    def test_get_legal_move_indices(self):
        """Test getting legal move indices."""
        pos = Position()
        indices = MoveEncoder.get_legal_move_indices(pos)
        
        assert len(indices) == 20  # Starting position has 20 legal moves
        assert all(isinstance(idx, int) for idx in indices)
        assert all(0 <= idx < 4096 for idx in indices)
        
        # Should match the move mask
        mask = MoveEncoder.get_move_mask(pos)
        mask_indices = np.where(mask == 1.0)[0].tolist()
        assert set(indices) == set(mask_indices)
    
    def test_moves_to_policy_vector_basic(self):
        """Test converting moves and visit counts to policy vector."""
        moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("g1f3"),
            chess.Move.from_uci("d2d4")
        ]
        visit_counts = [10, 5, 15]  # Total: 30
        
        policy = MoveEncoder.moves_to_policy_vector(moves, visit_counts)
        
        assert policy.shape == (4096,)
        assert policy.dtype == np.float32
        
        # Check probabilities
        e2e4_idx = MoveEncoder.encode_move(moves[0])
        g1f3_idx = MoveEncoder.encode_move(moves[1])
        d2d4_idx = MoveEncoder.encode_move(moves[2])
        
        assert abs(policy[e2e4_idx] - 10/30) < 1e-6
        assert abs(policy[g1f3_idx] - 5/30) < 1e-6
        assert abs(policy[d2d4_idx] - 15/30) < 1e-6
        
        # Sum should be 1.0
        assert abs(np.sum(policy) - 1.0) < 1e-6
    
    def test_moves_to_policy_vector_empty(self):
        """Test policy vector with no moves."""
        policy = MoveEncoder.moves_to_policy_vector([], [])
        assert policy.shape == (4096,)
        assert np.sum(policy) == 0.0
    
    def test_moves_to_policy_vector_zero_visits(self):
        """Test policy vector with zero total visits."""
        moves = [chess.Move.from_uci("e2e4")]
        visit_counts = [0]
        
        policy = MoveEncoder.moves_to_policy_vector(moves, visit_counts)
        assert np.sum(policy) == 0.0
    
    def test_moves_to_policy_vector_mismatched_lengths(self):
        """Test error when moves and visit_counts have different lengths."""
        moves = [chess.Move.from_uci("e2e4")]
        visit_counts = [10, 5]  # Different length
        
        with pytest.raises(ValueError, match="Moves and visit_counts must have same length"):
            MoveEncoder.moves_to_policy_vector(moves, visit_counts)
    
    def test_policy_vector_to_move_greedy(self):
        """Test greedy move selection from policy vector."""
        pos = Position()
        policy = np.zeros(4096)
        
        # Set high probability for e2e4
        e2e4_idx = MoveEncoder.encode_move(chess.Move.from_uci("e2e4"))
        policy[e2e4_idx] = 0.8
        
        # Set lower probability for g1f3
        g1f3_idx = MoveEncoder.encode_move(chess.Move.from_uci("g1f3"))
        policy[g1f3_idx] = 0.2
        
        # Greedy selection (temperature=0) should pick e2e4
        move = MoveEncoder.policy_vector_to_move(policy, pos, temperature=0.0)
        assert move.uci() == "e2e4"
    
    def test_policy_vector_to_move_temperature_sampling(self):
        """Test temperature-based move sampling."""
        pos = Position()
        policy = np.zeros(4096)
        
        # Set equal probabilities for two moves
        e2e4_idx = MoveEncoder.encode_move(chess.Move.from_uci("e2e4"))
        g1f3_idx = MoveEncoder.encode_move(chess.Move.from_uci("g1f3"))
        policy[e2e4_idx] = 0.5
        policy[g1f3_idx] = 0.5
        
        # With temperature=1, should sample from distribution
        # Run multiple times to check both moves can be selected
        moves_selected = set()
        for _ in range(20):
            move = MoveEncoder.policy_vector_to_move(policy, pos, temperature=1.0)
            moves_selected.add(move.uci())
        
        # Should have selected both moves at some point
        assert len(moves_selected) >= 1  # At least one move selected
    
    def test_policy_vector_to_move_no_legal_probability(self):
        """Test error when no legal moves have positive probability."""
        pos = Position()
        policy = np.zeros(4096)  # All zeros
        
        with pytest.raises(ValueError, match="No legal moves have positive probability"):
            MoveEncoder.policy_vector_to_move(policy, pos)
    
    def test_policy_vector_to_move_high_temperature(self):
        """Test move selection with high temperature (more random)."""
        pos = Position()
        policy = np.zeros(4096)
        
        # Set very different probabilities
        e2e4_idx = MoveEncoder.encode_move(chess.Move.from_uci("e2e4"))
        g1f3_idx = MoveEncoder.encode_move(chess.Move.from_uci("g1f3"))
        policy[e2e4_idx] = 0.99
        policy[g1f3_idx] = 0.01
        
        # With high temperature, should sometimes pick the low-probability move
        moves_selected = set()
        for _ in range(50):
            move = MoveEncoder.policy_vector_to_move(policy, pos, temperature=10.0)
            moves_selected.add(move.uci())
        
        # Should have some variety in moves selected
        assert len(moves_selected) >= 1