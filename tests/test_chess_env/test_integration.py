"""Integration tests for chess environment module."""

import numpy as np
import chess
from src.chess_env import Position, MoveEncoder, PositionGenerator


class TestChessEnvironmentIntegration:
    """Integration tests for the complete chess environment module."""
    
    def test_position_generator_with_move_encoder(self):
        """Test that generated positions work with move encoder."""
        gen = PositionGenerator(curriculum_level=1, seed=42)
        
        for _ in range(5):
            pos = gen.generate_position()
            
            # Should be able to get move mask
            mask = MoveEncoder.get_move_mask(pos)
            assert mask.shape == (4096,)
            assert np.sum(mask) > 0  # Should have legal moves
            
            # Should be able to get legal move indices
            indices = MoveEncoder.get_legal_move_indices(pos)
            assert len(indices) > 0
            
            # All indices should be valid
            for idx in indices:
                move = MoveEncoder.decode_move(idx)
                assert move in pos.get_legal_moves()
    
    def test_position_tensor_with_generated_positions(self):
        """Test tensor conversion works with generated positions."""
        gen = PositionGenerator(curriculum_level=0, seed=42)
        
        for _ in range(5):
            pos = gen.generate_position()
            
            # Should be able to convert to tensor
            tensor = pos.to_tensor()
            assert tensor.shape == (8, 8, 12)
            
            # Should have exactly 3 pieces (2 kings + 1 pawn)
            assert np.sum(tensor) == 3.0
    
    def test_canonical_form_with_move_encoding(self):
        """Test canonical form works with move encoding."""
        # Create position with black to move
        board = chess.Board("8/8/4k3/4P3/4K3/8/8/8 b - - 0 1")
        pos = Position(board)
        
        # Get canonical form
        canonical_pos = pos.get_canonical_form()
        
        # Should be able to encode moves from canonical position
        mask = MoveEncoder.get_move_mask(canonical_pos)
        assert np.sum(mask) > 0
        
        # Canonical position should be white to move
        assert canonical_pos.board.turn == chess.WHITE
    
    def test_complete_workflow(self):
        """Test complete workflow: generate -> encode -> make move -> validate."""
        gen = PositionGenerator(curriculum_level=1, seed=42)
        
        # Generate position
        pos = gen.generate_position()
        assert gen.is_valid_kp_endgame(pos.board)
        
        # Get legal moves
        legal_moves = pos.get_legal_moves()
        if len(legal_moves) == 0:
            return  # Skip if terminal position
        
        # Encode first legal move
        move = legal_moves[0]
        move_idx = MoveEncoder.encode_move(move)
        
        # Decode it back
        decoded_move = MoveEncoder.decode_move(move_idx)
        assert decoded_move.from_square == move.from_square
        assert decoded_move.to_square == move.to_square
        
        # Make the move
        new_pos = pos.make_move(move)
        
        # New position should still be valid (might not be KP endgame if pawn promoted)
        assert isinstance(new_pos, Position)
    
    def test_policy_vector_integration(self):
        """Test policy vector creation and move sampling."""
        gen = PositionGenerator(curriculum_level=1, seed=42)
        pos = gen.generate_position()
        
        legal_moves = pos.get_legal_moves()
        if len(legal_moves) == 0:
            return  # Skip terminal positions
        
        # Create mock visit counts
        visit_counts = [i + 1 for i in range(len(legal_moves))]
        
        # Convert to policy vector
        policy = MoveEncoder.moves_to_policy_vector(legal_moves, visit_counts)
        
        # Sample move from policy
        sampled_move = MoveEncoder.policy_vector_to_move(policy, pos, temperature=1.0)
        
        # Sampled move should be legal
        assert sampled_move in legal_moves
    
    def test_curriculum_progression(self):
        """Test that all curriculum levels generate valid positions."""
        for level in [0, 1, 2]:
            gen = PositionGenerator(curriculum_level=level, seed=42)
            
            # Generate multiple positions for each level
            for _ in range(3):
                pos = gen.generate_position()
                
                # Should be valid KP endgame
                assert gen.is_valid_kp_endgame(pos.board)
                
                # Should be able to convert to tensor
                tensor = pos.to_tensor()
                assert tensor.shape == (8, 8, 12)
                
                # Should be able to get move mask
                mask = MoveEncoder.get_move_mask(pos)
                assert mask.shape == (4096,)
                
                # Get curriculum stats
                stats = gen.get_curriculum_stats()
                assert stats["level"] == level