"""Unit tests for Position class."""

import pytest
import numpy as np
import chess
from src.chess_env.position import Position


class TestPosition:
    """Test cases for Position class."""
    
    def test_init_default(self):
        """Test Position initialization with default board."""
        pos = Position()
        assert isinstance(pos.board, chess.Board)
        assert pos.board.fen() == chess.STARTING_FEN
    
    def test_init_with_board(self):
        """Test Position initialization with custom board."""
        board = chess.Board("8/8/8/8/8/8/4P3/4K3 w - - 0 1")  # King-pawn endgame
        pos = Position(board)
        assert pos.board.fen() == "8/8/8/8/8/8/4P3/4K3 w - - 0 1"
    
    def test_to_tensor_shape(self):
        """Test tensor conversion produces correct shape."""
        pos = Position()
        tensor = pos.to_tensor()
        assert tensor.shape == (8, 8, 12)
        assert tensor.dtype == np.float32
    
    def test_to_tensor_starting_position(self):
        """Test tensor conversion for starting position."""
        pos = Position()
        tensor = pos.to_tensor()
        
        # Check white king on e1 (row 7, col 4, channel 0)
        assert tensor[7, 4, 0] == 1.0
        
        # Check black king on e8 (row 0, col 4, channel 6)
        assert tensor[0, 4, 6] == 1.0
        
        # Check white pawn on e2 (row 6, col 4, channel 5)
        assert tensor[6, 4, 5] == 1.0
        
        # Check black pawn on e7 (row 1, col 4, channel 11)
        assert tensor[1, 4, 11] == 1.0
        
        # Check empty square has all zeros
        assert np.all(tensor[4, 4, :] == 0.0)
    
    def test_to_tensor_king_pawn_endgame(self):
        """Test tensor conversion for king-pawn endgame."""
        # White king on e4, black king on e6, white pawn on e5
        board = chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1")
        pos = Position(board)
        tensor = pos.to_tensor()
        
        # Check white king on e4 (row 4, col 4, channel 0)
        assert tensor[4, 4, 0] == 1.0
        
        # Check black king on e6 (row 2, col 4, channel 6)
        assert tensor[2, 4, 6] == 1.0
        
        # Check white pawn on e5 (row 3, col 4, channel 5)
        assert tensor[3, 4, 5] == 1.0
        
        # Check that only these pieces are present
        assert np.sum(tensor) == 3.0
    
    def test_get_legal_moves(self):
        """Test legal move generation."""
        pos = Position()
        moves = pos.get_legal_moves()
        assert len(moves) == 20  # Starting position has 20 legal moves
        assert all(isinstance(move, chess.Move) for move in moves)
    
    def test_get_legal_moves_king_pawn_endgame(self):
        """Test legal moves in king-pawn endgame."""
        # White king on e4, black king on e6, white pawn on e5
        board = chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1")
        pos = Position(board)
        moves = pos.get_legal_moves()
        
        # Should have king moves (pawn is blocked by black king)
        assert len(moves) > 0
        move_strings = [str(move) for move in moves]
        # King can move to various squares
        assert any("e4" in move for move in move_strings)
    
    def test_make_move_legal(self):
        """Test making a legal move."""
        pos = Position()
        move = chess.Move.from_uci("e2e4")
        new_pos = pos.make_move(move)
        
        # Original position unchanged
        assert pos.board.fen() == chess.STARTING_FEN
        
        # New position has move applied (en passant square may vary)
        assert "4P3" in new_pos.board.fen()  # Pawn moved to e4
        assert new_pos.board.turn == chess.BLACK  # Black to move
    
    def test_make_move_illegal(self):
        """Test making an illegal move raises error."""
        pos = Position()
        illegal_move = chess.Move.from_uci("e2e5")  # Pawn can't move two squares from e2 to e5
        
        with pytest.raises(ValueError, match="Move .* is not legal"):
            pos.make_move(illegal_move)
    
    def test_is_terminal_starting_position(self):
        """Test terminal check on starting position."""
        pos = Position()
        assert not pos.is_terminal()
    
    def test_is_terminal_checkmate(self):
        """Test terminal check on checkmate position."""
        # Fool's mate
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        pos = Position(board)
        assert pos.is_terminal()
    
    def test_is_terminal_stalemate(self):
        """Test terminal check on stalemate position."""
        # Stalemate position
        board = chess.Board("8/8/8/8/8/8/8/k6K b - - 0 1")
        pos = Position(board)
        assert pos.is_terminal()
    
    def test_is_terminal_fifty_move_rule(self):
        """Test terminal check for 50-move rule."""
        board = chess.Board("8/8/8/8/8/8/8/k6K w - - 100 1")  # 50 moves without pawn move or capture
        pos = Position(board)
        assert pos.is_terminal()
    
    def test_get_result_checkmate(self):
        """Test result for checkmate position."""
        # Fool's mate - white is checkmated
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        pos = Position(board)
        assert pos.get_result() == -1.0  # Current player (white) loses
    
    def test_get_result_stalemate(self):
        """Test result for stalemate position."""
        board = chess.Board("8/8/8/8/8/8/8/k6K b - - 0 1")
        pos = Position(board)
        assert pos.get_result() == 0.0  # Draw
    
    def test_get_result_non_terminal(self):
        """Test result for non-terminal position raises error."""
        pos = Position()
        with pytest.raises(ValueError, match="Position is not terminal"):
            pos.get_result()
    
    def test_get_canonical_form_white_to_move(self):
        """Test canonical form when white to move."""
        board = chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1")
        pos = Position(board)
        canonical = pos.get_canonical_form()
        
        # Should be same position since white to move
        assert canonical.board.fen() == pos.board.fen()
    
    def test_get_canonical_form_black_to_move(self):
        """Test canonical form when black to move."""
        board = chess.Board("8/8/4k3/4P3/4K3/8/8/8 b - - 0 1")
        pos = Position(board)
        canonical = pos.get_canonical_form()
        
        # Should be flipped position
        assert canonical.board.fen() != pos.board.fen()
        # After flipping, it should be white to move
        assert canonical.board.turn == chess.WHITE
    
    def test_str_representation(self):
        """Test string representation."""
        pos = Position()
        str_repr = str(pos)
        # The string representation includes spaces, so check for individual pieces
        assert "r" in str_repr.lower() and "n" in str_repr.lower() and "k" in str_repr.lower()
    
    def test_repr_representation(self):
        """Test detailed string representation."""
        pos = Position()
        repr_str = repr(pos)
        assert "Position" in repr_str
        assert chess.STARTING_FEN in repr_str
    
    def test_equality(self):
        """Test position equality."""
        pos1 = Position()
        pos2 = Position()
        pos3 = Position(chess.Board("8/8/8/8/8/8/8/8 w - - 0 1"))
        
        assert pos1 == pos2
        assert pos1 != pos3
        assert pos1 != "not a position"
    
    def test_hash(self):
        """Test position hashing."""
        pos1 = Position()
        pos2 = Position()
        pos3 = Position(chess.Board("8/8/8/8/8/8/8/8 w - - 0 1"))
        
        assert hash(pos1) == hash(pos2)
        assert hash(pos1) != hash(pos3)
        
        # Can be used in sets
        position_set = {pos1, pos2, pos3}
        assert len(position_set) == 2  # pos1 and pos2 are equal