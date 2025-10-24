"""Position class for chess board representation and manipulation."""

from typing import List, Optional
import numpy as np
import chess
from copy import deepcopy


class Position:
    """Represents a chess position in king-pawn endgame.
    
    This class wraps python-chess Board and provides methods for neural network
    integration, including tensor conversion and canonical form representation.
    """
    
    def __init__(self, board: Optional[chess.Board] = None):
        """Initialize position with a chess board.
        
        Args:
            board: python-chess Board object. If None, creates empty board.
        """
        if board is None:
            self.board = chess.Board()
        else:
            self.board = deepcopy(board)
    
    def to_tensor(self) -> np.ndarray:
        """Convert position to 8x8x12 neural network input tensor.
        
        The tensor uses 12 channels:
        - Channels 0-5: White pieces (King, Queen, Rook, Bishop, Knight, Pawn)
        - Channels 6-11: Black pieces (King, Queen, Rook, Bishop, Knight, Pawn)
        
        Returns:
            numpy array of shape (8, 8, 12) with binary piece encoding
        """
        tensor = np.zeros((8, 8, 12), dtype=np.float32)
        
        # Piece type mapping to channel indices
        piece_to_channel = {
            chess.KING: 0,
            chess.QUEEN: 1, 
            chess.ROOK: 2,
            chess.BISHOP: 3,
            chess.KNIGHT: 4,
            chess.PAWN: 5
        }
        
        # Iterate through all squares
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                # Get row and column (chess uses different coordinate system)
                row = 7 - (square // 8)  # Flip row for neural network convention
                col = square % 8
                
                # Get base channel for piece type
                base_channel = piece_to_channel[piece.piece_type]
                
                # Add 6 for black pieces
                if piece.color == chess.BLACK:
                    base_channel += 6
                
                tensor[row, col, base_channel] = 1.0
        
        return tensor
    
    def get_legal_moves(self) -> List[chess.Move]:
        """Get all legal moves in current position.
        
        Returns:
            List of legal chess.Move objects
        """
        return list(self.board.legal_moves)
    
    def make_move(self, move: chess.Move) -> 'Position':
        """Return new position after making move.
        
        Args:
            move: chess.Move to make
            
        Returns:
            New Position object with move applied
            
        Raises:
            ValueError: If move is not legal in current position
        """
        if move not in self.board.legal_moves:
            raise ValueError(f"Move {move} is not legal in current position")
        
        new_board = deepcopy(self.board)
        new_board.push(move)
        return Position(new_board)
    
    def is_terminal(self) -> bool:
        """Check if position is game-ending.
        
        Returns:
            True if position is checkmate, stalemate, or draw by rule
        """
        # Check for checkmate or stalemate
        if self.board.is_checkmate() or self.board.is_stalemate():
            return True
        
        # Check for insufficient material
        if self.board.is_insufficient_material():
            return True
        
        # Check for 50-move rule
        if self.board.halfmove_clock >= 100:
            return True
        
        # Check for threefold repetition
        if self.board.is_repetition(count=3):
            return True
        
        return False
    
    def get_result(self) -> float:
        """Get game result from current player perspective.
        
        Returns:
            1.0 for win, 0.0 for draw, -1.0 for loss
        """
        if not self.is_terminal():
            raise ValueError("Position is not terminal")
        
        # Checkmate - current player loses
        if self.board.is_checkmate():
            return -1.0
        
        # All other terminal conditions are draws
        return 0.0
    
    def get_canonical_form(self) -> 'Position':
        """Return position from current player's perspective.
        
        For white to move, returns position as-is.
        For black to move, flips the board so black pieces appear as white.
        
        Returns:
            Position normalized to current player's perspective
        """
        if self.board.turn == chess.WHITE:
            # White to move - return as-is
            return Position(self.board)
        else:
            # Black to move - flip the board
            flipped_board = self.board.mirror()
            return Position(flipped_board)
    
    def __str__(self) -> str:
        """String representation of the position."""
        return str(self.board)
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Position('{self.board.fen()}')"
    
    def __eq__(self, other) -> bool:
        """Check equality with another Position."""
        if not isinstance(other, Position):
            return False
        return self.board.fen() == other.board.fen()
    
    def __hash__(self) -> int:
        """Hash based on board position."""
        return hash(self.board.fen())