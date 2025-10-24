"""Position generator for king-pawn endgame positions."""

import random
from typing import Optional, Tuple
import chess
from .position import Position


class PositionGenerator:
    """Generate random king-pawn endgame positions with curriculum learning.
    
    Supports three curriculum levels:
    - Level 0 (Simple): Pawn far advanced, easy wins
    - Level 1 (Medium): Requires king support, opposition concepts
    - Level 2 (Complex): Zugzwang, triangulation, complex endgames
    """
    
    def __init__(self, curriculum_level: int = 0, seed: Optional[int] = None):
        """Initialize position generator.
        
        Args:
            curriculum_level: Difficulty level (0=simple, 1=medium, 2=complex)
            seed: Random seed for reproducibility
        """
        if not (0 <= curriculum_level <= 2):
            raise ValueError("Curriculum level must be 0, 1, or 2")
        
        self.curriculum_level = curriculum_level
        if seed is not None:
            random.seed(seed)
    
    def generate_position(self) -> Position:
        """Generate random legal king-pawn position based on curriculum level.
        
        Returns:
            Position object with valid king-pawn endgame
        """
        max_attempts = 1000
        
        for _ in range(max_attempts):
            try:
                if self.curriculum_level == 0:
                    position = self._generate_simple_position()
                elif self.curriculum_level == 1:
                    position = self._generate_medium_position()
                else:  # curriculum_level == 2
                    position = self._generate_complex_position()
                
                if self.is_valid_kp_endgame(position.board):
                    return position
            except Exception:
                # If position generation fails, try again
                continue
        
        # Fallback to a known valid position
        return self._generate_fallback_position()
    
    def _generate_simple_position(self) -> Position:
        """Generate simple king-pawn position (pawn far advanced)."""
        board = chess.Board(fen=None)  # Empty board
        
        # Place white king randomly but not on back ranks
        white_king_square = random.choice([
            sq for sq in chess.SQUARES 
            if chess.square_rank(sq) in [2, 3, 4, 5]  # Ranks 3-6
        ])
        board.set_piece_at(white_king_square, chess.Piece(chess.KING, chess.WHITE))
        
        # Place pawn on 6th or 7th rank (far advanced)
        pawn_rank = random.choice([5, 6])  # 6th or 7th rank (0-indexed)
        pawn_file = random.randint(0, 7)
        pawn_square = chess.square(pawn_file, pawn_rank)
        
        # Make sure pawn square is not occupied
        if board.piece_at(pawn_square) is None:
            board.set_piece_at(pawn_square, chess.Piece(chess.PAWN, chess.WHITE))
        else:
            # Try adjacent file
            pawn_file = (pawn_file + 1) % 8
            pawn_square = chess.square(pawn_file, pawn_rank)
            board.set_piece_at(pawn_square, chess.Piece(chess.PAWN, chess.WHITE))
        
        # Place black king at least 2 squares away from white king and pawn
        valid_squares = []
        for sq in chess.SQUARES:
            if (board.piece_at(sq) is None and 
                chess.square_distance(sq, white_king_square) >= 2 and
                chess.square_distance(sq, pawn_square) >= 2):
                valid_squares.append(sq)
        
        if valid_squares:
            black_king_square = random.choice(valid_squares)
            board.set_piece_at(black_king_square, chess.Piece(chess.KING, chess.BLACK))
        
        # Set random turn
        board.turn = random.choice([chess.WHITE, chess.BLACK])
        
        return Position(board)
    
    def _generate_medium_position(self) -> Position:
        """Generate medium difficulty position (requires king support)."""
        board = chess.Board(fen=None)  # Empty board
        
        # Place pawn on 4th, 5th, or 6th rank
        pawn_rank = random.choice([3, 4, 5])  # 4th, 5th, or 6th rank
        pawn_file = random.randint(1, 6)  # Avoid edge files for more complexity
        pawn_square = chess.square(pawn_file, pawn_rank)
        board.set_piece_at(pawn_square, chess.Piece(chess.PAWN, chess.WHITE))
        
        # Place white king to support pawn (within 2-3 squares)
        king_candidates = []
        for sq in chess.SQUARES:
            distance = chess.square_distance(sq, pawn_square)
            if 1 <= distance <= 3 and chess.square_rank(sq) >= 1:  # Not on back rank
                king_candidates.append(sq)
        
        if king_candidates:
            white_king_square = random.choice(king_candidates)
            board.set_piece_at(white_king_square, chess.Piece(chess.KING, chess.WHITE))
        
        # Place black king to create opposition scenarios
        # Try to place it in front of or near the pawn
        black_king_candidates = []
        for sq in chess.SQUARES:
            if (board.piece_at(sq) is None and
                chess.square_distance(sq, white_king_square) >= 2):
                
                # Prefer squares that create interesting opposition
                sq_file = chess.square_file(sq)
                sq_rank = chess.square_rank(sq)
                
                # In front of pawn or on same file
                if (sq_file == pawn_file and sq_rank > pawn_rank) or \
                   abs(sq_file - pawn_file) <= 1:
                    black_king_candidates.append(sq)
        
        if not black_king_candidates:
            # Fallback: any valid square
            black_king_candidates = [
                sq for sq in chess.SQUARES 
                if (board.piece_at(sq) is None and
                    chess.square_distance(sq, white_king_square) >= 2)
            ]
        
        if black_king_candidates:
            black_king_square = random.choice(black_king_candidates)
            board.set_piece_at(black_king_square, chess.Piece(chess.KING, chess.BLACK))
        
        board.turn = random.choice([chess.WHITE, chess.BLACK])
        
        return Position(board)
    
    def _generate_complex_position(self) -> Position:
        """Generate complex position (zugzwang, triangulation)."""
        board = chess.Board(fen=None)  # Empty board
        
        # Place pawn on middle ranks for complex play
        pawn_rank = random.choice([2, 3, 4])  # 3rd, 4th, or 5th rank
        pawn_file = random.randint(2, 5)  # Central files
        pawn_square = chess.square(pawn_file, pawn_rank)
        board.set_piece_at(pawn_square, chess.Piece(chess.PAWN, chess.WHITE))
        
        # Place kings to create complex scenarios
        # White king behind or beside pawn
        white_king_candidates = []
        for sq in chess.SQUARES:
            sq_file = chess.square_file(sq)
            sq_rank = chess.square_rank(sq)
            
            # Behind pawn or on adjacent files
            if ((sq_file == pawn_file and sq_rank < pawn_rank) or
                (abs(sq_file - pawn_file) == 1 and abs(sq_rank - pawn_rank) <= 2)):
                white_king_candidates.append(sq)
        
        if white_king_candidates:
            white_king_square = random.choice(white_king_candidates)
            board.set_piece_at(white_king_square, chess.Piece(chess.KING, chess.WHITE))
        
        # Black king positioned for complex defense
        black_king_candidates = []
        for sq in chess.SQUARES:
            if (board.piece_at(sq) is None and
                chess.square_distance(sq, white_king_square) >= 2):
                
                sq_file = chess.square_file(sq)
                sq_rank = chess.square_rank(sq)
                
                # In front of pawn or creating opposition
                if (sq_file == pawn_file and sq_rank >= pawn_rank) or \
                   (abs(sq_file - pawn_file) <= 2 and sq_rank >= pawn_rank - 1):
                    black_king_candidates.append(sq)
        
        if not black_king_candidates:
            black_king_candidates = [
                sq for sq in chess.SQUARES 
                if (board.piece_at(sq) is None and
                    chess.square_distance(sq, white_king_square) >= 2)
            ]
        
        if black_king_candidates:
            black_king_square = random.choice(black_king_candidates)
            board.set_piece_at(black_king_square, chess.Piece(chess.KING, chess.BLACK))
        
        board.turn = random.choice([chess.WHITE, chess.BLACK])
        
        return Position(board)
    
    def _generate_fallback_position(self) -> Position:
        """Generate a known valid king-pawn position as fallback."""
        # Simple king-pawn endgame: White king e4, black king e6, white pawn e5
        board = chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1")
        return Position(board)
    
    def is_valid_kp_endgame(self, board: chess.Board) -> bool:
        """Validate that position is a legal king-pawn endgame.
        
        Args:
            board: chess.Board to validate
            
        Returns:
            True if position is valid king-pawn endgame
        """
        # Check that board is legal
        try:
            if not board.is_valid():
                return False
        except Exception:
            return False
        
        # Count pieces
        piece_counts = {
            chess.WHITE: {piece_type: 0 for piece_type in chess.PIECE_TYPES},
            chess.BLACK: {piece_type: 0 for piece_type in chess.PIECE_TYPES}
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_counts[piece.color][piece.piece_type] += 1
        
        # Must have exactly one king per side
        if (piece_counts[chess.WHITE][chess.KING] != 1 or
            piece_counts[chess.BLACK][chess.KING] != 1):
            return False
        
        # Must have exactly one white pawn and no black pawns
        if (piece_counts[chess.WHITE][chess.PAWN] != 1 or
            piece_counts[chess.BLACK][chess.PAWN] != 0):
            return False
        
        # No other pieces allowed
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                if piece_counts[color][piece_type] != 0:
                    return False
        
        # Kings must not be adjacent
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        if white_king_square is None or black_king_square is None:
            return False
        
        if chess.square_distance(white_king_square, black_king_square) < 2:
            return False
        
        # Pawn must not be on first or eighth rank
        pawn_square = None
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                pawn_square = square
                break
        
        if pawn_square is None:
            return False
        
        pawn_rank = chess.square_rank(pawn_square)
        if pawn_rank == 0 or pawn_rank == 7:
            return False
        
        return True
    
    def get_curriculum_stats(self) -> dict:
        """Get statistics about the current curriculum level.
        
        Returns:
            Dictionary with curriculum information
        """
        level_descriptions = {
            0: "Simple positions with pawn far advanced (6th-7th rank)",
            1: "Medium positions requiring king support and opposition",
            2: "Complex positions with zugzwang and triangulation"
        }
        
        return {
            "level": self.curriculum_level,
            "description": level_descriptions[self.curriculum_level],
            "expected_difficulty": ["Easy", "Medium", "Hard"][self.curriculum_level]
        }