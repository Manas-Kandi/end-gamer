"""Test suite for evaluating chess engine performance on king-pawn endgames."""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import chess

from ..chess_env.position import Position


class PositionDifficulty(Enum):
    """Difficulty levels for test positions."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ExpectedResult(Enum):
    """Expected theoretical result of a position."""
    WIN = "win"
    DRAW = "draw"
    LOSS = "loss"


@dataclass
class TestPosition:
    """Single test position with metadata."""
    
    position: Position
    fen: str
    description: str
    difficulty: PositionDifficulty
    expected_result: ExpectedResult
    key_concepts: List[str]
    source: Optional[str] = None
    
    def __post_init__(self):
        """Validate test position after initialization."""
        # Ensure position matches FEN
        if self.position.board.fen() != self.fen:
            raise ValueError(f"Position FEN mismatch: {self.position.board.fen()} != {self.fen}")
        
        # Validate it's a king-pawn endgame
        if not self._is_valid_kp_endgame():
            raise ValueError(f"Position is not a valid king-pawn endgame: {self.fen}")
    
    def _is_valid_kp_endgame(self) -> bool:
        """Check if position is a valid king-pawn endgame."""
        board = self.position.board
        
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
        
        return True


class TestSuite:
    """Collection of test positions for evaluating chess engine performance."""
    
    def __init__(self, name: str, positions: Optional[List[TestPosition]] = None):
        """Initialize test suite.
        
        Args:
            name: Name of the test suite
            positions: List of test positions (empty if None)
        """
        self.name = name
        self.positions = positions or []
    
    def add_position(self, position: TestPosition) -> None:
        """Add a test position to the suite.
        
        Args:
            position: TestPosition to add
        """
        self.positions.append(position)
    
    def get_positions_by_difficulty(self, difficulty: PositionDifficulty) -> List[TestPosition]:
        """Get positions filtered by difficulty level.
        
        Args:
            difficulty: Difficulty level to filter by
            
        Returns:
            List of positions with specified difficulty
        """
        return [pos for pos in self.positions if pos.difficulty == difficulty]
    
    def get_positions_by_result(self, expected_result: ExpectedResult) -> List[TestPosition]:
        """Get positions filtered by expected result.
        
        Args:
            expected_result: Expected result to filter by
            
        Returns:
            List of positions with specified expected result
        """
        return [pos for pos in self.positions if pos.expected_result == expected_result]
    
    def get_positions_by_concept(self, concept: str) -> List[TestPosition]:
        """Get positions that test a specific concept.
        
        Args:
            concept: Chess concept to filter by (e.g., "opposition", "zugzwang")
            
        Returns:
            List of positions that involve the specified concept
        """
        return [pos for pos in self.positions if concept in pos.key_concepts]
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the test suite.
        
        Returns:
            Dictionary with counts by difficulty and expected result
        """
        stats = {
            'total_positions': len(self.positions),
            'easy_positions': len(self.get_positions_by_difficulty(PositionDifficulty.EASY)),
            'medium_positions': len(self.get_positions_by_difficulty(PositionDifficulty.MEDIUM)),
            'hard_positions': len(self.get_positions_by_difficulty(PositionDifficulty.HARD)),
            'winning_positions': len(self.get_positions_by_result(ExpectedResult.WIN)),
            'drawn_positions': len(self.get_positions_by_result(ExpectedResult.DRAW)),
            'losing_positions': len(self.get_positions_by_result(ExpectedResult.LOSS))
        }
        return stats
    
    def __len__(self) -> int:
        """Get number of positions in suite."""
        return len(self.positions)
    
    def __iter__(self):
        """Iterate over positions in suite."""
        return iter(self.positions)
    
    def __getitem__(self, index: int) -> TestPosition:
        """Get position by index."""
        return self.positions[index]
    
    @classmethod
    def generate_standard_suite(cls) -> 'TestSuite':
        """Generate standard test suite with classic king-pawn positions.
        
        Returns:
            TestSuite with comprehensive set of king-pawn endgame positions
        """
        suite = cls("Standard King-Pawn Endgame Suite")
        
        # Easy positions - pawn far advanced, clear wins
        easy_positions = [
            # Pawn on 7th rank, easy promotion
            {
                'fen': '8/4P3/8/8/8/4k3/8/4K3 w - - 0 1',
                'description': 'Pawn on 7th rank, easy promotion',
                'expected_result': ExpectedResult.WIN,
                'key_concepts': ['pawn_promotion', 'basic_endgame']
            },
            # Pawn on 6th rank with king support
            {
                'fen': '8/8/4P3/4K3/8/8/8/4k3 w - - 0 1',
                'description': 'Pawn on 6th rank with active king support',
                'expected_result': ExpectedResult.WIN,
                'key_concepts': ['king_support', 'pawn_advancement']
            },
            # Pawn on 6th, black king far away
            {
                'fen': '8/8/4P3/8/8/8/8/k3K3 w - - 0 1',
                'description': 'Pawn on 6th rank, black king too far to stop',
                'expected_result': ExpectedResult.WIN,
                'key_concepts': ['king_distance', 'pawn_race']
            }
        ]
        
        # Medium positions - require understanding of opposition and king support
        medium_positions = [
            # Classic opposition position
            {
                'fen': '8/8/8/4k3/4P3/4K3/8/8 w - - 0 1',
                'description': 'Classic opposition with pawn on 4th rank',
                'expected_result': ExpectedResult.WIN,
                'key_concepts': ['opposition', 'king_support', 'pawn_breakthrough']
            },
            # Opposition - black to move (draw)
            {
                'fen': '8/8/8/4k3/4P3/4K3/8/8 b - - 0 1',
                'description': 'Opposition position, black to move - theoretical draw',
                'expected_result': ExpectedResult.DRAW,
                'key_concepts': ['opposition', 'defensive_technique']
            },
            # Pawn on 5th rank, kings facing each other
            {
                'fen': '8/8/4k3/4P3/4K3/8/8/8 w - - 0 1',
                'description': 'Pawn on 5th rank, direct king opposition',
                'expected_result': ExpectedResult.WIN,
                'key_concepts': ['opposition', 'key_squares', 'pawn_support']
            },
            # Key square control
            {
                'fen': '8/8/8/8/3kP3/8/3K4/8 w - - 0 1',
                'description': 'Key square battle - white king must advance',
                'expected_result': ExpectedResult.WIN,
                'key_concepts': ['key_squares', 'king_activity', 'centralization']
            }
        ]
        
        # Hard positions - zugzwang, triangulation, complex endgames
        hard_positions = [
            # Zugzwang position
            {
                'fen': '8/8/8/8/8/3k4/3P4/3K4 w - - 0 1',
                'description': 'Zugzwang - whoever moves first is worse',
                'expected_result': ExpectedResult.DRAW,
                'key_concepts': ['zugzwang', 'tempo', 'defensive_technique']
            },
            # Triangulation setup
            {
                'fen': '8/8/8/8/8/2k5/2P5/2K5 w - - 0 1',
                'description': 'Triangulation required to gain tempo',
                'expected_result': ExpectedResult.WIN,
                'key_concepts': ['triangulation', 'tempo', 'king_maneuver']
            },
            # Complex pawn on 3rd rank
            {
                'fen': '8/8/8/8/8/8/3P4/k2K4 w - - 0 1',
                'description': 'Pawn on 2nd rank, complex king play required',
                'expected_result': ExpectedResult.WIN,
                'key_concepts': ['king_activity', 'pawn_support', 'precise_play']
            },
            # Stalemate trap
            {
                'fen': '8/8/8/8/8/8/1P6/k1K5 w - - 0 1',
                'description': 'Beware of stalemate traps with rook pawn',
                'expected_result': ExpectedResult.WIN,
                'key_concepts': ['stalemate_avoidance', 'rook_pawn', 'king_placement']
            }
        ]
        
        # Theoretical draw positions
        draw_positions = [
            # King in front of pawn
            {
                'fen': '8/8/8/8/8/4k3/4P3/4K3 b - - 0 1',
                'description': 'Black king blocks pawn, theoretical draw',
                'expected_result': ExpectedResult.DRAW,
                'key_concepts': ['blockade', 'defensive_king', 'draw_technique']
            },
            # Distant opposition
            {
                'fen': '8/8/8/8/4k3/8/4P3/4K3 b - - 0 1',
                'description': 'Distant opposition, black holds the draw',
                'expected_result': ExpectedResult.DRAW,
                'key_concepts': ['distant_opposition', 'defensive_technique']
            },
            # Rook pawn draw
            {
                'fen': '8/8/8/8/8/7k/7P/7K w - - 0 1',
                'description': 'Rook pawn - theoretical draw due to stalemate',
                'expected_result': ExpectedResult.DRAW,
                'key_concepts': ['rook_pawn', 'stalemate', 'fortress']
            }
        ]
        
        # Add all positions to suite
        for pos_data in easy_positions:
            position = Position(chess.Board(pos_data['fen']))
            test_pos = TestPosition(
                position=position,
                fen=pos_data['fen'],
                description=pos_data['description'],
                difficulty=PositionDifficulty.EASY,
                expected_result=pos_data['expected_result'],
                key_concepts=pos_data['key_concepts'],
                source="Standard Suite"
            )
            suite.add_position(test_pos)
        
        for pos_data in medium_positions:
            position = Position(chess.Board(pos_data['fen']))
            test_pos = TestPosition(
                position=position,
                fen=pos_data['fen'],
                description=pos_data['description'],
                difficulty=PositionDifficulty.MEDIUM,
                expected_result=pos_data['expected_result'],
                key_concepts=pos_data['key_concepts'],
                source="Standard Suite"
            )
            suite.add_position(test_pos)
        
        for pos_data in hard_positions:
            position = Position(chess.Board(pos_data['fen']))
            test_pos = TestPosition(
                position=position,
                fen=pos_data['fen'],
                description=pos_data['description'],
                difficulty=PositionDifficulty.HARD,
                expected_result=pos_data['expected_result'],
                key_concepts=pos_data['key_concepts'],
                source="Standard Suite"
            )
            suite.add_position(test_pos)
        
        for pos_data in draw_positions:
            position = Position(chess.Board(pos_data['fen']))
            test_pos = TestPosition(
                position=position,
                fen=pos_data['fen'],
                description=pos_data['description'],
                difficulty=PositionDifficulty.MEDIUM,  # Most draws are medium difficulty
                expected_result=pos_data['expected_result'],
                key_concepts=pos_data['key_concepts'],
                source="Standard Suite"
            )
            suite.add_position(test_pos)
        
        return suite