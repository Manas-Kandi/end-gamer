"""Unit tests for TestSuite and TestPosition classes."""

import pytest
import chess
from src.evaluation.test_suite import (
    TestSuite, TestPosition, PositionDifficulty, ExpectedResult
)
from src.chess_env.position import Position


class TestTestPosition:
    """Test cases for TestPosition class."""
    
    def test_valid_test_position(self):
        """Test creating a valid test position."""
        fen = "8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"
        position = Position(chess.Board(fen))
        
        test_pos = TestPosition(
            position=position,
            fen=fen,
            description="Test position",
            difficulty=PositionDifficulty.MEDIUM,
            expected_result=ExpectedResult.WIN,
            key_concepts=["opposition", "king_support"]
        )
        
        assert test_pos.position == position
        assert test_pos.fen == fen
        assert test_pos.description == "Test position"
        assert test_pos.difficulty == PositionDifficulty.MEDIUM
        assert test_pos.expected_result == ExpectedResult.WIN
        assert test_pos.key_concepts == ["opposition", "king_support"]
        assert test_pos.source is None
    
    def test_test_position_with_source(self):
        """Test creating test position with source."""
        fen = "8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"
        position = Position(chess.Board(fen))
        
        test_pos = TestPosition(
            position=position,
            fen=fen,
            description="Test position",
            difficulty=PositionDifficulty.MEDIUM,
            expected_result=ExpectedResult.WIN,
            key_concepts=["opposition"],
            source="Test Suite"
        )
        
        assert test_pos.source == "Test Suite"
    
    def test_fen_mismatch_error(self):
        """Test error when position FEN doesn't match provided FEN."""
        fen1 = "8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"
        fen2 = "8/8/4k3/4P3/4K3/8/8/8 b - - 0 1"  # Different turn
        position = Position(chess.Board(fen1))
        
        with pytest.raises(ValueError, match="Position FEN mismatch"):
            TestPosition(
                position=position,
                fen=fen2,
                description="Test position",
                difficulty=PositionDifficulty.MEDIUM,
                expected_result=ExpectedResult.WIN,
                key_concepts=["test"]
            )
    
    def test_invalid_kp_endgame_error(self):
        """Test error when position is not a valid king-pawn endgame."""
        # Position with extra pieces
        fen = "8/8/4k3/4P3/4KQ2/8/8/8 w - - 0 1"
        position = Position(chess.Board(fen))
        
        with pytest.raises(ValueError, match="not a valid king-pawn endgame"):
            TestPosition(
                position=position,
                fen=fen,
                description="Invalid position",
                difficulty=PositionDifficulty.MEDIUM,
                expected_result=ExpectedResult.WIN,
                key_concepts=["test"]
            )
    
    def test_is_valid_kp_endgame_valid_position(self):
        """Test validation of valid king-pawn endgame."""
        fen = "8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"
        position = Position(chess.Board(fen))
        
        test_pos = TestPosition(
            position=position,
            fen=fen,
            description="Valid KP endgame",
            difficulty=PositionDifficulty.MEDIUM,
            expected_result=ExpectedResult.WIN,
            key_concepts=["basic"]
        )
        
        assert test_pos._is_valid_kp_endgame()
    
    def test_is_valid_kp_endgame_multiple_pawns(self):
        """Test validation rejects multiple pawns."""
        fen = "8/8/4k3/4PP2/4K3/8/8/8 w - - 0 1"
        position = Position(chess.Board(fen))
        
        with pytest.raises(ValueError, match="not a valid king-pawn endgame"):
            TestPosition(
                position=position,
                fen=fen,
                description="Multiple pawns",
                difficulty=PositionDifficulty.MEDIUM,
                expected_result=ExpectedResult.WIN,
                key_concepts=["test"]
            )
    
    def test_is_valid_kp_endgame_black_pawn(self):
        """Test validation rejects black pawn."""
        fen = "8/8/4k3/4Pp2/4K3/8/8/8 w - - 0 1"
        position = Position(chess.Board(fen))
        
        with pytest.raises(ValueError, match="not a valid king-pawn endgame"):
            TestPosition(
                position=position,
                fen=fen,
                description="Black pawn present",
                difficulty=PositionDifficulty.MEDIUM,
                expected_result=ExpectedResult.WIN,
                key_concepts=["test"]
            )
    
    def test_is_valid_kp_endgame_extra_pieces(self):
        """Test validation rejects extra pieces."""
        fen = "8/8/4k3/4P3/4KR2/8/8/8 w - - 0 1"
        position = Position(chess.Board(fen))
        
        with pytest.raises(ValueError, match="not a valid king-pawn endgame"):
            TestPosition(
                position=position,
                fen=fen,
                description="Extra rook",
                difficulty=PositionDifficulty.MEDIUM,
                expected_result=ExpectedResult.WIN,
                key_concepts=["test"]
            )


class TestTestSuite:
    """Test cases for TestSuite class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.suite = TestSuite("Test Suite")
        
        # Create sample test positions
        self.easy_pos = TestPosition(
            position=Position(chess.Board("8/4P3/8/8/8/4k3/8/4K3 w - - 0 1")),
            fen="8/4P3/8/8/8/4k3/8/4K3 w - - 0 1",
            description="Easy win",
            difficulty=PositionDifficulty.EASY,
            expected_result=ExpectedResult.WIN,
            key_concepts=["pawn_promotion"]
        )
        
        self.medium_pos = TestPosition(
            position=Position(chess.Board("8/8/8/4k3/4P3/4K3/8/8 w - - 0 1")),
            fen="8/8/8/4k3/4P3/4K3/8/8 w - - 0 1",
            description="Opposition",
            difficulty=PositionDifficulty.MEDIUM,
            expected_result=ExpectedResult.WIN,
            key_concepts=["opposition"]
        )
        
        self.hard_pos = TestPosition(
            position=Position(chess.Board("8/8/8/8/8/3k4/3P4/3K4 w - - 0 1")),
            fen="8/8/8/8/8/3k4/3P4/3K4 w - - 0 1",
            description="Zugzwang",
            difficulty=PositionDifficulty.HARD,
            expected_result=ExpectedResult.DRAW,
            key_concepts=["zugzwang"]
        )
        
        self.draw_pos = TestPosition(
            position=Position(chess.Board("8/8/8/8/8/4k3/4P3/4K3 b - - 0 1")),
            fen="8/8/8/8/8/4k3/4P3/4K3 b - - 0 1",
            description="Theoretical draw",
            difficulty=PositionDifficulty.MEDIUM,
            expected_result=ExpectedResult.DRAW,
            key_concepts=["blockade"]
        )
    
    def test_empty_suite_initialization(self):
        """Test initializing empty test suite."""
        suite = TestSuite("Empty Suite")
        assert suite.name == "Empty Suite"
        assert len(suite.positions) == 0
        assert len(suite) == 0
    
    def test_suite_with_positions(self):
        """Test initializing suite with positions."""
        positions = [self.easy_pos, self.medium_pos]
        suite = TestSuite("Test Suite", positions)
        
        assert suite.name == "Test Suite"
        assert len(suite) == 2
        assert suite.positions == positions
    
    def test_add_position(self):
        """Test adding position to suite."""
        assert len(self.suite) == 0
        
        self.suite.add_position(self.easy_pos)
        assert len(self.suite) == 1
        assert self.suite.positions[0] == self.easy_pos
        
        self.suite.add_position(self.medium_pos)
        assert len(self.suite) == 2
    
    def test_get_positions_by_difficulty(self):
        """Test filtering positions by difficulty."""
        self.suite.add_position(self.easy_pos)
        self.suite.add_position(self.medium_pos)
        self.suite.add_position(self.hard_pos)
        
        easy_positions = self.suite.get_positions_by_difficulty(PositionDifficulty.EASY)
        assert len(easy_positions) == 1
        assert easy_positions[0] == self.easy_pos
        
        medium_positions = self.suite.get_positions_by_difficulty(PositionDifficulty.MEDIUM)
        assert len(medium_positions) == 1
        assert medium_positions[0] == self.medium_pos
        
        hard_positions = self.suite.get_positions_by_difficulty(PositionDifficulty.HARD)
        assert len(hard_positions) == 1
        assert hard_positions[0] == self.hard_pos
    
    def test_get_positions_by_result(self):
        """Test filtering positions by expected result."""
        self.suite.add_position(self.easy_pos)  # WIN
        self.suite.add_position(self.medium_pos)  # WIN
        self.suite.add_position(self.hard_pos)  # DRAW
        self.suite.add_position(self.draw_pos)  # DRAW
        
        winning_positions = self.suite.get_positions_by_result(ExpectedResult.WIN)
        assert len(winning_positions) == 2
        assert self.easy_pos in winning_positions
        assert self.medium_pos in winning_positions
        
        drawn_positions = self.suite.get_positions_by_result(ExpectedResult.DRAW)
        assert len(drawn_positions) == 2
        assert self.hard_pos in drawn_positions
        assert self.draw_pos in drawn_positions
        
        losing_positions = self.suite.get_positions_by_result(ExpectedResult.LOSS)
        assert len(losing_positions) == 0
    
    def test_get_positions_by_concept(self):
        """Test filtering positions by key concept."""
        self.suite.add_position(self.easy_pos)  # pawn_promotion
        self.suite.add_position(self.medium_pos)  # opposition
        self.suite.add_position(self.hard_pos)  # zugzwang
        self.suite.add_position(self.draw_pos)  # blockade
        
        opposition_positions = self.suite.get_positions_by_concept("opposition")
        assert len(opposition_positions) == 1
        assert opposition_positions[0] == self.medium_pos
        
        zugzwang_positions = self.suite.get_positions_by_concept("zugzwang")
        assert len(zugzwang_positions) == 1
        assert zugzwang_positions[0] == self.hard_pos
        
        nonexistent_positions = self.suite.get_positions_by_concept("nonexistent")
        assert len(nonexistent_positions) == 0
    
    def test_get_statistics(self):
        """Test getting suite statistics."""
        self.suite.add_position(self.easy_pos)  # EASY, WIN
        self.suite.add_position(self.medium_pos)  # MEDIUM, WIN
        self.suite.add_position(self.hard_pos)  # HARD, DRAW
        self.suite.add_position(self.draw_pos)  # MEDIUM, DRAW
        
        stats = self.suite.get_statistics()
        
        assert stats['total_positions'] == 4
        assert stats['easy_positions'] == 1
        assert stats['medium_positions'] == 2
        assert stats['hard_positions'] == 1
        assert stats['winning_positions'] == 2
        assert stats['drawn_positions'] == 2
        assert stats['losing_positions'] == 0
    
    def test_iteration(self):
        """Test iterating over suite positions."""
        positions = [self.easy_pos, self.medium_pos, self.hard_pos]
        for pos in positions:
            self.suite.add_position(pos)
        
        iterated_positions = list(self.suite)
        assert len(iterated_positions) == 3
        assert iterated_positions == positions
    
    def test_indexing(self):
        """Test accessing positions by index."""
        self.suite.add_position(self.easy_pos)
        self.suite.add_position(self.medium_pos)
        
        assert self.suite[0] == self.easy_pos
        assert self.suite[1] == self.medium_pos
        
        with pytest.raises(IndexError):
            _ = self.suite[2]
    
    def test_generate_standard_suite(self):
        """Test generating standard test suite."""
        suite = TestSuite.generate_standard_suite()
        
        assert suite.name == "Standard King-Pawn Endgame Suite"
        assert len(suite) > 0
        
        # Check that we have positions of all difficulties
        stats = suite.get_statistics()
        assert stats['easy_positions'] > 0
        assert stats['medium_positions'] > 0
        assert stats['hard_positions'] > 0
        
        # Check that we have both winning and drawn positions
        assert stats['winning_positions'] > 0
        assert stats['drawn_positions'] > 0
        
        # Verify all positions are valid king-pawn endgames
        for position in suite:
            assert position._is_valid_kp_endgame()
            assert len(position.key_concepts) > 0
            assert position.source == "Standard Suite"
    
    def test_standard_suite_specific_positions(self):
        """Test that standard suite contains expected specific positions."""
        suite = TestSuite.generate_standard_suite()
        
        # Check for some specific positions we expect
        fens = [pos.fen for pos in suite]
        
        # Should have pawn on 7th rank position
        assert "8/4P3/8/8/8/4k3/8/4K3 w - - 0 1" in fens
        
        # Should have classic opposition position
        assert "8/8/8/4k3/4P3/4K3/8/8 w - - 0 1" in fens
        
        # Should have zugzwang position
        assert "8/8/8/8/8/3k4/3P4/3K4 w - - 0 1" in fens
        
        # Should have theoretical draw position
        assert "8/8/8/8/8/4k3/4P3/4K3 b - - 0 1" in fens
    
    def test_standard_suite_concepts(self):
        """Test that standard suite covers key endgame concepts."""
        suite = TestSuite.generate_standard_suite()
        
        # Collect all concepts
        all_concepts = set()
        for position in suite:
            all_concepts.update(position.key_concepts)
        
        # Check for important concepts
        expected_concepts = [
            "opposition", "zugzwang", "triangulation", "key_squares",
            "pawn_promotion", "king_support", "blockade", "stalemate"
        ]
        
        for concept in expected_concepts:
            assert any(concept in pos.key_concepts for pos in suite), \
                f"Concept '{concept}' not found in any position"
    
    def test_standard_suite_difficulty_distribution(self):
        """Test that standard suite has reasonable difficulty distribution."""
        suite = TestSuite.generate_standard_suite()
        stats = suite.get_statistics()
        
        # Should have positions at all difficulty levels
        assert stats['easy_positions'] >= 2
        assert stats['medium_positions'] >= 4
        assert stats['hard_positions'] >= 2
        
        # Should have both wins and draws
        assert stats['winning_positions'] >= 5
        assert stats['drawn_positions'] >= 3
        
        # Total should be reasonable size
        assert 10 <= stats['total_positions'] <= 20