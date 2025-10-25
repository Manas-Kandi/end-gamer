"""Unit tests for benchmark opponents."""

import pytest
import chess
from unittest.mock import Mock, patch

from src.evaluation.benchmark_opponents import (
    BenchmarkOpponent, RandomPlayer, SimpleHeuristicPlayer, MinimaxPlayer,
    get_standard_opponents, MatchResult, play_match, _play_game
)
from src.chess_env.position import Position


class TestBenchmarkOpponent:
    """Test cases for BenchmarkOpponent abstract base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract base class cannot be instantiated."""
        with pytest.raises(TypeError):
            BenchmarkOpponent("Test", 1200.0)
    
    def test_str_representation(self):
        """Test string representation of opponent."""
        # Create a concrete subclass for testing
        class TestOpponent(BenchmarkOpponent):
            def get_move(self, position):
                return None
        
        opponent = TestOpponent("Test Player", 1500.0)
        assert str(opponent) == "Test Player (Elo: 1500.0)"


class TestRandomPlayer:
    """Test cases for RandomPlayer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.player = RandomPlayer()
    
    def test_initialization(self):
        """Test random player initialization."""
        assert self.player.name == "Random Player"
        assert self.player.estimated_elo == 800.0
    
    def test_get_move_with_legal_moves(self):
        """Test getting move when legal moves are available."""
        position = Position(chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"))
        
        move = self.player.get_move(position)
        
        assert move is not None
        assert move in position.get_legal_moves()
    
    def test_get_move_no_legal_moves(self):
        """Test getting move when no legal moves available."""
        # Mock position with no legal moves
        position = Mock()
        position.get_legal_moves.return_value = []
        
        move = self.player.get_move(position)
        
        assert move is None
    
    def test_reproducible_randomness(self):
        """Test that random player produces reproducible results."""
        position = Position(chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"))
        
        # Get moves from two different instances
        player1 = RandomPlayer()
        player2 = RandomPlayer()
        
        move1 = player1.get_move(position)
        move2 = player2.get_move(position)
        
        # Should be the same due to fixed seed
        assert move1 == move2


class TestSimpleHeuristicPlayer:
    """Test cases for SimpleHeuristicPlayer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.player = SimpleHeuristicPlayer()
    
    def test_initialization(self):
        """Test simple heuristic player initialization."""
        assert self.player.name == "Simple Heuristic Player"
        assert self.player.estimated_elo == 1000.0
    
    def test_get_move_with_legal_moves(self):
        """Test getting move when legal moves are available."""
        position = Position(chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"))
        
        move = self.player.get_move(position)
        
        assert move is not None
        assert move in position.get_legal_moves()
    
    def test_get_move_no_legal_moves(self):
        """Test getting move when no legal moves available."""
        # Mock position with no legal moves
        position = Mock()
        position.get_legal_moves.return_value = []
        
        move = self.player.get_move(position)
        
        assert move is None
    
    def test_evaluate_position_terminal(self):
        """Test position evaluation for terminal positions."""
        # Mock terminal position
        position = Mock()
        position.is_terminal.return_value = True
        position.get_result.return_value = 1.0
        
        score = self.player._evaluate_position(position)
        
        assert score == 1000.0  # Large bonus for winning
    
    def test_evaluate_position_with_pawn(self):
        """Test position evaluation with pawn present."""
        position = Position(chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"))
        
        score = self.player._evaluate_position(position)
        
        assert isinstance(score, float)
        # Should be positive for white (pawn advanced, king support)
        assert score > 0
    
    def test_evaluate_position_no_pawn(self):
        """Test position evaluation with no pawn."""
        position = Position(chess.Board("8/8/4k3/8/4K3/8/8/8 w - - 0 1"))
        
        score = self.player._evaluate_position(position)
        
        assert isinstance(score, float)
        # Should be close to 0 without pawn
        assert abs(score) < 50
    
    def test_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        # Distance between e4 (28) and e6 (44)
        distance = self.player._manhattan_distance(chess.E4, chess.E6)
        assert distance == 2
        
        # Distance between a1 (0) and h8 (63)
        distance = self.player._manhattan_distance(chess.A1, chess.H8)
        assert distance == 14  # 7 files + 7 ranks
    
    def test_is_opposition_direct(self):
        """Test direct opposition detection."""
        # Kings on e4 and e6 (direct opposition)
        assert self.player._is_opposition(chess.E4, chess.E6)
        
        # Kings on d4 and f4 (horizontal opposition, 2 squares apart)
        assert self.player._is_opposition(chess.D4, chess.F4)
        
        # Kings on d4 and e5 (not opposition - diagonal)
        assert not self.player._is_opposition(chess.D4, chess.E5)
    
    def test_is_opposition_distant(self):
        """Test distant opposition detection."""
        # Kings on e2 and e8 (distant opposition)
        assert self.player._is_opposition(chess.E2, chess.E8)
        
        # Kings on e2 and e7 (not opposition - odd distance)
        assert not self.player._is_opposition(chess.E2, chess.E7)
    
    def test_king_centrality(self):
        """Test king centrality calculation."""
        # Center squares should have high centrality
        center_score = self.player._king_centrality(chess.E4)
        corner_score = self.player._king_centrality(chess.A1)
        
        assert center_score > corner_score


class TestMinimaxPlayer:
    """Test cases for MinimaxPlayer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.player = MinimaxPlayer(depth=2)
    
    def test_initialization(self):
        """Test minimax player initialization."""
        assert "Minimax Player (depth 2)" in self.player.name
        assert self.player.estimated_elo == 1400.0  # 1200 + 2*100
        assert self.player.depth == 2
    
    def test_initialization_different_depth(self):
        """Test minimax player with different depth."""
        player = MinimaxPlayer(depth=4)
        assert self.player.depth == 2
        assert player.depth == 4
        assert player.estimated_elo == 1600.0  # 1200 + 4*100
    
    def test_get_move_with_legal_moves(self):
        """Test getting move when legal moves are available."""
        position = Position(chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"))
        
        move = self.player.get_move(position)
        
        assert move is not None
        assert move in position.get_legal_moves()
    
    def test_get_move_no_legal_moves(self):
        """Test getting move when no legal moves available."""
        # Mock position with no legal moves
        position = Mock()
        position.get_legal_moves.return_value = []
        
        move = self.player.get_move(position)
        
        assert move is None
    
    def test_minimax_terminal_position(self):
        """Test minimax evaluation of terminal position."""
        # Mock terminal position
        position = Mock()
        position.is_terminal.return_value = True
        position.get_result.return_value = 1.0
        
        score = self.player._minimax(position, 0, True, -float('inf'), float('inf'))
        
        assert score == 1000.0  # Large bonus for winning
    
    def test_minimax_depth_zero(self):
        """Test minimax evaluation at depth zero."""
        position = Position(chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"))
        
        score = self.player._minimax(position, 0, True, -float('inf'), float('inf'))
        
        assert isinstance(score, float)
    
    def test_minimax_with_depth(self):
        """Test minimax search with depth > 0."""
        position = Position(chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"))
        
        # Should not raise exception and return a score
        score = self.player._minimax(position, 1, True, -float('inf'), float('inf'))
        
        assert isinstance(score, float)


class TestStandardOpponents:
    """Test cases for standard opponents list."""
    
    def test_get_standard_opponents(self):
        """Test getting standard opponents list."""
        opponents = get_standard_opponents()
        
        assert len(opponents) > 0
        assert all(isinstance(opp, BenchmarkOpponent) for opp in opponents)
        
        # Check that opponents are in order of increasing strength
        elos = [opp.estimated_elo for opp in opponents]
        assert elos == sorted(elos)
        
        # Check that we have different types of opponents
        opponent_types = [type(opp).__name__ for opp in opponents]
        assert 'RandomPlayer' in opponent_types
        assert 'SimpleHeuristicPlayer' in opponent_types
        assert 'MinimaxPlayer' in opponent_types


class TestMatchResult:
    """Test cases for MatchResult class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.result = MatchResult("Player 1", "Player 2")
    
    def test_initialization(self):
        """Test match result initialization."""
        assert self.result.player1_name == "Player 1"
        assert self.result.player2_name == "Player 2"
        assert self.result.player1_wins == 0
        assert self.result.player2_wins == 0
        assert self.result.draws == 0
        assert self.result.total_games == 0
    
    def test_add_win_result(self):
        """Test adding win result."""
        self.result.add_result(1.0)
        
        assert self.result.player1_wins == 1
        assert self.result.player2_wins == 0
        assert self.result.draws == 0
        assert self.result.total_games == 1
    
    def test_add_loss_result(self):
        """Test adding loss result."""
        self.result.add_result(-1.0)
        
        assert self.result.player1_wins == 0
        assert self.result.player2_wins == 1
        assert self.result.draws == 0
        assert self.result.total_games == 1
    
    def test_add_draw_result(self):
        """Test adding draw result."""
        self.result.add_result(0.0)
        
        assert self.result.player1_wins == 0
        assert self.result.player2_wins == 0
        assert self.result.draws == 1
        assert self.result.total_games == 1
    
    def test_get_score_no_games(self):
        """Test getting score with no games played."""
        score = self.result.get_score()
        assert score == 0.5
    
    def test_get_score_all_wins(self):
        """Test getting score with all wins."""
        self.result.add_result(1.0)
        self.result.add_result(1.0)
        
        score = self.result.get_score()
        assert score == 1.0
    
    def test_get_score_all_losses(self):
        """Test getting score with all losses."""
        self.result.add_result(-1.0)
        self.result.add_result(-1.0)
        
        score = self.result.get_score()
        assert score == 0.0
    
    def test_get_score_mixed_results(self):
        """Test getting score with mixed results."""
        self.result.add_result(1.0)  # Win
        self.result.add_result(0.0)  # Draw
        self.result.add_result(-1.0)  # Loss
        
        score = self.result.get_score()
        assert score == 0.5  # (1 + 0.5 + 0) / 3
    
    def test_str_representation(self):
        """Test string representation of match result."""
        self.result.add_result(1.0)
        self.result.add_result(0.0)
        self.result.add_result(-1.0)
        
        result_str = str(self.result)
        
        assert "Player 1 vs Player 2" in result_str
        assert "1-1-1" in result_str
        assert "0.500" in result_str


class TestMatchPlaying:
    """Test cases for match playing functions."""
    
    def test_play_game_terminal_position(self):
        """Test playing game from terminal position."""
        # Mock terminal position
        position = Mock()
        position.is_terminal.return_value = True
        position.get_result.return_value = 1.0
        position.board.turn = chess.WHITE
        
        player1 = RandomPlayer()
        player2 = SimpleHeuristicPlayer()
        
        result = _play_game(position, player1, player2)
        
        # Result should be from white's perspective
        assert abs(result) == 1.0  # Either 1.0 or -1.0 depending on perspective
    
    def test_play_game_max_moves_reached(self):
        """Test playing game until max moves reached."""
        position = Position(chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"))
        
        player1 = RandomPlayer()
        player2 = SimpleHeuristicPlayer()
        
        # Play with very low max moves
        result = _play_game(position, player1, player2, max_moves=2)
        
        assert result == 0.0  # Draw due to max moves
    
    def test_play_match_basic(self):
        """Test playing basic match between opponents."""
        player1 = RandomPlayer()
        player2 = SimpleHeuristicPlayer()
        
        # Create simple starting positions
        positions = [
            Position(chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1")),
            Position(chess.Board("8/8/4k3/4P3/4K3/8/8/8 b - - 0 1"))
        ]
        
        result = play_match(player1, player2, num_games=2, starting_positions=positions)
        
        assert isinstance(result, MatchResult)
        assert result.total_games == 2
        assert result.player1_name == player1.name
        assert result.player2_name == player2.name
    
    def test_play_match_no_starting_positions(self):
        """Test playing match without providing starting positions."""
        player1 = RandomPlayer()
        player2 = SimpleHeuristicPlayer()
        
        # Should use standard test suite positions
        result = play_match(player1, player2, num_games=2)
        
        assert isinstance(result, MatchResult)
        assert result.total_games == 2
    
    def test_play_match_alternating_colors(self):
        """Test that match alternates colors properly."""
        # This is more of an integration test to ensure the logic works
        player1 = RandomPlayer()
        player2 = SimpleHeuristicPlayer()
        
        positions = [Position(chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"))]
        
        result = play_match(player1, player2, num_games=4, starting_positions=positions)
        
        assert result.total_games == 4
        # Can't easily test color alternation without mocking, but at least verify it runs