"""Benchmark opponents for evaluating chess engine performance."""

from abc import ABC, abstractmethod
from typing import Optional, List
import random
import chess
import math

from ..chess_env.position import Position


class BenchmarkOpponent(ABC):
    """Abstract base class for benchmark opponents."""
    
    def __init__(self, name: str, estimated_elo: float):
        """Initialize benchmark opponent.
        
        Args:
            name: Name of the opponent
            estimated_elo: Estimated Elo rating of the opponent
        """
        self.name = name
        self.estimated_elo = estimated_elo
    
    @abstractmethod
    def get_move(self, position: Position) -> Optional[chess.Move]:
        """Get the opponent's move for the given position.
        
        Args:
            position: Current chess position
            
        Returns:
            Chosen move, or None if no legal moves available
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the opponent."""
        return f"{self.name} (Elo: {self.estimated_elo})"


class RandomPlayer(BenchmarkOpponent):
    """Random move player for sanity check baseline."""
    
    def __init__(self):
        """Initialize random player."""
        super().__init__("Random Player", 800.0)
        self.random = random.Random(42)  # Fixed seed for reproducibility
    
    def get_move(self, position: Position) -> Optional[chess.Move]:
        """Get random legal move.
        
        Args:
            position: Current chess position
            
        Returns:
            Random legal move, or None if no legal moves
        """
        legal_moves = position.get_legal_moves()
        if not legal_moves:
            return None
        
        return self.random.choice(legal_moves)


class SimpleHeuristicPlayer(BenchmarkOpponent):
    """Simple heuristic player using material counting and basic positional factors."""
    
    def __init__(self):
        """Initialize simple heuristic player."""
        super().__init__("Simple Heuristic Player", 1000.0)
    
    def get_move(self, position: Position) -> Optional[chess.Move]:
        """Get move using simple heuristic evaluation.
        
        Args:
            position: Current chess position
            
        Returns:
            Best move according to simple heuristic, or None if no legal moves
        """
        legal_moves = position.get_legal_moves()
        if not legal_moves:
            return None
        
        best_move = None
        best_score = -float('inf')
        
        for move in legal_moves:
            # Make move and evaluate resulting position
            new_position = position.make_move(move)
            score = self._evaluate_position(new_position)
            
            # Negate score since we want the best move for current player
            score = -score
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move or legal_moves[0]
    
    def _evaluate_position(self, position: Position) -> float:
        """Evaluate position using simple heuristics.
        
        Args:
            position: Position to evaluate
            
        Returns:
            Evaluation score from current player's perspective
        """
        board = position.board
        
        # Terminal position check
        if position.is_terminal():
            result = position.get_result()
            return result * 1000  # Large bonus/penalty for terminal positions
        
        score = 0.0
        
        # Find pieces
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        # Find white pawn
        white_pawn = None
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                white_pawn = square
                break
        
        if white_pawn is not None:
            # Pawn advancement bonus
            pawn_rank = chess.square_rank(white_pawn)
            score += pawn_rank * 20  # 20 points per rank advanced
            
            # King support evaluation
            white_king_distance = self._manhattan_distance(white_king, white_pawn)
            black_king_distance = self._manhattan_distance(black_king, white_pawn)
            
            # Prefer white king close to pawn, black king far from pawn
            score += (8 - white_king_distance) * 10
            score += black_king_distance * 5
            
            # Opposition evaluation (simplified)
            if self._is_opposition(white_king, black_king):
                if board.turn == chess.WHITE:
                    score += 30  # Good for white if white to move
                else:
                    score -= 30  # Bad for white if black to move
            
            # Key square control (simplified)
            pawn_file = chess.square_file(white_pawn)
            key_squares = [
                chess.square(pawn_file, pawn_rank + 1),
                chess.square(pawn_file, pawn_rank + 2)
            ]
            
            for key_square in key_squares:
                if 0 <= key_square <= 63:
                    if self._controls_square(board, chess.WHITE, key_square):
                        score += 15
                    elif self._controls_square(board, chess.BLACK, key_square):
                        score -= 15
        
        # King activity (centralization)
        white_king_centrality = self._king_centrality(white_king)
        black_king_centrality = self._king_centrality(black_king)
        score += (white_king_centrality - black_king_centrality) * 5
        
        # Adjust for current turn
        if board.turn == chess.BLACK:
            score = -score
        
        return score
    
    def _manhattan_distance(self, square1: int, square2: int) -> int:
        """Calculate Manhattan distance between two squares.
        
        Args:
            square1: First square
            square2: Second square
            
        Returns:
            Manhattan distance
        """
        file1, rank1 = chess.square_file(square1), chess.square_rank(square1)
        file2, rank2 = chess.square_file(square2), chess.square_rank(square2)
        return abs(file1 - file2) + abs(rank1 - rank2)
    
    def _is_opposition(self, white_king: int, black_king: int) -> bool:
        """Check if kings are in opposition.
        
        Args:
            white_king: White king square
            black_king: Black king square
            
        Returns:
            True if kings are in opposition
        """
        file_diff = abs(chess.square_file(white_king) - chess.square_file(black_king))
        rank_diff = abs(chess.square_rank(white_king) - chess.square_rank(black_king))
        
        # Direct opposition: same file/rank, 2 squares apart
        # Distant opposition: same file/rank, even number of squares apart
        return ((file_diff == 0 and rank_diff % 2 == 0 and rank_diff >= 2) or
                (rank_diff == 0 and file_diff % 2 == 0 and file_diff >= 2))
    
    def _controls_square(self, board: chess.Board, color: chess.Color, square: int) -> bool:
        """Check if a color controls a square.
        
        Args:
            board: Chess board
            color: Color to check
            square: Square to check
            
        Returns:
            True if color controls the square
        """
        # Simple check: can the king move to this square?
        king = board.king(color)
        if king is None:
            return False
        
        king_distance = self._manhattan_distance(king, square)
        return king_distance <= 1
    
    def _king_centrality(self, king_square: int) -> float:
        """Calculate king centrality score.
        
        Args:
            king_square: King's square
            
        Returns:
            Centrality score (higher is more central)
        """
        file = chess.square_file(king_square)
        rank = chess.square_rank(king_square)
        
        # Distance from center (3.5, 3.5)
        center_distance = abs(file - 3.5) + abs(rank - 3.5)
        return 7 - center_distance  # Higher score for more central positions


class MinimaxPlayer(BenchmarkOpponent):
    """Minimax player with configurable depth."""
    
    def __init__(self, depth: int = 3):
        """Initialize minimax player.
        
        Args:
            depth: Search depth for minimax algorithm
        """
        estimated_elo = 1200 + depth * 100  # Rough Elo estimate based on depth
        super().__init__(f"Minimax Player (depth {depth})", estimated_elo)
        self.depth = depth
        self.heuristic_player = SimpleHeuristicPlayer()
    
    def get_move(self, position: Position) -> Optional[chess.Move]:
        """Get move using minimax search.
        
        Args:
            position: Current chess position
            
        Returns:
            Best move according to minimax search, or None if no legal moves
        """
        legal_moves = position.get_legal_moves()
        if not legal_moves:
            return None
        
        best_move = None
        best_score = -float('inf')
        
        for move in legal_moves:
            new_position = position.make_move(move)
            score = self._minimax(new_position, self.depth - 1, False, -float('inf'), float('inf'))
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move or legal_moves[0]
    
    def _minimax(self, position: Position, depth: int, maximizing: bool, 
                alpha: float, beta: float) -> float:
        """Minimax search with alpha-beta pruning.
        
        Args:
            position: Current position
            depth: Remaining search depth
            maximizing: True if maximizing player's turn
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            
        Returns:
            Position evaluation score
        """
        # Terminal node or depth limit reached
        if depth == 0 or position.is_terminal():
            score = self.heuristic_player._evaluate_position(position)
            return score if maximizing else -score
        
        legal_moves = position.get_legal_moves()
        if not legal_moves:
            # No legal moves - terminal position
            result = position.get_result()
            return result * 1000 if maximizing else -result * 1000
        
        if maximizing:
            max_score = -float('inf')
            for move in legal_moves:
                new_position = position.make_move(move)
                score = self._minimax(new_position, depth - 1, False, alpha, beta)
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return max_score
        else:
            min_score = float('inf')
            for move in legal_moves:
                new_position = position.make_move(move)
                score = self._minimax(new_position, depth - 1, True, alpha, beta)
                min_score = min(min_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return min_score


def get_standard_opponents() -> List[BenchmarkOpponent]:
    """Get list of standard benchmark opponents.
    
    Returns:
        List of benchmark opponents in order of increasing strength
    """
    return [
        RandomPlayer(),
        SimpleHeuristicPlayer(),
        MinimaxPlayer(depth=2),
        MinimaxPlayer(depth=3),
        MinimaxPlayer(depth=4),
        MinimaxPlayer(depth=5)
    ]


class MatchResult:
    """Result of a match between two players."""
    
    def __init__(self, player1_name: str, player2_name: str):
        """Initialize match result.
        
        Args:
            player1_name: Name of first player
            player2_name: Name of second player
        """
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.player1_wins = 0
        self.player2_wins = 0
        self.draws = 0
        self.total_games = 0
    
    def add_result(self, result: float) -> None:
        """Add game result from player1's perspective.
        
        Args:
            result: Game result (1.0=win, 0.0=draw, -1.0=loss)
        """
        self.total_games += 1
        if result > 0.5:
            self.player1_wins += 1
        elif result < -0.5:
            self.player2_wins += 1
        else:
            self.draws += 1
    
    def get_score(self) -> float:
        """Get player1's score as percentage.
        
        Returns:
            Score from 0.0 to 1.0
        """
        if self.total_games == 0:
            return 0.5
        return (self.player1_wins + 0.5 * self.draws) / self.total_games
    
    def __str__(self) -> str:
        """String representation of match result."""
        score = self.get_score()
        return (f"{self.player1_name} vs {self.player2_name}: "
                f"{self.player1_wins}-{self.draws}-{self.player2_wins} "
                f"(Score: {score:.3f})")


def play_match(player1: BenchmarkOpponent, player2: BenchmarkOpponent,
               num_games: int = 10, starting_positions: Optional[List[Position]] = None) -> MatchResult:
    """Play a match between two opponents.
    
    Args:
        player1: First player
        player2: Second player
        num_games: Number of games to play
        starting_positions: Optional list of starting positions
        
    Returns:
        Match result
    """
    from ..evaluation.test_suite import TestSuite
    
    result = MatchResult(player1.name, player2.name)
    
    # Use standard test positions if none provided
    if starting_positions is None:
        test_suite = TestSuite.generate_standard_suite()
        starting_positions = [pos.position for pos in test_suite.positions[:num_games]]
    
    # Ensure we have enough positions
    while len(starting_positions) < num_games:
        starting_positions.extend(starting_positions)
    
    for i in range(num_games):
        position = starting_positions[i % len(starting_positions)]
        
        # Alternate who plays white
        if i % 2 == 0:
            white_player, black_player = player1, player2
            white_is_player1 = True
        else:
            white_player, black_player = player2, player1
            white_is_player1 = False
        
        # Play the game
        game_result = _play_game(position, white_player, black_player)
        
        # Adjust result for player1's perspective
        if not white_is_player1:
            game_result = -game_result
        
        result.add_result(game_result)
    
    return result


def _play_game(starting_position: Position, white_player: BenchmarkOpponent,
               black_player: BenchmarkOpponent, max_moves: int = 200) -> float:
    """Play a single game between two opponents.
    
    Args:
        starting_position: Starting position
        white_player: Player playing white
        black_player: Player playing black
        max_moves: Maximum number of moves
        
    Returns:
        Game result from white's perspective (1.0=win, 0.0=draw, -1.0=loss)
    """
    position = Position(starting_position.board)  # Copy position
    move_count = 0
    
    while not position.is_terminal() and move_count < max_moves:
        # Determine current player
        current_player = white_player if position.board.turn == chess.WHITE else black_player
        
        # Get move
        move = current_player.get_move(position)
        if move is None:
            break
        
        # Make move
        position = position.make_move(move)
        move_count += 1
    
    if position.is_terminal():
        result = position.get_result()
        # Adjust for white's perspective
        if position.board.turn == chess.BLACK:
            result = -result
        return result
    else:
        # Game didn't finish - consider it a draw
        return 0.0