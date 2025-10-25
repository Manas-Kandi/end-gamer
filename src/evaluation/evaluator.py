"""Evaluator class for measuring chess engine performance."""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
import chess
import math
import random
from collections import defaultdict

from .test_suite import TestSuite, TestPosition, ExpectedResult, PositionDifficulty
from .tablebase import TablebaseInterface
from ..config.config import Config
from ..mcts.mcts import MCTS
from ..chess_env.position import Position
from ..chess_env.move_encoder import MoveEncoder


class Evaluator:
    """Evaluate model performance against test positions and benchmarks."""
    
    def __init__(self, config: Config, tablebase_path: Optional[str] = None):
        """Initialize evaluator.
        
        Args:
            config: Configuration object
            tablebase_path: Optional path to Syzygy tablebase files
        """
        self.config = config
        self.test_suite = TestSuite.generate_standard_suite()
        self.tablebase = TablebaseInterface(tablebase_path)
        
        # Cache for repeated evaluations
        self._position_cache: Dict[str, float] = {}
    
    def evaluate(self, neural_net: torch.nn.Module) -> Dict[str, float]:
        """Run comprehensive evaluation.
        
        Args:
            neural_net: Neural network to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        # Set neural network to evaluation mode
        neural_net.eval()
        
        # Create MCTS instance for evaluation
        mcts = MCTS(
            neural_net=neural_net,
            num_simulations=self.config.mcts_simulations,
            c_puct=self.config.c_puct,
            device=self.config.device
        )
        
        print("Evaluating win rate in winning positions...")
        metrics['win_rate'] = self._evaluate_win_rate(mcts)
        
        print("Evaluating draw rate in drawn positions...")
        metrics['draw_rate'] = self._evaluate_draw_rate(mcts)
        
        print("Evaluating move accuracy...")
        metrics['move_accuracy'] = self._evaluate_move_accuracy(mcts)
        
        print("Estimating Elo rating...")
        metrics['elo_estimate'] = self._estimate_elo(mcts)
        
        # Additional detailed metrics
        metrics.update(self._get_detailed_metrics(mcts))
        
        return metrics
    
    def _evaluate_win_rate(self, mcts: MCTS) -> float:
        """Test win rate in theoretically winning positions.
        
        Args:
            mcts: MCTS instance for move selection
            
        Returns:
            Win rate as percentage (0.0 to 1.0)
        """
        winning_positions = self.test_suite.get_positions_by_result(ExpectedResult.WIN)
        if not winning_positions:
            return 0.0
        
        wins = 0
        total = 0
        
        for test_pos in winning_positions:
            result = self._play_position_to_end(test_pos.position, mcts, max_moves=100)
            if result > 0.5:  # Consider > 0.5 as win (accounting for draws)
                wins += 1
            total += 1
        
        return wins / total if total > 0 else 0.0
    
    def _evaluate_draw_rate(self, mcts: MCTS) -> float:
        """Test draw rate in theoretical draw positions.
        
        Args:
            mcts: MCTS instance for move selection
            
        Returns:
            Draw rate as percentage (0.0 to 1.0)
        """
        drawn_positions = self.test_suite.get_positions_by_result(ExpectedResult.DRAW)
        if not drawn_positions:
            return 0.0
        
        draws = 0
        total = 0
        
        for test_pos in drawn_positions:
            result = self._play_position_to_end(test_pos.position, mcts, max_moves=100)
            if abs(result) < 0.1:  # Consider close to 0 as draw
                draws += 1
            total += 1
        
        return draws / total if total > 0 else 0.0
    
    def _evaluate_move_accuracy(self, mcts: MCTS) -> float:
        """Compare moves with tablebase optimal moves.
        
        Args:
            mcts: MCTS instance for move selection
            
        Returns:
            Move accuracy as percentage (0.0 to 1.0)
        """
        correct_moves = 0
        total = 0
        
        # Test on a subset of positions to avoid long evaluation times
        test_positions = self.test_suite.positions[:10]  # First 10 positions
        
        for test_pos in test_positions:
            if test_pos.position.is_terminal():
                continue
                
            # Get model's preferred move
            policy = mcts.search(test_pos.position)
            model_move = self._get_best_move_from_policy(policy, test_pos.position)
            
            # Try to get optimal move from tablebase, fallback to heuristic
            if self.tablebase.is_available():
                optimal_move = self.tablebase.get_best_move(test_pos.position)
                if optimal_move is None:
                    # Position not in tablebase, use heuristic
                    optimal_move = self._get_heuristic_best_move(test_pos.position)
            else:
                # Tablebase not available, use heuristic
                optimal_move = self._get_heuristic_best_move(test_pos.position)
            
            if model_move == optimal_move:
                correct_moves += 1
            total += 1
        
        return correct_moves / total if total > 0 else 0.0
    
    def _play_position_to_end(self, position: Position, mcts: MCTS, 
                             max_moves: int = 200) -> float:
        """Play position until terminal or max moves.
        
        Args:
            position: Starting position
            mcts: MCTS instance for move selection
            max_moves: Maximum number of moves to play
            
        Returns:
            Game result from original position perspective (1.0=win, 0.0=draw, -1.0=loss)
        """
        current_pos = Position(position.board)  # Copy position
        move_count = 0
        original_turn = position.board.turn
        
        while not current_pos.is_terminal() and move_count < max_moves:
            # Get move from MCTS
            policy = mcts.search(current_pos)
            move = self._get_best_move_from_policy(policy, current_pos)
            
            if move is None:
                break
                
            # Make move
            current_pos = current_pos.make_move(move)
            move_count += 1
        
        if current_pos.is_terminal():
            result = current_pos.get_result()
            # Adjust result based on original turn
            if original_turn != current_pos.board.turn:
                result = -result
            return result
        else:
            # Game didn't finish - try tablebase, fallback to heuristic
            if self.tablebase.is_available():
                tb_result = self.tablebase.probe(current_pos)
                if tb_result is not None:
                    # Adjust result based on original turn
                    if original_turn != current_pos.board.turn:
                        tb_result = -tb_result
                    return tb_result
            
            # Fallback to heuristic evaluation
            return self._heuristic_evaluation(current_pos)
    
    def _get_best_move_from_policy(self, policy: np.ndarray, position: Position) -> Optional[chess.Move]:
        """Get best move from policy distribution.
        
        Args:
            policy: Policy vector of shape (4096,)
            position: Current position
            
        Returns:
            Best legal move, or None if no legal moves
        """
        legal_moves = position.get_legal_moves()
        if not legal_moves:
            return None
        
        best_move = None
        best_prob = -1.0
        
        for move in legal_moves:
            move_idx = MoveEncoder.encode_move(move)
            if policy[move_idx] > best_prob:
                best_prob = policy[move_idx]
                best_move = move
        
        return best_move
    
    def _get_heuristic_best_move(self, position: Position) -> Optional[chess.Move]:
        """Get heuristic best move (placeholder for tablebase).
        
        Args:
            position: Position to analyze
            
        Returns:
            Heuristically best move
        """
        legal_moves = position.get_legal_moves()
        if not legal_moves:
            return None
        
        # Simple heuristic: prefer pawn advances and king moves toward center
        best_move = None
        best_score = -float('inf')
        
        for move in legal_moves:
            score = 0
            
            # Prefer pawn moves
            if position.board.piece_at(move.from_square).piece_type == chess.PAWN:
                score += 10
                # Prefer advancing pawn
                if move.to_square > move.from_square:
                    score += 5
            
            # Prefer king centralization
            if position.board.piece_at(move.from_square).piece_type == chess.KING:
                to_file = chess.square_file(move.to_square)
                to_rank = chess.square_rank(move.to_square)
                # Distance from center
                center_distance = abs(to_file - 3.5) + abs(to_rank - 3.5)
                score += (7 - center_distance)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move or legal_moves[0]
    
    def _heuristic_evaluation(self, position: Position) -> float:
        """Simple heuristic evaluation for non-terminal positions.
        
        Args:
            position: Position to evaluate
            
        Returns:
            Evaluation from current player perspective
        """
        # Simple material and king activity evaluation
        board = position.board
        
        # Find kings and pawn
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        # Find white pawn
        white_pawn = None
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                white_pawn = square
                break
        
        if white_pawn is None:
            return 0.0  # No pawn, likely draw
        
        # Evaluate based on pawn advancement and king positions
        pawn_rank = chess.square_rank(white_pawn)
        pawn_file = chess.square_file(white_pawn)
        
        white_king_file = chess.square_file(white_king)
        white_king_rank = chess.square_rank(white_king)
        black_king_file = chess.square_file(black_king)
        black_king_rank = chess.square_rank(black_king)
        
        # Pawn advancement bonus
        score = pawn_rank * 0.2
        
        # King support bonus
        king_pawn_distance = abs(white_king_file - pawn_file) + abs(white_king_rank - pawn_rank)
        score += max(0, 4 - king_pawn_distance) * 0.1
        
        # Black king distance penalty (for white)
        black_king_pawn_distance = abs(black_king_file - pawn_file) + abs(black_king_rank - pawn_rank)
        score += black_king_pawn_distance * 0.05
        
        # Adjust for current turn
        if board.turn == chess.BLACK:
            score = -score
        
        return max(-1.0, min(1.0, score))
    
    def _estimate_elo(self, mcts: MCTS) -> float:
        """Estimate Elo rating through matches against calibrated opponents.
        
        Args:
            mcts: MCTS instance for evaluation
            
        Returns:
            Estimated Elo rating
        """
        from .benchmark_opponents import get_standard_opponents, play_match
        
        # Create a wrapper for the neural network + MCTS
        class NeuralNetworkPlayer:
            def __init__(self, mcts_engine):
                self.mcts = mcts_engine
                self.name = "Neural Network"
                self.estimated_elo = 1500.0  # Initial estimate
            
            def get_move(self, position):
                policy = self.mcts.search(position)
                return self._get_best_move_from_policy(policy, position)
            
            def _get_best_move_from_policy(self, policy, position):
                legal_moves = position.get_legal_moves()
                if not legal_moves:
                    return None
                
                best_move = None
                best_prob = -1.0
                
                for move in legal_moves:
                    from ..chess_env.move_encoder import MoveEncoder
                    move_idx = MoveEncoder.encode_move(move)
                    if policy[move_idx] > best_prob:
                        best_prob = policy[move_idx]
                        best_move = move
                
                return best_move
        
        nn_player = NeuralNetworkPlayer(mcts)
        opponents = get_standard_opponents()
        
        # Play matches against opponents and collect results
        match_results = []
        
        # Limit to first few opponents for performance
        for opponent in opponents[:3]:  # Random, Heuristic, Minimax depth 2
            print(f"Playing match against {opponent.name}...")
            result = play_match(nn_player, opponent, num_games=6)  # Reduced for speed
            match_results.append((opponent.estimated_elo, result.get_score()))
        
        # Calculate Elo based on match results
        estimated_elo = self._calculate_elo_from_results(match_results)
        
        return estimated_elo
    
    def _calculate_elo_from_results(self, match_results: List[Tuple[float, float]]) -> float:
        """Calculate Elo rating from match results against known opponents.
        
        Args:
            match_results: List of (opponent_elo, score) tuples
            
        Returns:
            Estimated Elo rating
        """
        if not match_results:
            return 1200.0  # Default rating
        
        # Use logistic regression approach to estimate Elo
        total_weight = 0.0
        weighted_elo_sum = 0.0
        
        for opponent_elo, score in match_results:
            # Convert score to Elo difference using logistic formula
            # score = 1 / (1 + 10^((opponent_elo - player_elo) / 400))
            # Solving for player_elo:
            # player_elo = opponent_elo + 400 * log10(score / (1 - score))
            
            if score <= 0.0:
                score = 0.01  # Avoid log(0)
            elif score >= 1.0:
                score = 0.99  # Avoid log(inf)
            
            elo_diff = 400 * math.log10(score / (1 - score))
            estimated_elo = opponent_elo + elo_diff
            
            # Weight by number of games (assume equal weight for now)
            weight = 1.0
            weighted_elo_sum += estimated_elo * weight
            total_weight += weight
        
        if total_weight == 0:
            return 1200.0
        
        final_elo = weighted_elo_sum / total_weight
        
        # Clamp to reasonable range
        return max(800.0, min(2400.0, final_elo))
    
    def _get_detailed_metrics(self, mcts: MCTS) -> Dict[str, float]:
        """Get additional detailed evaluation metrics.
        
        Args:
            mcts: MCTS instance for evaluation
            
        Returns:
            Dictionary with detailed metrics
        """
        metrics = {}
        
        # Performance by difficulty
        for difficulty in PositionDifficulty:
            positions = self.test_suite.get_positions_by_difficulty(difficulty)
            if positions:
                correct = 0
                total = 0
                
                for test_pos in positions:
                    if test_pos.expected_result == ExpectedResult.WIN:
                        result = self._play_position_to_end(test_pos.position, mcts, max_moves=50)
                        if result > 0.5:
                            correct += 1
                    elif test_pos.expected_result == ExpectedResult.DRAW:
                        result = self._play_position_to_end(test_pos.position, mcts, max_moves=50)
                        if abs(result) < 0.1:
                            correct += 1
                    total += 1
                
                metrics[f'{difficulty.value}_accuracy'] = correct / total if total > 0 else 0.0
        
        # Performance by key concepts
        concept_performance = defaultdict(list)
        for test_pos in self.test_suite.positions[:8]:  # Limit for performance
            result = self._play_position_to_end(test_pos.position, mcts, max_moves=30)
            
            # Check if result matches expectation
            expected_correct = False
            if test_pos.expected_result == ExpectedResult.WIN and result > 0.5:
                expected_correct = True
            elif test_pos.expected_result == ExpectedResult.DRAW and abs(result) < 0.1:
                expected_correct = True
            elif test_pos.expected_result == ExpectedResult.LOSS and result < -0.5:
                expected_correct = True
            
            for concept in test_pos.key_concepts:
                concept_performance[concept].append(1.0 if expected_correct else 0.0)
        
        # Average performance by concept
        for concept, results in concept_performance.items():
            if results:
                metrics[f'concept_{concept}'] = sum(results) / len(results)
        
        return metrics