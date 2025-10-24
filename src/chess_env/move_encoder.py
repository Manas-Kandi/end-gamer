"""Move encoder for converting between chess moves and neural network format."""

from typing import List
import numpy as np
import chess
from .position import Position


class MoveEncoder:
    """Encode and decode chess moves for neural network processing.
    
    Uses a simple from-to representation where each move is encoded as:
    move_index = from_square * 64 + to_square
    
    This gives us 64 * 64 = 4096 possible moves, covering all from-to combinations.
    """
    
    @staticmethod
    def encode_move(move: chess.Move) -> int:
        """Convert chess.Move to integer index [0, 4095].
        
        Args:
            move: chess.Move object to encode
            
        Returns:
            Integer index in range [0, 4095]
            
        Example:
            move from e2 to e4: from_square=12, to_square=28
            encoded = 12 * 64 + 28 = 796
        """
        return move.from_square * 64 + move.to_square
    
    @staticmethod
    def decode_move(move_idx: int) -> chess.Move:
        """Convert integer index to chess.Move.
        
        Args:
            move_idx: Integer index in range [0, 4095]
            
        Returns:
            chess.Move object
            
        Raises:
            ValueError: If move_idx is out of valid range
        """
        if not (0 <= move_idx < 4096):
            raise ValueError(f"Move index {move_idx} must be in range [0, 4095]")
        
        from_square = move_idx // 64
        to_square = move_idx % 64
        
        return chess.Move(from_square, to_square)
    
    @staticmethod
    def get_move_mask(position: Position) -> np.ndarray:
        """Get binary mask for legal moves in position.
        
        Args:
            position: Position object to get legal moves for
            
        Returns:
            Binary numpy array of shape (4096,) where 1 indicates legal move
        """
        mask = np.zeros(4096, dtype=np.float32)
        
        legal_moves = position.get_legal_moves()
        for move in legal_moves:
            move_idx = MoveEncoder.encode_move(move)
            mask[move_idx] = 1.0
        
        return mask
    
    @staticmethod
    def get_legal_move_indices(position: Position) -> List[int]:
        """Get list of legal move indices for position.
        
        Args:
            position: Position object to get legal moves for
            
        Returns:
            List of integer indices for legal moves
        """
        legal_moves = position.get_legal_moves()
        return [MoveEncoder.encode_move(move) for move in legal_moves]
    
    @staticmethod
    def moves_to_policy_vector(moves: List[chess.Move], 
                              visit_counts: List[int]) -> np.ndarray:
        """Convert moves and visit counts to policy vector.
        
        Args:
            moves: List of chess.Move objects
            visit_counts: List of visit counts for each move
            
        Returns:
            Policy vector of shape (4096,) with probabilities
        """
        if len(moves) != len(visit_counts):
            raise ValueError("Moves and visit_counts must have same length")
        
        policy = np.zeros(4096, dtype=np.float32)
        total_visits = sum(visit_counts)
        
        if total_visits == 0:
            return policy
        
        for move, count in zip(moves, visit_counts):
            move_idx = MoveEncoder.encode_move(move)
            policy[move_idx] = count / total_visits
        
        return policy
    
    @staticmethod
    def policy_vector_to_move(policy: np.ndarray, position: Position, 
                             temperature: float = 1.0) -> chess.Move:
        """Sample move from policy vector.
        
        Args:
            policy: Policy vector of shape (4096,)
            position: Position to validate legal moves
            temperature: Temperature for sampling (0 = greedy, >1 = more random)
            
        Returns:
            Sampled chess.Move
            
        Raises:
            ValueError: If no legal moves have positive probability
        """
        # Get legal moves and their probabilities
        legal_moves = position.get_legal_moves()
        legal_probs = []
        
        for move in legal_moves:
            move_idx = MoveEncoder.encode_move(move)
            legal_probs.append(policy[move_idx])
        
        legal_probs = np.array(legal_probs)
        
        if np.sum(legal_probs) == 0:
            raise ValueError("No legal moves have positive probability")
        
        # Apply temperature
        if temperature == 0:
            # Greedy selection
            best_idx = np.argmax(legal_probs)
            return legal_moves[best_idx]
        else:
            # Temperature sampling
            legal_probs = legal_probs ** (1.0 / temperature)
            legal_probs = legal_probs / np.sum(legal_probs)
            
            # Sample from distribution
            move_idx = np.random.choice(len(legal_moves), p=legal_probs)
            return legal_moves[move_idx]