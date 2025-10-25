"""Monte Carlo Tree Search implementation."""

from typing import Tuple, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
import chess
import time
import logging

from .mcts_node import MCTSNode
from ..chess_env.position import Position
from ..chess_env.move_encoder import MoveEncoder
from ..exceptions import SearchTimeoutError

logger = logging.getLogger(__name__)


class MCTS:
    """Monte Carlo Tree Search for chess move selection.
    
    Implements the MCTS algorithm with neural network guidance for position
    evaluation and move priors. Uses UCB1 for node selection and combines
    neural network evaluation with tree search statistics.
    """
    
    def __init__(self, neural_net: torch.nn.Module, num_simulations: int = 400,
                 c_puct: float = 1.0, device: str = 'cuda', timeout: Optional[float] = None):
        """Initialize MCTS.
        
        Args:
            neural_net: Neural network for position evaluation and move priors
            num_simulations: Number of MCTS simulations per search
            c_puct: Exploration constant for UCB1 formula
            device: Device to run neural network on ('cuda' or 'cpu')
            timeout: Optional timeout in seconds for search (None for no timeout)
        """
        self.neural_net = neural_net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        self.timeout = timeout
        
        # Set neural network to evaluation mode
        self.neural_net.eval()
    
    def search(self, root_position: Position, fallback_on_timeout: bool = True) -> np.ndarray:
        """Run MCTS from root position and return policy.
        
        Args:
            root_position: Starting position for search
            fallback_on_timeout: If True, return neural network policy on timeout
            
        Returns:
            Policy vector of shape (4096,) with move probabilities
            
        Raises:
            SearchTimeoutError: If timeout occurs and fallback_on_timeout is False
        """
        start_time = time.time()
        
        # Create root node
        root = MCTSNode(root_position)
        
        # Expand root node if not terminal
        if not root_position.is_terminal():
            self._expand_node(root)
        
        # Run simulations
        simulations_completed = 0
        try:
            for sim in range(self.num_simulations):
                # Check timeout
                if self.timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed > self.timeout:
                        logger.warning(
                            f"MCTS search timed out after {elapsed:.2f}s "
                            f"({simulations_completed}/{self.num_simulations} simulations)"
                        )
                        if fallback_on_timeout:
                            # Fallback to neural network policy if we have some simulations
                            if simulations_completed > 0:
                                logger.info("Using partial MCTS results as fallback")
                                return self._get_policy_from_visits(root)
                            else:
                                logger.info("Using neural network policy as fallback")
                                return self._get_neural_network_policy(root_position)
                        else:
                            raise SearchTimeoutError(self.timeout, simulations_completed)
                
                # Selection: traverse tree to leaf
                node = root
                search_path = [node]
                
                while not node.is_leaf() and not node.position.is_terminal():
                    node = self._select_child(node)
                    search_path.append(node)
                
                # Evaluation: get value for leaf node
                value = self._evaluate_node(node)
                
                # Backpropagation: update statistics along path
                self._backpropagate(search_path, value)
                
                simulations_completed += 1
        
        except Exception as e:
            logger.error(f"Error during MCTS search: {e}")
            if fallback_on_timeout and simulations_completed > 0:
                logger.info(f"Using partial results from {simulations_completed} simulations")
                return self._get_policy_from_visits(root)
            raise
        
        # Return policy based on visit counts
        return self._get_policy_from_visits(root)
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB score.
        
        Args:
            node: Parent node to select child from
            
        Returns:
            Child node with highest UCB score
        """
        best_score = -float('inf')
        best_child = None
        
        for child in node.children.values():
            score = child.get_ucb_score(self.c_puct, node.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _expand_node(self, node: MCTSNode) -> None:
        """Expand node by adding children for all legal moves.
        
        Args:
            node: Node to expand
        """
        if node.position.is_terminal():
            return
        
        # Get neural network predictions for position
        policy_probs, _ = self._evaluate_position(node.position)
        
        # Create child nodes for all legal moves
        legal_moves = node.position.get_legal_moves()
        for move in legal_moves:
            move_idx = MoveEncoder.encode_move(move)
            prior_prob = policy_probs[move_idx]
            
            # Create child position
            child_position = node.position.make_move(move)
            
            # Add child node
            node.add_child(move, child_position, prior_prob)
    
    def _evaluate_node(self, node: MCTSNode) -> float:
        """Evaluate node using neural network or game result.
        
        Args:
            node: Node to evaluate
            
        Returns:
            Value of the node from current player's perspective
        """
        # Terminal nodes return game result
        if node.position.is_terminal():
            return node.position.get_result()
        
        # Expand leaf nodes
        if node.is_leaf():
            self._expand_node(node)
        
        # Get neural network evaluation
        _, value = self._evaluate_position(node.position)
        return value
    
    def _evaluate_position(self, position: Position) -> Tuple[np.ndarray, float]:
        """Get neural network evaluation of position.
        
        Args:
            position: Position to evaluate
            
        Returns:
            Tuple of (policy_probabilities, value):
                - policy_probabilities: (4096,) array of move probabilities
                - value: Position value from current player's perspective
        """
        # Convert position to tensor
        board_tensor = torch.from_numpy(position.to_tensor()).float()
        board_tensor = board_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        board_tensor = board_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Get neural network predictions
        with torch.no_grad():
            policy_logits, value = self.neural_net(board_tensor)
        
        # Convert to numpy
        policy_logits = policy_logits.cpu().numpy()[0]  # Remove batch dimension
        value = value.cpu().numpy()[0, 0]  # Remove batch and feature dimensions
        
        # Apply legal move mask to policy
        move_mask = MoveEncoder.get_move_mask(position)
        
        # Mask illegal moves with very negative logits
        masked_logits = policy_logits * move_mask + (1 - move_mask) * (-1e8)
        
        # Convert to probabilities
        policy_probs = self._softmax(masked_logits)
        
        return policy_probs, value
    
    def _backpropagate(self, search_path: list, value: float) -> None:
        """Backpropagate value through search path.
        
        Args:
            search_path: List of nodes from root to leaf
            value: Value to backpropagate
        """
        for node in reversed(search_path):
            node.update(value)
            # Flip value for opponent (zero-sum game)
            value = -value
    
    def _get_policy_from_visits(self, root: MCTSNode) -> np.ndarray:
        """Convert visit counts to policy distribution.
        
        Args:
            root: Root node with visit statistics
            
        Returns:
            Policy vector of shape (4096,) with move probabilities
        """
        policy = np.zeros(4096, dtype=np.float32)
        
        # Get visit counts for each move
        for move, child in root.children.items():
            move_idx = MoveEncoder.encode_move(move)
            policy[move_idx] = child.visit_count
        
        # Normalize to probabilities
        total_visits = np.sum(policy)
        if total_visits > 0:
            policy = policy / total_visits
        
        return policy
    
    def _get_neural_network_policy(self, position: Position) -> np.ndarray:
        """Get policy directly from neural network without MCTS.
        
        Used as fallback when MCTS times out or fails.
        
        Args:
            position: Position to evaluate
            
        Returns:
            Policy vector of shape (4096,) with move probabilities
        """
        try:
            policy_probs, _ = self._evaluate_position(position)
            return policy_probs
        except Exception as e:
            logger.error(f"Failed to get neural network policy: {e}")
            # Ultimate fallback: uniform distribution over legal moves
            policy = np.zeros(4096)
            legal_moves = position.get_legal_moves()
            if legal_moves:
                for move in legal_moves:
                    move_idx = MoveEncoder.encode_move(move)
                    policy[move_idx] = 1.0 / len(legal_moves)
            return policy
    
    def get_best_move(self, position: Position) -> Optional[chess.Move]:
        """Get the best move for a position.
        
        Args:
            position: Position to find best move for
            
        Returns:
            Best move according to MCTS, or None if no legal moves
        """
        if position.is_terminal():
            return None
        
        policy = self.search(position)
        legal_moves = position.get_legal_moves()
        
        if not legal_moves:
            return None
        
        # Find move with highest probability
        best_move = None
        best_prob = -1.0
        
        for move in legal_moves:
            move_idx = MoveEncoder.encode_move(move)
            prob = policy[move_idx]
            if prob > best_prob:
                best_prob = prob
                best_move = move
        
        return best_move
    
    def get_move_probabilities(self, position: Position) -> dict:
        """Get move probabilities for all legal moves.
        
        Args:
            position: Position to analyze
            
        Returns:
            Dictionary mapping moves to their probabilities
        """
        policy = self.search(position)
        legal_moves = position.get_legal_moves()
        
        move_probs = {}
        for move in legal_moves:
            move_idx = MoveEncoder.encode_move(move)
            move_probs[move] = policy[move_idx]
        
        return move_probs
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax of array.
        
        Args:
            x: Input array
            
        Returns:
            Softmax probabilities
        """
        # Subtract max for numerical stability
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x)
    
    def set_num_simulations(self, num_simulations: int) -> None:
        """Set number of simulations for search.
        
        Args:
            num_simulations: New number of simulations
        """
        self.num_simulations = num_simulations
    
    def set_c_puct(self, c_puct: float) -> None:
        """Set exploration constant.
        
        Args:
            c_puct: New exploration constant
        """
        self.c_puct = c_puct