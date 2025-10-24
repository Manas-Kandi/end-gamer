"""MCTS Node implementation for chess tree search."""

from typing import Dict, Optional
import math
import chess
from ..chess_env.position import Position


class MCTSNode:
    """Node in the MCTS tree.
    
    Each node represents a chess position and stores statistics for
    the UCB1 selection algorithm and value backpropagation.
    """
    
    def __init__(self, position: Position, parent: Optional['MCTSNode'] = None,
                 prior_prob: float = 0.0, move: Optional[chess.Move] = None):
        """Initialize MCTS node.
        
        Args:
            position: Chess position this node represents
            parent: Parent node in the tree (None for root)
            prior_prob: Prior probability P(s,a) from neural network
            move: Move that led to this position from parent
        """
        self.position = position
        self.parent = parent
        self.prior_prob = prior_prob
        self.move = move
        
        # Child nodes indexed by move
        self.children: Dict[chess.Move, 'MCTSNode'] = {}
        
        # MCTS statistics
        self.visit_count = 0
        self.total_value = 0.0
    
    def get_value(self) -> float:
        """Get mean action value Q(s,a).
        
        Returns:
            Mean value of all visits to this node, or 0.0 if unvisited
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    def get_ucb_score(self, c_puct: float, parent_visits: int) -> float:
        """Calculate UCB score for node selection.
        
        Uses the UCB1 formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        where:
        - Q(s,a) is the mean action value
        - P(s,a) is the prior probability from neural network
        - N(s) is the parent visit count
        - N(s,a) is this node's visit count
        - c_puct is the exploration constant
        
        Args:
            c_puct: Exploration constant balancing exploitation vs exploration
            parent_visits: Number of visits to parent node
            
        Returns:
            UCB score for this node
        """
        q_value = self.get_value()
        
        # Exploration term: c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        u_value = (c_puct * self.prior_prob * 
                  math.sqrt(parent_visits) / (1 + self.visit_count))
        
        return q_value + u_value
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf (not expanded).
        
        Returns:
            True if node has no children, False otherwise
        """
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """Check if node is the root of the tree.
        
        Returns:
            True if node has no parent, False otherwise
        """
        return self.parent is None
    
    def add_child(self, move: chess.Move, child_position: Position, 
                  prior_prob: float) -> 'MCTSNode':
        """Add a child node for the given move.
        
        Args:
            move: Chess move leading to child position
            child_position: Position after making the move
            prior_prob: Prior probability for this move from neural network
            
        Returns:
            The newly created child node
            
        Raises:
            ValueError: If child already exists for this move
        """
        if move in self.children:
            raise ValueError(f"Child already exists for move {move}")
        
        child = MCTSNode(child_position, parent=self, 
                        prior_prob=prior_prob, move=move)
        self.children[move] = child
        return child
    
    def get_child(self, move: chess.Move) -> Optional['MCTSNode']:
        """Get child node for the given move.
        
        Args:
            move: Chess move to look up
            
        Returns:
            Child node if it exists, None otherwise
        """
        return self.children.get(move)
    
    def has_children(self) -> bool:
        """Check if node has any children.
        
        Returns:
            True if node has children, False otherwise
        """
        return len(self.children) > 0
    
    def get_most_visited_child(self) -> Optional['MCTSNode']:
        """Get the child with the highest visit count.
        
        Returns:
            Child node with most visits, or None if no children
        """
        if not self.children:
            return None
        
        return max(self.children.values(), key=lambda child: child.visit_count)
    
    def get_best_child(self, c_puct: float) -> Optional['MCTSNode']:
        """Get the child with the highest UCB score.
        
        Args:
            c_puct: Exploration constant for UCB calculation
            
        Returns:
            Child node with highest UCB score, or None if no children
        """
        if not self.children:
            return None
        
        return max(self.children.values(), 
                  key=lambda child: child.get_ucb_score(c_puct, self.visit_count))
    
    def update(self, value: float) -> None:
        """Update node statistics with a new value.
        
        Args:
            value: Value to add to node statistics
        """
        self.visit_count += 1
        self.total_value += value
    
    def __str__(self) -> str:
        """String representation of the node."""
        move_str = str(self.move) if self.move else "root"
        return (f"MCTSNode(move={move_str}, visits={self.visit_count}, "
                f"value={self.get_value():.3f}, prior={self.prior_prob:.3f})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()