"""Unit tests for MCTSNode class."""

import pytest
import chess
import math
from src.chess_env.position import Position
from src.mcts.mcts_node import MCTSNode


class TestMCTSNode:
    """Test cases for MCTSNode class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple king-pawn endgame position
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.E2, chess.Piece(chess.PAWN, chess.WHITE))
        board.turn = chess.WHITE
        
        self.position = Position(board)
        self.root_node = MCTSNode(self.position)
    
    def test_node_initialization(self):
        """Test MCTSNode initialization."""
        # Test root node
        assert self.root_node.position == self.position
        assert self.root_node.parent is None
        assert self.root_node.prior_prob == 0.0
        assert self.root_node.move is None
        assert len(self.root_node.children) == 0
        assert self.root_node.visit_count == 0
        assert self.root_node.total_value == 0.0
        
        # Test child node
        move = chess.Move.from_uci("e2e3")
        child_position = self.position.make_move(move)
        child_node = MCTSNode(child_position, parent=self.root_node,
                             prior_prob=0.5, move=move)
        
        assert child_node.position == child_position
        assert child_node.parent == self.root_node
        assert child_node.prior_prob == 0.5
        assert child_node.move == move
        assert len(child_node.children) == 0
        assert child_node.visit_count == 0
        assert child_node.total_value == 0.0
    
    def test_get_value(self):
        """Test get_value method."""
        # Unvisited node should return 0.0
        assert self.root_node.get_value() == 0.0
        
        # Update node with some values
        self.root_node.update(1.0)
        assert self.root_node.get_value() == 1.0
        
        self.root_node.update(-0.5)
        assert self.root_node.get_value() == 0.25  # (1.0 + (-0.5)) / 2
        
        self.root_node.update(0.5)
        assert abs(self.root_node.get_value() - 1.0/3) < 1e-6  # (1.0 - 0.5 + 0.5) / 3
    
    def test_get_ucb_score(self):
        """Test get_ucb_score method."""
        # Create child node
        move = chess.Move.from_uci("e2e3")
        child_position = self.position.make_move(move)
        child_node = MCTSNode(child_position, parent=self.root_node,
                             prior_prob=0.3, move=move)
        
        # Test UCB score calculation
        c_puct = 1.0
        parent_visits = 10
        
        # Unvisited child: Q(s,a) = 0, UCB = 0 + 1.0 * 0.3 * sqrt(10) / (1 + 0)
        expected_ucb = 0.0 + 1.0 * 0.3 * math.sqrt(10) / (1 + 0)
        assert abs(child_node.get_ucb_score(c_puct, parent_visits) - expected_ucb) < 1e-6
        
        # Update child and test again
        child_node.update(0.5)
        # Q(s,a) = 0.5, UCB = 0.5 + 1.0 * 0.3 * sqrt(10) / (1 + 1)
        expected_ucb = 0.5 + 1.0 * 0.3 * math.sqrt(10) / (1 + 1)
        assert abs(child_node.get_ucb_score(c_puct, parent_visits) - expected_ucb) < 1e-6
    
    def test_is_leaf(self):
        """Test is_leaf method."""
        # New node should be a leaf
        assert self.root_node.is_leaf()
        
        # Add a child
        move = chess.Move.from_uci("e2e3")
        child_position = self.position.make_move(move)
        self.root_node.add_child(move, child_position, 0.3)
        
        # Node with children should not be a leaf
        assert not self.root_node.is_leaf()
    
    def test_is_root(self):
        """Test is_root method."""
        # Root node should return True
        assert self.root_node.is_root()
        
        # Child node should return False
        move = chess.Move.from_uci("e2e3")
        child_position = self.position.make_move(move)
        child_node = MCTSNode(child_position, parent=self.root_node,
                             prior_prob=0.3, move=move)
        assert not child_node.is_root()
    
    def test_add_child(self):
        """Test add_child method."""
        move = chess.Move.from_uci("e2e3")
        child_position = self.position.make_move(move)
        
        # Add child
        child_node = self.root_node.add_child(move, child_position, 0.3)
        
        # Verify child was added correctly
        assert len(self.root_node.children) == 1
        assert move in self.root_node.children
        assert self.root_node.children[move] == child_node
        assert child_node.parent == self.root_node
        assert child_node.move == move
        assert child_node.prior_prob == 0.3
        
        # Test adding duplicate child raises error
        with pytest.raises(ValueError, match="Child already exists"):
            self.root_node.add_child(move, child_position, 0.5)
    
    def test_get_child(self):
        """Test get_child method."""
        move = chess.Move.from_uci("e2e3")
        child_position = self.position.make_move(move)
        
        # Non-existent child should return None
        assert self.root_node.get_child(move) is None
        
        # Add child and test retrieval
        child_node = self.root_node.add_child(move, child_position, 0.3)
        assert self.root_node.get_child(move) == child_node
    
    def test_has_children(self):
        """Test has_children method."""
        # New node should have no children
        assert not self.root_node.has_children()
        
        # Add child
        move = chess.Move.from_uci("e2e3")
        child_position = self.position.make_move(move)
        self.root_node.add_child(move, child_position, 0.3)
        
        # Node should now have children
        assert self.root_node.has_children()
    
    def test_get_most_visited_child(self):
        """Test get_most_visited_child method."""
        # No children should return None
        assert self.root_node.get_most_visited_child() is None
        
        # Add multiple children
        move1 = chess.Move.from_uci("e2e3")
        move2 = chess.Move.from_uci("e2e4")
        child1_pos = self.position.make_move(move1)
        child2_pos = self.position.make_move(move2)
        
        child1 = self.root_node.add_child(move1, child1_pos, 0.3)
        child2 = self.root_node.add_child(move2, child2_pos, 0.7)
        
        # Initially both have 0 visits, should return one of them
        most_visited = self.root_node.get_most_visited_child()
        assert most_visited in [child1, child2]
        
        # Update visit counts
        child1.update(0.5)
        child1.update(0.3)  # 2 visits
        child2.update(0.8)  # 1 visit
        
        # child1 should be most visited
        assert self.root_node.get_most_visited_child() == child1
    
    def test_get_best_child(self):
        """Test get_best_child method."""
        # No children should return None
        assert self.root_node.get_best_child(1.0) is None
        
        # Add multiple children with different priors
        move1 = chess.Move.from_uci("e2e3")
        move2 = chess.Move.from_uci("e2e4")
        child1_pos = self.position.make_move(move1)
        child2_pos = self.position.make_move(move2)
        
        child1 = self.root_node.add_child(move1, child1_pos, 0.3)
        child2 = self.root_node.add_child(move2, child2_pos, 0.7)
        
        # Update parent visit count for UCB calculation
        self.root_node.update(0.0)
        
        # child2 has higher prior, so should have higher UCB initially
        best_child = self.root_node.get_best_child(1.0)
        assert best_child == child2
        
        # Update child1 with high value
        child1.update(1.0)
        
        # Now child1 might be best due to high Q value
        # (depends on exact UCB calculation)
        best_child = self.root_node.get_best_child(1.0)
        assert best_child in [child1, child2]
    
    def test_update(self):
        """Test update method."""
        # Initial state
        assert self.root_node.visit_count == 0
        assert self.root_node.total_value == 0.0
        assert self.root_node.get_value() == 0.0
        
        # First update
        self.root_node.update(0.5)
        assert self.root_node.visit_count == 1
        assert self.root_node.total_value == 0.5
        assert self.root_node.get_value() == 0.5
        
        # Second update
        self.root_node.update(-0.3)
        assert self.root_node.visit_count == 2
        assert self.root_node.total_value == 0.2
        assert abs(self.root_node.get_value() - 0.1) < 1e-6
    
    def test_string_representations(self):
        """Test __str__ and __repr__ methods."""
        # Root node
        str_repr = str(self.root_node)
        assert "root" in str_repr
        assert "visits=0" in str_repr
        assert "value=0.000" in str_repr
        assert "prior=0.000" in str_repr
        
        # Child node
        move = chess.Move.from_uci("e2e3")
        child_position = self.position.make_move(move)
        child_node = MCTSNode(child_position, parent=self.root_node,
                             prior_prob=0.3, move=move)
        child_node.update(0.5)
        
        str_repr = str(child_node)
        assert "e2e3" in str_repr
        assert "visits=1" in str_repr
        assert "value=0.500" in str_repr
        assert "prior=0.300" in str_repr
        
        # __repr__ should be same as __str__
        assert repr(child_node) == str(child_node)