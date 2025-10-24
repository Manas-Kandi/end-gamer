"""Unit tests for MCTS class."""

import pytest
import numpy as np
import torch
import torch.nn as nn
import chess
from unittest.mock import Mock, patch

from src.chess_env.position import Position
from src.mcts.mcts import MCTS
from src.mcts.mcts_node import MCTSNode


class MockNeuralNet(nn.Module):
    """Mock neural network for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # Dummy layer
    
    def forward(self, x):
        batch_size = x.shape[0]
        # Return uniform policy and zero value
        policy = torch.zeros(batch_size, 4096)
        value = torch.zeros(batch_size, 1)
        return policy, value


class TestMCTS:
    """Test cases for MCTS class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock neural network
        self.mock_net = MockNeuralNet()
        
        # Create MCTS instance
        self.mcts = MCTS(
            neural_net=self.mock_net,
            num_simulations=10,  # Small number for fast tests
            c_puct=1.0,
            device='cpu'
        )
        
        # Create test position
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.E2, chess.Piece(chess.PAWN, chess.WHITE))
        board.turn = chess.WHITE
        
        self.position = Position(board)
    
    def test_mcts_initialization(self):
        """Test MCTS initialization."""
        assert self.mcts.neural_net == self.mock_net
        assert self.mcts.num_simulations == 10
        assert self.mcts.c_puct == 1.0
        assert self.mcts.device == 'cpu'
        
        # Neural network should be in eval mode
        assert not self.mcts.neural_net.training
    
    def test_search_basic(self):
        """Test basic MCTS search functionality."""
        policy = self.mcts.search(self.position)
        
        # Policy should be valid probability distribution
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (4096,)
        assert np.all(policy >= 0)
        assert np.abs(np.sum(policy) - 1.0) < 1e-6  # Should sum to 1
        
        # Should have non-zero probabilities for legal moves only
        legal_moves = self.position.get_legal_moves()
        assert len(legal_moves) > 0
        
        # At least one legal move should have positive probability
        has_positive_prob = False
        for move in legal_moves:
            from src.chess_env.move_encoder import MoveEncoder
            move_idx = MoveEncoder.encode_move(move)
            if policy[move_idx] > 0:
                has_positive_prob = True
                break
        assert has_positive_prob
    
    def test_search_terminal_position(self):
        """Test MCTS search on terminal position."""
        # Create checkmate position
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.A7, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(chess.B6, chess.Piece(chess.KING, chess.WHITE))
        board.turn = chess.BLACK
        
        terminal_position = Position(board)
        assert terminal_position.is_terminal()
        
        # Search should still work and return valid policy
        policy = self.mcts.search(terminal_position)
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (4096,)
        # All probabilities should be zero for terminal position
        assert np.all(policy == 0)
    
    def test_select_child(self):
        """Test child selection using UCB scores."""
        # Create root node with children
        root = MCTSNode(self.position)
        
        # Add some children manually
        legal_moves = self.position.get_legal_moves()
        children = []
        for i, move in enumerate(legal_moves[:3]):  # Take first 3 moves
            child_pos = self.position.make_move(move)
            prior = 0.1 + i * 0.1  # Different priors
            child = root.add_child(move, child_pos, prior)
            children.append(child)
        
        # Update root visit count for UCB calculation
        root.update(0.0)
        
        # Select child should return one of the children
        selected = self.mcts._select_child(root)
        assert selected in children
        
        # Update one child to have high value
        children[0].update(1.0)
        children[0].update(1.0)  # High value, multiple visits
        
        # Should prefer the high-value child
        selected = self.mcts._select_child(root)
        # Note: Exact selection depends on UCB calculation, so we just verify it's valid
        assert selected in children
    
    def test_expand_node(self):
        """Test node expansion."""
        root = MCTSNode(self.position)
        
        # Initially should be leaf
        assert root.is_leaf()
        
        # Expand node
        self.mcts._expand_node(root)
        
        # Should no longer be leaf
        assert not root.is_leaf()
        
        # Should have children for all legal moves
        legal_moves = self.position.get_legal_moves()
        assert len(root.children) == len(legal_moves)
        
        # All legal moves should have corresponding children
        for move in legal_moves:
            assert move in root.children
            child = root.children[move]
            assert child.move == move
            assert child.parent == root
            assert 0 <= child.prior_prob <= 1
    
    def test_expand_terminal_node(self):
        """Test expanding terminal node does nothing."""
        # Create terminal position
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.A7, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(chess.B6, chess.Piece(chess.KING, chess.WHITE))
        board.turn = chess.BLACK
        
        terminal_position = Position(board)
        terminal_node = MCTSNode(terminal_position)
        
        # Should be leaf initially
        assert terminal_node.is_leaf()
        
        # Expand should do nothing
        self.mcts._expand_node(terminal_node)
        
        # Should still be leaf
        assert terminal_node.is_leaf()
        assert len(terminal_node.children) == 0
    
    def test_evaluate_node_terminal(self):
        """Test evaluating terminal node."""
        # Create terminal position (checkmate)
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.A7, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(chess.B6, chess.Piece(chess.KING, chess.WHITE))
        board.turn = chess.BLACK
        
        terminal_position = Position(board)
        terminal_node = MCTSNode(terminal_position)
        
        # Evaluate should return game result
        value = self.mcts._evaluate_node(terminal_node)
        assert value == -1.0  # Black is checkmated
    
    def test_evaluate_node_non_terminal(self):
        """Test evaluating non-terminal node."""
        root = MCTSNode(self.position)
        
        # Should expand and return neural network value
        value = self.mcts._evaluate_node(root)
        
        # Mock network returns 0, so value should be 0
        assert value == 0.0
        
        # Node should be expanded
        assert not root.is_leaf()
    
    def test_evaluate_position(self):
        """Test position evaluation with neural network."""
        policy_probs, value = self.mcts._evaluate_position(self.position)
        
        # Should return valid policy and value
        assert isinstance(policy_probs, np.ndarray)
        assert policy_probs.shape == (4096,)
        assert np.all(policy_probs >= 0)
        assert np.abs(np.sum(policy_probs) - 1.0) < 1e-6
        
        assert isinstance(value, (float, np.floating))
        # Mock network returns 0
        assert value == 0.0
        
        # Only legal moves should have positive probability
        legal_moves = self.position.get_legal_moves()
        from src.chess_env.move_encoder import MoveEncoder
        
        for i in range(4096):
            try:
                move = MoveEncoder.decode_move(i)
                if move in legal_moves:
                    # Legal moves can have any probability >= 0
                    assert policy_probs[i] >= 0
                else:
                    # Illegal moves should have 0 probability
                    assert policy_probs[i] == 0
            except ValueError:
                # Invalid move index should have 0 probability
                assert policy_probs[i] == 0
    
    def test_backpropagate(self):
        """Test value backpropagation."""
        # Create search path
        root = MCTSNode(self.position)
        
        legal_moves = self.position.get_legal_moves()
        child_pos = self.position.make_move(legal_moves[0])
        child = MCTSNode(child_pos, parent=root, move=legal_moves[0])
        
        search_path = [root, child]
        
        # Backpropagate value
        self.mcts._backpropagate(search_path, 0.5)
        
        # Both nodes should be updated, but values are flipped as we go up
        assert root.visit_count == 1
        assert child.visit_count == 1
        
        # The leaf gets the original value, parent gets flipped value
        assert child.total_value == 0.5
        assert child.get_value() == 0.5
        
        assert root.total_value == -0.5  # Flipped for parent
        assert root.get_value() == -0.5
    
    def test_get_policy_from_visits(self):
        """Test converting visit counts to policy."""
        root = MCTSNode(self.position)
        
        # Add children with different visit counts
        legal_moves = self.position.get_legal_moves()
        visit_counts = [10, 5, 2]
        
        for i, move in enumerate(legal_moves[:3]):
            child_pos = self.position.make_move(move)
            child = root.add_child(move, child_pos, 0.1)
            
            # Simulate visits
            for _ in range(visit_counts[i]):
                child.update(0.0)
        
        # Get policy
        policy = self.mcts._get_policy_from_visits(root)
        
        # Check policy is valid
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (4096,)
        assert np.all(policy >= 0)
        assert np.abs(np.sum(policy) - 1.0) < 1e-6
        
        # Check visit count proportions
        from src.chess_env.move_encoder import MoveEncoder
        total_visits = sum(visit_counts)
        
        for i, move in enumerate(legal_moves[:3]):
            move_idx = MoveEncoder.encode_move(move)
            expected_prob = visit_counts[i] / total_visits
            assert np.abs(policy[move_idx] - expected_prob) < 1e-6
    
    def test_get_best_move(self):
        """Test getting best move from position."""
        best_move = self.mcts.get_best_move(self.position)
        
        # Should return a legal move
        legal_moves = self.position.get_legal_moves()
        assert best_move in legal_moves
    
    def test_get_best_move_terminal(self):
        """Test getting best move from terminal position."""
        # Create terminal position
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.A7, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(chess.B6, chess.Piece(chess.KING, chess.WHITE))
        board.turn = chess.BLACK
        
        terminal_position = Position(board)
        best_move = self.mcts.get_best_move(terminal_position)
        
        # Should return None for terminal position
        assert best_move is None
    
    def test_get_move_probabilities(self):
        """Test getting move probabilities."""
        move_probs = self.mcts.get_move_probabilities(self.position)
        
        # Should return dict with legal moves
        legal_moves = self.position.get_legal_moves()
        assert isinstance(move_probs, dict)
        assert len(move_probs) == len(legal_moves)
        
        # All legal moves should be in dict
        for move in legal_moves:
            assert move in move_probs
            assert isinstance(move_probs[move], (float, np.floating))
            assert move_probs[move] >= 0
        
        # Probabilities should sum to 1
        total_prob = sum(move_probs.values())
        assert np.abs(total_prob - 1.0) < 1e-6
    
    def test_softmax(self):
        """Test softmax function."""
        x = np.array([1.0, 2.0, 3.0])
        probs = MCTS._softmax(x)
        
        # Should be valid probability distribution
        assert np.all(probs >= 0)
        assert np.abs(np.sum(probs) - 1.0) < 1e-6
        
        # Higher values should have higher probabilities
        assert probs[2] > probs[1] > probs[0]
        
        # Test numerical stability with large values
        x_large = np.array([1000.0, 1001.0, 1002.0])
        probs_large = MCTS._softmax(x_large)
        assert np.all(np.isfinite(probs_large))
        assert np.abs(np.sum(probs_large) - 1.0) < 1e-6
    
    def test_set_parameters(self):
        """Test setting MCTS parameters."""
        # Test setting number of simulations
        self.mcts.set_num_simulations(100)
        assert self.mcts.num_simulations == 100
        
        # Test setting exploration constant
        self.mcts.set_c_puct(2.0)
        assert self.mcts.c_puct == 2.0
    
    def test_integration_with_real_neural_net(self):
        """Test MCTS integration with actual neural network structure."""
        # Create a minimal neural network that matches expected interface
        class MinimalNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(12, 1, 1)
                self.policy_fc = nn.Linear(64, 4096)
                self.value_fc = nn.Linear(64, 1)
            
            def forward(self, x):
                batch_size = x.shape[0]
                x = self.conv(x)
                x = x.view(batch_size, -1)
                policy = self.policy_fc(x)
                value = torch.tanh(self.value_fc(x))
                return policy, value
        
        real_net = MinimalNet()
        real_mcts = MCTS(real_net, num_simulations=5, device='cpu')
        
        # Should work without errors
        policy = real_mcts.search(self.position)
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (4096,)
        
        best_move = real_mcts.get_best_move(self.position)
        legal_moves = self.position.get_legal_moves()
        assert best_move in legal_moves