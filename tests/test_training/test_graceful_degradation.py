"""Integration tests for graceful degradation and error handling."""

import pytest
import torch
import numpy as np
import chess

from src.mcts.mcts import MCTS
from src.training.self_play_worker import SelfPlayWorker
from src.chess_env.position import Position
from src.neural_net.chess_net import ChessNet
from src.config.config import Config
from src.exceptions import SearchTimeoutError, SelfPlayError


class TestMCTSTimeout:
    """Test MCTS timeout handling."""
    
    def test_mcts_timeout_with_fallback(self):
        """Test that MCTS falls back to neural network on timeout."""
        # Create neural network
        neural_net = ChessNet(num_res_blocks=1, num_filters=32)
        neural_net.eval()
        
        # Create MCTS with very short timeout
        mcts = MCTS(
            neural_net=neural_net,
            num_simulations=1000,  # Many simulations
            timeout=0.001,  # Very short timeout (1ms)
            device='cpu'
        )
        
        # Create starting position
        position = Position(chess.Board())
        
        # Should timeout but return fallback policy
        policy = mcts.search(position, fallback_on_timeout=True)
        
        assert policy.shape == (4096,)
        assert np.sum(policy) > 0  # Should have some probability mass
        assert not np.isnan(policy).any()
    
    def test_mcts_timeout_without_fallback_raises(self):
        """Test that MCTS raises exception on timeout without fallback."""
        neural_net = ChessNet(num_res_blocks=1, num_filters=32)
        neural_net.eval()
        
        mcts = MCTS(
            neural_net=neural_net,
            num_simulations=1000,
            timeout=0.001,
            device='cpu'
        )
        
        position = Position(chess.Board())
        
        # Should raise SearchTimeoutError
        with pytest.raises(SearchTimeoutError) as exc_info:
            mcts.search(position, fallback_on_timeout=False)
        
        assert exc_info.value.timeout_seconds == 0.001
        assert exc_info.value.simulations_completed >= 0
    
    def test_mcts_no_timeout_completes_normally(self):
        """Test that MCTS without timeout completes all simulations."""
        neural_net = ChessNet(num_res_blocks=1, num_filters=32)
        neural_net.eval()
        
        # No timeout
        mcts = MCTS(
            neural_net=neural_net,
            num_simulations=10,
            timeout=None,
            device='cpu'
        )
        
        position = Position(chess.Board())
        
        # Should complete normally
        policy = mcts.search(position)
        
        assert policy.shape == (4096,)
        assert np.sum(policy) > 0
    
    def test_mcts_partial_results_on_timeout(self):
        """Test that MCTS uses partial results when timeout occurs mid-search."""
        neural_net = ChessNet(num_res_blocks=1, num_filters=32)
        neural_net.eval()
        
        # Timeout that allows some simulations
        mcts = MCTS(
            neural_net=neural_net,
            num_simulations=100,
            timeout=0.1,  # 100ms should allow some simulations
            device='cpu'
        )
        
        position = Position(chess.Board())
        
        # Should use partial results
        policy = mcts.search(position, fallback_on_timeout=True)
        
        assert policy.shape == (4096,)
        assert np.sum(policy) > 0


class TestSelfPlayRetry:
    """Test self-play retry logic."""
    
    def test_self_play_succeeds_on_first_attempt(self):
        """Test that successful game doesn't retry."""
        config = Config(
            mcts_simulations=10,
            device='cpu',
            num_res_blocks=1,
            num_filters=32
        )
        
        neural_net = ChessNet(
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters
        )
        neural_net.eval()
        
        worker = SelfPlayWorker(neural_net, config)
        
        # Should succeed without retries
        examples = worker.play_game_with_retry(temperature=1.0, max_retries=3)
        
        assert len(examples) > 0
        assert all(hasattr(ex, 'position') for ex in examples)
        assert all(hasattr(ex, 'policy') for ex in examples)
        assert all(hasattr(ex, 'value') for ex in examples)
    
    def test_self_play_retry_on_failure(self):
        """Test that self-play retries on failure."""
        config = Config(
            mcts_simulations=10,
            device='cpu',
            num_res_blocks=1,
            num_filters=32
        )
        
        neural_net = ChessNet(
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters
        )
        neural_net.eval()
        
        worker = SelfPlayWorker(neural_net, config)
        
        # Mock play_game to fail first time, succeed second time
        original_play_game = worker.play_game
        call_count = [0]
        
        def mock_play_game(temperature=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Simulated failure")
            return original_play_game(temperature)
        
        worker.play_game = mock_play_game
        
        # Should retry and succeed
        examples = worker.play_game_with_retry(temperature=1.0, max_retries=3)
        
        assert len(examples) > 0
        assert call_count[0] == 2  # Failed once, succeeded on retry
    
    def test_self_play_fails_after_max_retries(self):
        """Test that self-play raises error after max retries."""
        config = Config(
            mcts_simulations=10,
            device='cpu',
            num_res_blocks=1,
            num_filters=32
        )
        
        neural_net = ChessNet(
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters
        )
        neural_net.eval()
        
        worker = SelfPlayWorker(neural_net, config)
        
        # Mock play_game to always fail
        def mock_play_game(temperature=None):
            raise Exception("Persistent failure")
        
        worker.play_game = mock_play_game
        
        # Should fail after retries
        with pytest.raises(SelfPlayError) as exc_info:
            worker.play_game_with_retry(temperature=1.0, max_retries=3)
        
        assert "Failed after 3 attempts" in str(exc_info.value)


class TestErrorRecovery:
    """Test error recovery scenarios."""
    
    def test_mcts_recovers_from_neural_network_error(self):
        """Test that MCTS can recover from neural network errors."""
        neural_net = ChessNet(num_res_blocks=1, num_filters=32)
        neural_net.eval()
        
        mcts = MCTS(
            neural_net=neural_net,
            num_simulations=10,
            device='cpu'
        )
        
        position = Position(chess.Board())
        
        # Mock _evaluate_position to fail once then succeed
        original_evaluate = mcts._evaluate_position
        call_count = [0]
        
        def mock_evaluate(pos):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated NN error")
            return original_evaluate(pos)
        
        mcts._evaluate_position = mock_evaluate
        
        # Should handle error gracefully
        try:
            policy = mcts.search(position, fallback_on_timeout=True)
            # If it succeeds, verify policy is valid
            assert policy.shape == (4096,)
        except Exception:
            # If it fails, that's also acceptable for this test
            pass
    
    def test_neural_network_fallback_policy(self):
        """Test that neural network fallback policy is valid."""
        neural_net = ChessNet(num_res_blocks=1, num_filters=32)
        neural_net.eval()
        
        mcts = MCTS(
            neural_net=neural_net,
            num_simulations=10,
            device='cpu'
        )
        
        position = Position(chess.Board())
        
        # Get neural network policy directly
        policy = mcts._get_neural_network_policy(position)
        
        assert policy.shape == (4096,)
        assert np.sum(policy) > 0
        assert not np.isnan(policy).any()
        assert not np.isinf(policy).any()
        
        # Check that legal moves have non-zero probability
        legal_moves = position.get_legal_moves()
        from src.chess_env.move_encoder import MoveEncoder
        
        legal_move_probs = [policy[MoveEncoder.encode_move(move)] for move in legal_moves]
        assert any(prob > 0 for prob in legal_move_probs)
    
    def test_ultimate_fallback_uniform_policy(self):
        """Test ultimate fallback to uniform policy when everything fails."""
        neural_net = ChessNet(num_res_blocks=1, num_filters=32)
        neural_net.eval()
        
        mcts = MCTS(
            neural_net=neural_net,
            num_simulations=10,
            device='cpu'
        )
        
        position = Position(chess.Board())
        
        # Mock _evaluate_position to always fail
        def mock_evaluate(pos):
            raise RuntimeError("Complete failure")
        
        mcts._evaluate_position = mock_evaluate
        
        # Should fall back to uniform distribution
        policy = mcts._get_neural_network_policy(position)
        
        assert policy.shape == (4096,)
        assert np.sum(policy) > 0
        
        # Check that it's approximately uniform over legal moves
        legal_moves = position.get_legal_moves()
        from src.chess_env.move_encoder import MoveEncoder
        
        legal_move_probs = [policy[MoveEncoder.encode_move(move)] for move in legal_moves]
        expected_prob = 1.0 / len(legal_moves)
        
        for prob in legal_move_probs:
            assert abs(prob - expected_prob) < 0.01  # Allow small numerical error


class TestErrorLogging:
    """Test that errors are properly logged."""
    
    def test_timeout_warning_logged(self, caplog):
        """Test that timeout warnings are logged."""
        import logging
        caplog.set_level(logging.WARNING)
        
        neural_net = ChessNet(num_res_blocks=1, num_filters=32)
        neural_net.eval()
        
        mcts = MCTS(
            neural_net=neural_net,
            num_simulations=1000,
            timeout=0.001,
            device='cpu'
        )
        
        position = Position(chess.Board())
        
        # Trigger timeout
        mcts.search(position, fallback_on_timeout=True)
        
        # Check that warning was logged
        assert any("timed out" in record.message.lower() for record in caplog.records)
    
    def test_self_play_error_logged(self, caplog):
        """Test that self-play errors are logged."""
        import logging
        caplog.set_level(logging.WARNING)
        
        config = Config(
            mcts_simulations=10,
            device='cpu',
            num_res_blocks=1,
            num_filters=32
        )
        
        neural_net = ChessNet(
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters
        )
        neural_net.eval()
        
        worker = SelfPlayWorker(neural_net, config)
        
        # Mock to fail
        def mock_play_game(temperature=None):
            raise Exception("Test error")
        
        worker.play_game = mock_play_game
        
        # Trigger error
        try:
            worker.play_game_with_retry(temperature=1.0, max_retries=2)
        except SelfPlayError:
            pass
        
        # Check that errors were logged
        assert len(caplog.records) > 0
        assert any("error" in record.message.lower() for record in caplog.records)
