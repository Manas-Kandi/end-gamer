"""Unit tests for tablebase integration."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import chess
import chess.syzygy

from src.evaluation.tablebase import TablebaseInterface
from src.chess_env.position import Position


class TestTablebaseInterface(unittest.TestCase):
    """Test tablebase interface with mock tablebase."""
    
    def test_init_without_path(self):
        """Test initialization without tablebase path."""
        tb = TablebaseInterface()
        self.assertFalse(tb.is_available())
        self.assertIsNone(tb.tablebase)
    
    def test_init_with_invalid_path(self):
        """Test initialization with invalid path."""
        tb = TablebaseInterface("/nonexistent/path")
        self.assertFalse(tb.is_available())
        self.assertIsNone(tb.tablebase)
    
    @patch('os.path.exists')
    @patch('chess.syzygy.open_tablebase')
    def test_init_with_valid_path(self, mock_open_tb, mock_exists):
        """Test initialization with valid tablebase path."""
        mock_exists.return_value = True
        mock_tb = MagicMock()
        mock_tb.max_fds = 5
        mock_open_tb.return_value = mock_tb
        
        tb = TablebaseInterface("/valid/path")
        
        self.assertTrue(tb.is_available())
        self.assertIsNotNone(tb.tablebase)
        mock_open_tb.assert_called_once_with("/valid/path")
    
    @patch('os.path.exists')
    @patch('chess.syzygy.open_tablebase')
    def test_init_with_environment_variable(self, mock_open_tb, mock_exists):
        """Test initialization using environment variable."""
        mock_exists.return_value = True
        mock_tb = MagicMock()
        mock_open_tb.return_value = mock_tb
        
        with patch.dict('os.environ', {'TABLEBASE_PATH': '/env/path'}):
            tb = TablebaseInterface()
            
            self.assertTrue(tb.is_available())
            mock_open_tb.assert_called_once_with('/env/path')
    
    def test_probe_when_unavailable(self):
        """Test probe returns None when tablebase unavailable."""
        tb = TablebaseInterface()
        
        board = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
        position = Position(board)
        
        result = tb.probe(position)
        self.assertIsNone(result)
    
    @patch('os.path.exists')
    @patch('chess.syzygy.open_tablebase')
    def test_probe_winning_position(self, mock_open_tb, mock_exists):
        """Test probe returns 1.0 for winning position."""
        mock_exists.return_value = True
        mock_tb = MagicMock()
        mock_tb.max_fds = 5
        mock_tb.probe_wdl.return_value = 2  # Win
        mock_open_tb.return_value = mock_tb
        
        tb = TablebaseInterface("/valid/path")
        
        board = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
        position = Position(board)
        
        result = tb.probe(position)
        self.assertEqual(result, 1.0)
    
    @patch('os.path.exists')
    @patch('chess.syzygy.open_tablebase')
    def test_probe_drawn_position(self, mock_open_tb, mock_exists):
        """Test probe returns 0.0 for drawn position."""
        mock_exists.return_value = True
        mock_tb = MagicMock()
        mock_tb.max_fds = 5
        mock_tb.probe_wdl.return_value = 0  # Draw
        mock_open_tb.return_value = mock_tb
        
        tb = TablebaseInterface("/valid/path")
        
        board = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
        position = Position(board)
        
        result = tb.probe(position)
        self.assertEqual(result, 0.0)
    
    @patch('os.path.exists')
    @patch('chess.syzygy.open_tablebase')
    def test_probe_losing_position(self, mock_open_tb, mock_exists):
        """Test probe returns -1.0 for losing position."""
        mock_exists.return_value = True
        mock_tb = MagicMock()
        mock_tb.max_fds = 5
        mock_tb.probe_wdl.return_value = -2  # Loss
        mock_open_tb.return_value = mock_tb
        
        tb = TablebaseInterface("/valid/path")
        
        board = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
        position = Position(board)
        
        result = tb.probe(position)
        self.assertEqual(result, -1.0)
    
    @patch('os.path.exists')
    @patch('chess.syzygy.open_tablebase')
    def test_probe_cursed_win(self, mock_open_tb, mock_exists):
        """Test probe returns 1.0 for cursed win."""
        mock_exists.return_value = True
        mock_tb = MagicMock()
        mock_tb.max_fds = 5
        mock_tb.probe_wdl.return_value = 1  # Cursed win
        mock_open_tb.return_value = mock_tb
        
        tb = TablebaseInterface("/valid/path")
        
        board = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
        position = Position(board)
        
        result = tb.probe(position)
        self.assertEqual(result, 1.0)
    
    @patch('os.path.exists')
    @patch('chess.syzygy.open_tablebase')
    def test_probe_blessed_loss(self, mock_open_tb, mock_exists):
        """Test probe returns -1.0 for blessed loss."""
        mock_exists.return_value = True
        mock_tb = MagicMock()
        mock_tb.max_fds = 5
        mock_tb.probe_wdl.return_value = -1  # Blessed loss
        mock_open_tb.return_value = mock_tb
        
        tb = TablebaseInterface("/valid/path")
        
        board = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
        position = Position(board)
        
        result = tb.probe(position)
        self.assertEqual(result, -1.0)
    
    @patch('os.path.exists')
    @patch('chess.syzygy.open_tablebase')
    def test_probe_position_not_in_tablebase(self, mock_open_tb, mock_exists):
        """Test probe returns None for position not in tablebase."""
        mock_exists.return_value = True
        mock_tb = MagicMock()
        mock_tb.max_fds = 5
        mock_tb.probe_wdl.side_effect = KeyError("Position not found")
        mock_open_tb.return_value = mock_tb
        
        tb = TablebaseInterface("/valid/path")
        
        board = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
        position = Position(board)
        
        result = tb.probe(position)
        self.assertIsNone(result)
    
    @patch('os.path.exists')
    @patch('chess.syzygy.open_tablebase')
    def test_probe_too_many_pieces(self, mock_open_tb, mock_exists):
        """Test probe returns None when position has too many pieces."""
        mock_exists.return_value = True
        mock_tb = MagicMock()
        mock_tb.max_fds = 3  # Only 3-piece tablebase
        mock_open_tb.return_value = mock_tb
        
        tb = TablebaseInterface("/valid/path")
        
        # Position with 4 pieces (too many)
        board = chess.Board("8/8/8/4k3/8/8/4P3/3QK3 w - - 0 1")
        position = Position(board)
        
        result = tb.probe(position)
        self.assertIsNone(result)
    
    def test_get_best_move_when_unavailable(self):
        """Test get_best_move returns None when tablebase unavailable."""
        tb = TablebaseInterface()
        
        board = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
        position = Position(board)
        
        result = tb.get_best_move(position)
        self.assertIsNone(result)
    
    @patch('os.path.exists')
    @patch('chess.syzygy.open_tablebase')
    def test_get_best_move_returns_optimal_move(self, mock_open_tb, mock_exists):
        """Test get_best_move returns optimal move from tablebase."""
        mock_exists.return_value = True
        mock_tb = MagicMock()
        mock_tb.max_fds = 5
        
        # Mock WDL probes to prefer e2-e3 move
        def mock_probe_wdl(board):
            # Return better WDL for e2-e3 move
            last_move = board.peek() if board.move_stack else None
            if last_move and last_move.uci() == "e2e3":
                return -2  # Opponent loses (we win)
            return -1  # Opponent draws or worse
        
        mock_tb.probe_wdl.side_effect = mock_probe_wdl
        mock_tb.probe_dtz.return_value = 10
        mock_open_tb.return_value = mock_tb
        
        tb = TablebaseInterface("/valid/path")
        
        board = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
        position = Position(board)
        
        best_move = tb.get_best_move(position)
        self.assertIsNotNone(best_move)
        self.assertIsInstance(best_move, chess.Move)
    
    @patch('os.path.exists')
    @patch('chess.syzygy.open_tablebase')
    def test_get_best_move_no_legal_moves(self, mock_open_tb, mock_exists):
        """Test get_best_move returns None when no legal moves."""
        mock_exists.return_value = True
        mock_tb = MagicMock()
        mock_tb.max_fds = 5
        mock_open_tb.return_value = mock_tb
        
        tb = TablebaseInterface("/valid/path")
        
        # Stalemate position
        board = chess.Board("7k/8/6K1/8/8/8/8/6Q1 b - - 0 1")
        position = Position(board)
        
        best_move = tb.get_best_move(position)
        self.assertIsNone(best_move)
    
    @patch('os.path.exists')
    @patch('chess.syzygy.open_tablebase')
    def test_probe_dtz(self, mock_open_tb, mock_exists):
        """Test probe_dtz returns distance to zeroing."""
        mock_exists.return_value = True
        mock_tb = MagicMock()
        mock_tb.max_fds = 5
        mock_tb.probe_dtz.return_value = 15
        mock_open_tb.return_value = mock_tb
        
        tb = TablebaseInterface("/valid/path")
        
        board = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
        position = Position(board)
        
        dtz = tb.probe_dtz(position)
        self.assertEqual(dtz, 15)
    
    @patch('os.path.exists')
    @patch('chess.syzygy.open_tablebase')
    def test_probe_dtz_unavailable(self, mock_open_tb, mock_exists):
        """Test probe_dtz returns None when unavailable."""
        mock_exists.return_value = True
        mock_tb = MagicMock()
        mock_tb.max_fds = 5
        mock_tb.probe_dtz.side_effect = KeyError("DTZ not found")
        mock_open_tb.return_value = mock_tb
        
        tb = TablebaseInterface("/valid/path")
        
        board = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
        position = Position(board)
        
        dtz = tb.probe_dtz(position)
        self.assertIsNone(dtz)
    
    def test_probe_dtz_when_tablebase_unavailable(self):
        """Test probe_dtz returns None when tablebase unavailable."""
        tb = TablebaseInterface()
        
        board = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
        position = Position(board)
        
        dtz = tb.probe_dtz(position)
        self.assertIsNone(dtz)
    
    @patch('os.path.exists')
    @patch('chess.syzygy.open_tablebase')
    def test_get_max_pieces(self, mock_open_tb, mock_exists):
        """Test get_max_pieces returns correct value."""
        mock_exists.return_value = True
        mock_tb = MagicMock()
        mock_tb.max_fds = 5
        mock_open_tb.return_value = mock_tb
        
        tb = TablebaseInterface("/valid/path")
        
        max_pieces = tb.get_max_pieces()
        self.assertEqual(max_pieces, 5)
    
    def test_get_max_pieces_when_unavailable(self):
        """Test get_max_pieces returns 0 when tablebase unavailable."""
        tb = TablebaseInterface()
        
        max_pieces = tb.get_max_pieces()
        self.assertEqual(max_pieces, 0)
    
    @patch('os.path.exists')
    @patch('chess.syzygy.open_tablebase')
    def test_graceful_error_handling(self, mock_open_tb, mock_exists):
        """Test graceful handling of unexpected errors."""
        mock_exists.return_value = True
        mock_tb = MagicMock()
        mock_tb.max_fds = 5
        mock_tb.probe_wdl.side_effect = Exception("Unexpected error")
        mock_open_tb.return_value = mock_tb
        
        tb = TablebaseInterface("/valid/path")
        
        board = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
        position = Position(board)
        
        # Should not raise exception, should return None
        result = tb.probe(position)
        self.assertIsNone(result)


class TestTablebaseIntegration(unittest.TestCase):
    """Integration tests for tablebase with real positions."""
    
    def test_fallback_behavior_without_tablebase(self):
        """Test that system works correctly without tablebase."""
        tb = TablebaseInterface()
        
        # Create a simple king-pawn endgame position
        board = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
        position = Position(board)
        
        # All operations should return None gracefully
        self.assertIsNone(tb.probe(position))
        self.assertIsNone(tb.get_best_move(position))
        self.assertIsNone(tb.probe_dtz(position))
        self.assertFalse(tb.is_available())
        self.assertEqual(tb.get_max_pieces(), 0)
    
    def test_multiple_positions(self):
        """Test tablebase with multiple different positions."""
        tb = TablebaseInterface()
        
        positions = [
            "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1",  # KPK
            "8/8/8/8/4k3/8/4P3/4K3 w - - 0 1",  # KPK different
            "8/8/8/4k3/8/8/8/4K3 w - - 0 1",     # KK (draw)
        ]
        
        for fen in positions:
            board = chess.Board(fen)
            position = Position(board)
            
            # Should handle all positions gracefully
            result = tb.probe(position)
            self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
