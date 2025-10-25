"""Tablebase integration for Syzygy endgame tablebases."""

from typing import Optional
import chess
import chess.syzygy
import os
from pathlib import Path

from ..chess_env.position import Position


class TablebaseInterface:
    """Interface for Syzygy tablebase access with graceful fallback."""
    
    def __init__(self, tablebase_path: Optional[str] = None):
        """Initialize tablebase interface.
        
        Args:
            tablebase_path: Path to Syzygy tablebase files. If None, will check
                          environment variable TABLEBASE_PATH or disable tablebase.
        """
        self.available = False
        self.tablebase = None
        
        # Try to find tablebase path
        if tablebase_path is None:
            tablebase_path = os.environ.get('TABLEBASE_PATH')
        
        if tablebase_path and os.path.exists(tablebase_path):
            try:
                self.tablebase = chess.syzygy.open_tablebase(tablebase_path)
                self.available = True
                print(f"Tablebase loaded from: {tablebase_path}")
            except Exception as e:
                print(f"Warning: Failed to load tablebase from {tablebase_path}: {e}")
                self.available = False
        else:
            if tablebase_path:
                print(f"Warning: Tablebase path not found: {tablebase_path}")
            else:
                print("Info: No tablebase path specified. Tablebase features disabled.")
    
    def probe(self, position: Position) -> Optional[float]:
        """Probe tablebase for position evaluation.
        
        Args:
            position: Position to evaluate
            
        Returns:
            Position value from current player perspective:
            - 1.0 for winning position
            - 0.0 for drawn position
            - -1.0 for losing position
            - None if tablebase unavailable or position not in tablebase
        """
        if not self.available or self.tablebase is None:
            return None
        
        try:
            board = position.board
            
            # Check if position is in tablebase (piece count)
            if len(board.piece_map()) > self.tablebase.max_fds:
                return None
            
            # Probe WDL (Win/Draw/Loss)
            wdl = self.tablebase.probe_wdl(board)
            
            # Convert WDL to our value format
            # WDL values: 2=win, 1=cursed win, 0=draw, -1=blessed loss, -2=loss
            if wdl >= 1:
                return 1.0  # Win or cursed win
            elif wdl <= -1:
                return -1.0  # Loss or blessed loss
            else:
                return 0.0  # Draw
                
        except (KeyError, chess.syzygy.MissingTableError):
            # Position not in tablebase
            return None
        except Exception as e:
            print(f"Warning: Tablebase probe error: {e}")
            return None
    
    def get_best_move(self, position: Position) -> Optional[chess.Move]:
        """Get best move from tablebase.
        
        Args:
            position: Position to analyze
            
        Returns:
            Best move according to tablebase, or None if unavailable
        """
        if not self.available or self.tablebase is None:
            return None
        
        try:
            board = position.board
            
            # Check if position is in tablebase
            if len(board.piece_map()) > self.tablebase.max_fds:
                return None
            
            # Get all legal moves and their WDL values
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None
            
            best_move = None
            best_wdl = -float('inf')
            best_dtz = float('inf')  # Distance to zeroing (for tie-breaking)
            
            for move in legal_moves:
                # Make move temporarily
                board.push(move)
                
                try:
                    # Probe WDL for resulting position (from opponent's perspective)
                    wdl = -self.tablebase.probe_wdl(board)
                    
                    # Try to get DTZ for tie-breaking
                    try:
                        dtz = abs(self.tablebase.probe_dtz(board))
                    except:
                        dtz = 100  # Default if DTZ unavailable
                    
                    # Select move with best WDL, breaking ties with DTZ
                    if wdl > best_wdl or (wdl == best_wdl and dtz < best_dtz):
                        best_wdl = wdl
                        best_dtz = dtz
                        best_move = move
                        
                except (KeyError, chess.syzygy.MissingTableError):
                    pass
                finally:
                    board.pop()
            
            return best_move
            
        except Exception as e:
            print(f"Warning: Tablebase move lookup error: {e}")
            return None
    
    def probe_dtz(self, position: Position) -> Optional[int]:
        """Probe distance to zeroing (DTZ) from tablebase.
        
        Args:
            position: Position to analyze
            
        Returns:
            Distance to zeroing (number of moves until pawn move or capture),
            or None if unavailable
        """
        if not self.available or self.tablebase is None:
            return None
        
        try:
            board = position.board
            
            if len(board.piece_map()) > self.tablebase.max_fds:
                return None
            
            dtz = self.tablebase.probe_dtz(board)
            return dtz
            
        except (KeyError, chess.syzygy.MissingTableError):
            return None
        except Exception as e:
            print(f"Warning: Tablebase DTZ probe error: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if tablebase is available.
        
        Returns:
            True if tablebase is loaded and available
        """
        return self.available
    
    def get_max_pieces(self) -> int:
        """Get maximum number of pieces supported by loaded tablebase.
        
        Returns:
            Maximum piece count, or 0 if tablebase unavailable
        """
        if self.available and self.tablebase:
            return self.tablebase.max_fds
        return 0
