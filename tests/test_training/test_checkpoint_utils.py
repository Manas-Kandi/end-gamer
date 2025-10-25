"""Unit tests for checkpoint recovery utilities."""

import os
import tempfile
import shutil
from pathlib import Path
import pytest
import torch

from src.training.checkpoint_utils import (
    validate_checkpoint,
    load_checkpoint_with_fallback,
    find_checkpoints,
    cleanup_old_checkpoints,
    get_latest_checkpoint,
    safe_save_checkpoint,
)
from src.exceptions import (
    CheckpointLoadError,
    CheckpointValidationError,
    CheckpointSaveError,
)
from src.config.config import Config


class TestValidateCheckpoint:
    """Test checkpoint validation."""
    
    def test_validate_nonexistent_checkpoint(self):
        """Test validation of non-existent checkpoint."""
        errors = validate_checkpoint("/nonexistent/checkpoint.pt")
        
        assert len(errors) > 0
        assert any("does not exist" in error for error in errors)
    
    def test_validate_valid_checkpoint(self, tmp_path):
        """Test validation of valid checkpoint."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        
        # Create valid checkpoint
        checkpoint = {
            'model_state_dict': {'layer1.weight': torch.randn(10, 10)},
            'iteration': 5,
            'total_games': 1000,
            'config': Config(),
        }
        torch.save(checkpoint, checkpoint_path)
        
        errors = validate_checkpoint(str(checkpoint_path))
        assert len(errors) == 0
    
    def test_validate_checkpoint_missing_fields(self, tmp_path):
        """Test validation of checkpoint with missing required fields."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        
        # Create checkpoint missing required fields
        checkpoint = {
            'model_state_dict': {'layer1.weight': torch.randn(10, 10)},
            # Missing: iteration, total_games, config
        }
        torch.save(checkpoint, checkpoint_path)
        
        errors = validate_checkpoint(str(checkpoint_path))
        
        assert len(errors) > 0
        assert any("iteration" in error for error in errors)
        assert any("total_games" in error for error in errors)
        assert any("config" in error for error in errors)
    
    def test_validate_checkpoint_empty_model_state(self, tmp_path):
        """Test validation of checkpoint with empty model state."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        
        checkpoint = {
            'model_state_dict': {},  # Empty
            'iteration': 5,
            'total_games': 1000,
            'config': Config(),
        }
        torch.save(checkpoint, checkpoint_path)
        
        errors = validate_checkpoint(str(checkpoint_path))
        
        assert len(errors) > 0
        assert any("empty" in error.lower() for error in errors)
    
    def test_validate_checkpoint_invalid_iteration(self, tmp_path):
        """Test validation of checkpoint with invalid iteration."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        
        checkpoint = {
            'model_state_dict': {'layer1.weight': torch.randn(10, 10)},
            'iteration': -1,  # Invalid
            'total_games': 1000,
            'config': Config(),
        }
        torch.save(checkpoint, checkpoint_path)
        
        errors = validate_checkpoint(str(checkpoint_path))
        
        assert len(errors) > 0
        assert any("iteration" in error for error in errors)
    
    def test_validate_corrupted_checkpoint(self, tmp_path):
        """Test validation of corrupted checkpoint file."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        
        # Write corrupted data
        with open(checkpoint_path, 'wb') as f:
            f.write(b'corrupted data')
        
        errors = validate_checkpoint(str(checkpoint_path))
        
        assert len(errors) > 0
        assert any("Failed to load" in error for error in errors)


class TestLoadCheckpointWithFallback:
    """Test checkpoint loading with fallback."""
    
    def test_load_valid_checkpoint(self, tmp_path):
        """Test loading a valid checkpoint."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        
        checkpoint = {
            'model_state_dict': {'layer1.weight': torch.randn(10, 10)},
            'iteration': 5,
            'total_games': 1000,
            'config': Config(),
        }
        torch.save(checkpoint, checkpoint_path)
        
        loaded = load_checkpoint_with_fallback(str(checkpoint_path))
        
        assert loaded['iteration'] == 5
        assert loaded['total_games'] == 1000
    
    def test_load_with_fallback_to_previous(self, tmp_path):
        """Test fallback to previous checkpoint when primary fails."""
        # Create corrupted primary checkpoint
        primary_path = tmp_path / "checkpoint_2000.pt"
        with open(primary_path, 'wb') as f:
            f.write(b'corrupted')
        
        # Create valid fallback checkpoint
        fallback_path = tmp_path / "checkpoint_1000.pt"
        checkpoint = {
            'model_state_dict': {'layer1.weight': torch.randn(10, 10)},
            'iteration': 3,
            'total_games': 1000,
            'config': Config(),
        }
        torch.save(checkpoint, fallback_path)
        
        # Should fallback to valid checkpoint
        loaded = load_checkpoint_with_fallback(
            str(primary_path),
            checkpoint_dir=str(tmp_path)
        )
        
        assert loaded['total_games'] == 1000
    
    def test_load_fails_when_all_invalid(self, tmp_path):
        """Test that loading fails when all checkpoints are invalid."""
        # Create corrupted checkpoints
        for i in range(3):
            checkpoint_path = tmp_path / f"checkpoint_{i}.pt"
            with open(checkpoint_path, 'wb') as f:
                f.write(b'corrupted')
        
        with pytest.raises(CheckpointLoadError):
            load_checkpoint_with_fallback(
                str(tmp_path / "checkpoint_0.pt"),
                checkpoint_dir=str(tmp_path)
            )
    
    def test_load_respects_max_fallback_attempts(self, tmp_path):
        """Test that fallback respects max attempts limit."""
        # Create many corrupted checkpoints
        for i in range(10):
            checkpoint_path = tmp_path / f"checkpoint_{i:04d}.pt"
            with open(checkpoint_path, 'wb') as f:
                f.write(b'corrupted')
        
        # Should only try max_fallback_attempts
        with pytest.raises(CheckpointLoadError) as exc_info:
            load_checkpoint_with_fallback(
                str(tmp_path / "checkpoint_0000.pt"),
                checkpoint_dir=str(tmp_path),
                max_fallback_attempts=3
            )
        
        assert "3 fallback attempts" in str(exc_info.value)


class TestFindCheckpoints:
    """Test finding checkpoints in directory."""
    
    def test_find_checkpoints_empty_directory(self, tmp_path):
        """Test finding checkpoints in empty directory."""
        checkpoints = find_checkpoints(str(tmp_path))
        assert len(checkpoints) == 0
    
    def test_find_checkpoints_sorted_by_time(self, tmp_path):
        """Test that checkpoints are sorted by modification time."""
        import time
        
        # Create checkpoints with different timestamps
        paths = []
        for i in range(3):
            checkpoint_path = tmp_path / f"checkpoint_{i}.pt"
            checkpoint = {
                'model_state_dict': {'layer1.weight': torch.randn(10, 10)},
                'iteration': i,
                'total_games': i * 1000,
                'config': Config(),
            }
            torch.save(checkpoint, checkpoint_path)
            paths.append(checkpoint_path)
            time.sleep(0.01)  # Ensure different timestamps
        
        checkpoints = find_checkpoints(str(tmp_path))
        
        # Should be sorted newest first
        assert len(checkpoints) == 3
        assert checkpoints[0] == str(paths[2])
        assert checkpoints[1] == str(paths[1])
        assert checkpoints[2] == str(paths[0])
    
    def test_find_checkpoints_with_pattern(self, tmp_path):
        """Test finding checkpoints with custom pattern."""
        # Create various files
        torch.save({}, tmp_path / "checkpoint_1.pt")
        torch.save({}, tmp_path / "checkpoint_2.pt")
        torch.save({}, tmp_path / "other_file.pt")
        
        checkpoints = find_checkpoints(str(tmp_path), pattern="checkpoint_*.pt")
        
        assert len(checkpoints) == 2
        assert all("checkpoint_" in cp for cp in checkpoints)


class TestCleanupOldCheckpoints:
    """Test checkpoint cleanup."""
    
    def test_cleanup_keeps_recent_checkpoints(self, tmp_path):
        """Test that cleanup keeps the most recent checkpoints."""
        # Create 10 checkpoints
        for i in range(10):
            checkpoint_path = tmp_path / f"checkpoint_{i:04d}.pt"
            checkpoint = {
                'model_state_dict': {'layer1.weight': torch.randn(10, 10)},
                'iteration': i,
                'total_games': i * 1000,
                'config': Config(),
            }
            torch.save(checkpoint, checkpoint_path)
        
        # Keep only 5
        deleted = cleanup_old_checkpoints(str(tmp_path), keep_count=5)
        
        assert deleted == 5
        remaining = find_checkpoints(str(tmp_path))
        assert len(remaining) == 5
    
    def test_cleanup_preserves_pattern_matches(self, tmp_path):
        """Test that cleanup preserves checkpoints matching keep pattern."""
        import time
        
        # Create final checkpoint first (older)
        final_path = tmp_path / "checkpoint_final.pt"
        torch.save({}, final_path)
        time.sleep(0.01)
        
        # Create checkpoints (newer)
        for i in range(5):
            checkpoint_path = tmp_path / f"checkpoint_{i}.pt"
            torch.save({}, checkpoint_path)
            time.sleep(0.01)
        
        # Cleanup keeping only 2, but preserve final
        deleted = cleanup_old_checkpoints(
            str(tmp_path),
            keep_count=2,
            keep_pattern="*_final.pt"
        )
        
        # Should keep 2 recent + final = 3 total
        remaining = find_checkpoints(str(tmp_path))
        assert len(remaining) == 3
        assert str(final_path) in remaining
    
    def test_cleanup_preserves_latest_checkpoint(self, tmp_path):
        """Test that cleanup always preserves latest_checkpoint.pt."""
        # Create checkpoints
        for i in range(5):
            checkpoint_path = tmp_path / f"checkpoint_{i}.pt"
            torch.save({}, checkpoint_path)
        
        # Create latest checkpoint
        latest_path = tmp_path / "latest_checkpoint.pt"
        torch.save({}, latest_path)
        
        # Cleanup keeping only 2
        cleanup_old_checkpoints(str(tmp_path), keep_count=2)
        
        # latest_checkpoint.pt should still exist
        assert latest_path.exists()
    
    def test_cleanup_nonexistent_directory(self):
        """Test cleanup with non-existent directory."""
        deleted = cleanup_old_checkpoints("/nonexistent/directory")
        assert deleted == 0


class TestGetLatestCheckpoint:
    """Test getting latest checkpoint."""
    
    def test_get_latest_checkpoint_prefers_latest_file(self, tmp_path):
        """Test that get_latest_checkpoint prefers latest_checkpoint.pt."""
        # Create regular checkpoints
        for i in range(3):
            checkpoint_path = tmp_path / f"checkpoint_{i}.pt"
            checkpoint = {
                'model_state_dict': {'layer1.weight': torch.randn(10, 10)},
                'iteration': i,
                'total_games': i * 1000,
                'config': Config(),
            }
            torch.save(checkpoint, checkpoint_path)
        
        # Create latest checkpoint
        latest_path = tmp_path / "latest_checkpoint.pt"
        checkpoint = {
            'model_state_dict': {'layer1.weight': torch.randn(10, 10)},
            'iteration': 10,
            'total_games': 10000,
            'config': Config(),
        }
        torch.save(checkpoint, latest_path)
        
        latest = get_latest_checkpoint(str(tmp_path))
        assert latest == str(latest_path)
    
    def test_get_latest_checkpoint_falls_back(self, tmp_path):
        """Test fallback when latest_checkpoint.pt is invalid."""
        # Create invalid latest checkpoint
        latest_path = tmp_path / "latest_checkpoint.pt"
        with open(latest_path, 'wb') as f:
            f.write(b'corrupted')
        
        # Create valid checkpoint
        checkpoint_path = tmp_path / "checkpoint_1.pt"
        checkpoint = {
            'model_state_dict': {'layer1.weight': torch.randn(10, 10)},
            'iteration': 1,
            'total_games': 1000,
            'config': Config(),
        }
        torch.save(checkpoint, checkpoint_path)
        
        latest = get_latest_checkpoint(str(tmp_path))
        assert latest == str(checkpoint_path)
    
    def test_get_latest_checkpoint_empty_directory(self, tmp_path):
        """Test getting latest checkpoint from empty directory."""
        latest = get_latest_checkpoint(str(tmp_path))
        assert latest is None


class TestSafeSaveCheckpoint:
    """Test safe checkpoint saving."""
    
    def test_safe_save_atomic(self, tmp_path):
        """Test atomic checkpoint saving."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        
        checkpoint = {
            'model_state_dict': {'layer1.weight': torch.randn(10, 10)},
            'iteration': 5,
            'total_games': 1000,
            'config': Config(),
        }
        
        safe_save_checkpoint(checkpoint, str(checkpoint_path), atomic=True)
        
        assert checkpoint_path.exists()
        
        # Verify checkpoint is valid
        errors = validate_checkpoint(str(checkpoint_path))
        assert len(errors) == 0
        
        # Verify no temp file left behind
        temp_path = checkpoint_path.with_suffix('.pt.tmp')
        assert not temp_path.exists()
    
    def test_safe_save_non_atomic(self, tmp_path):
        """Test non-atomic checkpoint saving."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        
        checkpoint = {
            'model_state_dict': {'layer1.weight': torch.randn(10, 10)},
            'iteration': 5,
            'total_games': 1000,
            'config': Config(),
        }
        
        safe_save_checkpoint(checkpoint, str(checkpoint_path), atomic=False)
        
        assert checkpoint_path.exists()
        errors = validate_checkpoint(str(checkpoint_path))
        assert len(errors) == 0
    
    def test_safe_save_creates_directory(self, tmp_path):
        """Test that safe_save creates parent directories."""
        checkpoint_path = tmp_path / "subdir" / "checkpoint.pt"
        
        checkpoint = {
            'model_state_dict': {'layer1.weight': torch.randn(10, 10)},
            'iteration': 5,
            'total_games': 1000,
            'config': Config(),
        }
        
        safe_save_checkpoint(checkpoint, str(checkpoint_path))
        
        assert checkpoint_path.exists()
        assert checkpoint_path.parent.exists()
    
    def test_safe_save_cleans_up_on_failure(self, tmp_path):
        """Test that temp file is cleaned up on save failure."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        
        # Create invalid checkpoint (will fail validation)
        checkpoint = {
            'model_state_dict': {},  # Empty - will fail validation
            'iteration': 5,
            'total_games': 1000,
            'config': Config(),
        }
        
        with pytest.raises(CheckpointSaveError):
            safe_save_checkpoint(checkpoint, str(checkpoint_path), atomic=True)
        
        # Temp file should be cleaned up
        temp_path = checkpoint_path.with_suffix('.pt.tmp')
        assert not temp_path.exists()


class TestCheckpointRecoveryIntegration:
    """Integration tests for checkpoint recovery scenarios."""
    
    def test_recovery_from_interrupted_training(self, tmp_path):
        """Test recovery scenario: training interrupted, last checkpoint corrupted."""
        # Simulate training with multiple checkpoints
        for i in range(5):
            checkpoint_path = tmp_path / f"checkpoint_{i * 1000}.pt"
            checkpoint = {
                'model_state_dict': {'layer1.weight': torch.randn(10, 10)},
                'iteration': i,
                'total_games': i * 1000,
                'config': Config(),
            }
            torch.save(checkpoint, checkpoint_path)
        
        # Simulate interrupted save - corrupt latest checkpoint
        latest_path = tmp_path / "checkpoint_5000.pt"
        with open(latest_path, 'wb') as f:
            f.write(b'corrupted during save')
        
        # Should recover from previous checkpoint
        loaded = load_checkpoint_with_fallback(
            str(latest_path),
            checkpoint_dir=str(tmp_path)
        )
        
        assert loaded['total_games'] == 4000  # Previous checkpoint
    
    def test_cleanup_after_successful_training(self, tmp_path):
        """Test cleanup scenario: keep only important checkpoints after training."""
        import time
        
        # Create final checkpoint first (older)
        final_path = tmp_path / "checkpoint_final.pt"
        torch.save({}, final_path)
        time.sleep(0.01)
        
        # Create many checkpoints during training (newer)
        for i in range(20):
            checkpoint_path = tmp_path / f"checkpoint_{i * 1000}.pt"
            torch.save({}, checkpoint_path)
            time.sleep(0.001)
        
        # Cleanup keeping only 3 recent + final
        deleted = cleanup_old_checkpoints(
            str(tmp_path),
            keep_count=3,
            keep_pattern="*_final.pt"
        )
        
        assert deleted == 17  # 21 total - 3 recent - 1 final = 17
        remaining = find_checkpoints(str(tmp_path))
        assert len(remaining) == 4  # 3 recent + final
