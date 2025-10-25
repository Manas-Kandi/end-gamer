"""Checkpoint recovery and management utilities."""

import os
import glob
from pathlib import Path
from typing import Optional, List, Dict, Any
import torch
import logging

from ..exceptions import (
    CheckpointLoadError,
    CheckpointValidationError,
    CheckpointSaveError,
)

logger = logging.getLogger(__name__)


def validate_checkpoint(checkpoint_path: str) -> List[str]:
    """Validate checkpoint file structure and contents.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check file exists
    if not os.path.exists(checkpoint_path):
        errors.append(f"Checkpoint file does not exist: {checkpoint_path}")
        return errors
    
    # Check file is readable
    if not os.access(checkpoint_path, os.R_OK):
        errors.append(f"Checkpoint file is not readable: {checkpoint_path}")
        return errors
    
    # Try to load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        errors.append(f"Failed to load checkpoint file: {str(e)}")
        return errors
    
    # Validate required fields
    required_fields = [
        'model_state_dict',
        'iteration',
        'total_games',
        'config',
    ]
    
    for field in required_fields:
        if field not in checkpoint:
            errors.append(f"Missing required field: {field}")
    
    # Validate model_state_dict is a dictionary
    if 'model_state_dict' in checkpoint:
        if not isinstance(checkpoint['model_state_dict'], dict):
            errors.append("model_state_dict must be a dictionary")
        elif len(checkpoint['model_state_dict']) == 0:
            errors.append("model_state_dict is empty")
    
    # Validate numeric fields
    if 'iteration' in checkpoint:
        if not isinstance(checkpoint['iteration'], int) or checkpoint['iteration'] < 0:
            errors.append("iteration must be a non-negative integer")
    
    if 'total_games' in checkpoint:
        if not isinstance(checkpoint['total_games'], int) or checkpoint['total_games'] < 0:
            errors.append("total_games must be a non-negative integer")
    
    # Validate config
    if 'config' in checkpoint:
        config = checkpoint['config']
        if not hasattr(config, 'device'):
            errors.append("config missing required attribute: device")
        if not hasattr(config, 'num_res_blocks'):
            errors.append("config missing required attribute: num_res_blocks")
        if not hasattr(config, 'num_filters'):
            errors.append("config missing required attribute: num_filters")
    
    return errors


def load_checkpoint_with_fallback(
    checkpoint_path: str,
    checkpoint_dir: Optional[str] = None,
    max_fallback_attempts: int = 5,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Load checkpoint with fallback to previous checkpoints on failure.
    
    Attempts to load the specified checkpoint. If it fails validation or loading,
    tries to load previous checkpoints from the same directory.
    
    Args:
        checkpoint_path: Primary checkpoint path to load
        checkpoint_dir: Directory containing checkpoints (for fallback)
        max_fallback_attempts: Maximum number of fallback attempts
        device: Device to load checkpoint to
        
    Returns:
        Loaded checkpoint dictionary
        
    Raises:
        CheckpointLoadError: If all checkpoint loading attempts fail
    """
    # Try primary checkpoint first
    validation_errors = validate_checkpoint(checkpoint_path)
    
    if not validation_errors:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            logger.info(f"Successfully loaded checkpoint: {checkpoint_path}")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
    else:
        logger.warning(f"Checkpoint validation failed for {checkpoint_path}: {validation_errors}")
    
    # If primary failed and we have a checkpoint directory, try fallbacks
    if checkpoint_dir is None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
    
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        raise CheckpointLoadError(
            checkpoint_path,
            f"Primary checkpoint failed and no valid checkpoint directory: {checkpoint_dir}"
        )
    
    # Find all checkpoints in directory
    fallback_checkpoints = find_checkpoints(checkpoint_dir)
    
    # Remove the primary checkpoint from fallbacks
    fallback_checkpoints = [cp for cp in fallback_checkpoints if cp != checkpoint_path]
    
    # Sort by modification time (newest first)
    fallback_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Try fallback checkpoints
    attempts = 0
    for fallback_path in fallback_checkpoints:
        if attempts >= max_fallback_attempts:
            break
        
        attempts += 1
        logger.info(f"Attempting fallback checkpoint {attempts}/{max_fallback_attempts}: {fallback_path}")
        
        validation_errors = validate_checkpoint(fallback_path)
        if validation_errors:
            logger.warning(f"Fallback checkpoint validation failed: {validation_errors}")
            continue
        
        try:
            checkpoint = torch.load(fallback_path, map_location=device, weights_only=False)
            logger.info(f"Successfully loaded fallback checkpoint: {fallback_path}")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load fallback checkpoint {fallback_path}: {e}")
    
    # All attempts failed
    raise CheckpointLoadError(
        checkpoint_path,
        f"Failed to load checkpoint after {attempts} fallback attempts"
    )


def find_checkpoints(checkpoint_dir: str, pattern: str = "checkpoint_*.pt") -> List[str]:
    """Find all checkpoint files in a directory.
    
    Args:
        checkpoint_dir: Directory to search
        pattern: Glob pattern for checkpoint files
        
    Returns:
        List of checkpoint file paths sorted by modification time (newest first)
    """
    checkpoint_pattern = os.path.join(checkpoint_dir, pattern)
    checkpoints = glob.glob(checkpoint_pattern)
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return checkpoints


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_count: int = 5,
    keep_pattern: Optional[str] = None
) -> int:
    """Clean up old checkpoint files, keeping only the most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_count: Number of recent checkpoints to keep
        keep_pattern: Optional pattern for checkpoints to always keep (e.g., "*_final.pt")
        
    Returns:
        Number of checkpoints deleted
    """
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return 0
    
    # Find all checkpoints
    all_checkpoints = find_checkpoints(checkpoint_dir)
    
    # Find checkpoints to keep based on pattern
    keep_checkpoints = set()
    if keep_pattern:
        keep_checkpoints = set(glob.glob(os.path.join(checkpoint_dir, keep_pattern)))
    
    # Always keep latest_checkpoint.pt
    latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    if os.path.exists(latest_checkpoint):
        keep_checkpoints.add(latest_checkpoint)
    
    # Keep the most recent checkpoints
    if len(all_checkpoints) > keep_count:
        keep_checkpoints.update(all_checkpoints[:keep_count])
    else:
        keep_checkpoints.update(all_checkpoints)
    
    # Delete old checkpoints
    deleted_count = 0
    for checkpoint_path in all_checkpoints:
        if checkpoint_path not in keep_checkpoints:
            try:
                os.remove(checkpoint_path)
                logger.info(f"Deleted old checkpoint: {checkpoint_path}")
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {checkpoint_path}: {e}")
    
    return deleted_count


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get the path to the most recent checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    # Check for latest_checkpoint.pt first
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    if os.path.exists(latest_path):
        validation_errors = validate_checkpoint(latest_path)
        if not validation_errors:
            return latest_path
        else:
            logger.warning(f"latest_checkpoint.pt failed validation: {validation_errors}")
    
    # Find all checkpoints
    checkpoints = find_checkpoints(checkpoint_dir)
    
    # Return the most recent valid checkpoint
    for checkpoint_path in checkpoints:
        validation_errors = validate_checkpoint(checkpoint_path)
        if not validation_errors:
            return checkpoint_path
    
    return None


def safe_save_checkpoint(
    checkpoint: Dict[str, Any],
    checkpoint_path: str,
    atomic: bool = True
) -> None:
    """Safely save checkpoint with atomic write to prevent corruption.
    
    Args:
        checkpoint: Checkpoint dictionary to save
        checkpoint_path: Path to save checkpoint
        atomic: If True, use atomic write (write to temp file then rename)
        
    Raises:
        CheckpointSaveError: If checkpoint saving fails
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Create directory if it doesn't exist
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    if atomic:
        # Write to temporary file first
        temp_path = checkpoint_path.with_suffix('.pt.tmp')
        
        try:
            torch.save(checkpoint, temp_path)
            
            # Verify the temporary file
            validation_errors = validate_checkpoint(str(temp_path))
            if validation_errors:
                raise CheckpointValidationError(str(temp_path), validation_errors)
            
            # Atomic rename
            temp_path.replace(checkpoint_path)
            logger.info(f"Successfully saved checkpoint: {checkpoint_path}")
            
        except Exception as e:
            # Clean up temp file on failure
            if temp_path.exists():
                temp_path.unlink()
            raise CheckpointSaveError(str(checkpoint_path), str(e))
    else:
        # Direct write (not atomic)
        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Successfully saved checkpoint: {checkpoint_path}")
        except Exception as e:
            raise CheckpointSaveError(str(checkpoint_path), str(e))
