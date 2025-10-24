"""Replay buffer for storing and sampling training examples."""

from typing import List, Tuple, Optional
from collections import deque
import numpy as np
import torch
import random
from .training_example import TrainingExample


class ReplayBuffer:
    """Circular buffer for storing training examples from self-play games.
    
    Maintains a fixed-size buffer of training examples and provides efficient
    random sampling for neural network training. Uses a deque for O(1) append
    and automatic size management.
    """
    
    def __init__(self, max_size: int = 100000, seed: Optional[int] = None):
        """Initialize replay buffer.
        
        Args:
            max_size: Maximum number of examples to store
            seed: Random seed for reproducible sampling
        """
        if max_size <= 0:
            raise ValueError("Buffer max_size must be positive")
        
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def add_examples(self, examples: List[TrainingExample]) -> None:
        """Add training examples to buffer.
        
        Args:
            examples: List of TrainingExample objects to add
        """
        if not isinstance(examples, list):
            raise TypeError("Examples must be a list")
        
        for example in examples:
            if not isinstance(example, TrainingExample):
                raise TypeError("All examples must be TrainingExample instances")
            self.buffer.append(example)
    
    def add_example(self, example: TrainingExample) -> None:
        """Add single training example to buffer.
        
        Args:
            example: TrainingExample to add
        """
        if not isinstance(example, TrainingExample):
            raise TypeError("Example must be TrainingExample instance")
        
        self.buffer.append(example)
    
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample random batch of training examples.
        
        Args:
            batch_size: Number of examples to sample
            
        Returns:
            Tuple of (positions, policies, values):
                - positions: (batch_size, 12, 8, 8) tensor
                - policies: (batch_size, 4096) tensor  
                - values: (batch_size, 1) tensor
                
        Raises:
            ValueError: If batch_size > buffer size or batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if batch_size > len(self.buffer):
            raise ValueError(f"Batch size {batch_size} exceeds buffer size {len(self.buffer)}")
        
        # Sample random indices without replacement
        indices = random.sample(range(len(self.buffer)), batch_size)
        
        # Collect data
        positions = []
        policies = []
        values = []
        
        for idx in indices:
            example = self.buffer[idx]
            # Convert position from (8,8,12) to (12,8,8) for PyTorch
            position = np.transpose(example.position, (2, 0, 1))
            positions.append(position)
            policies.append(example.policy)
            values.append(example.value)
        
        # Convert to tensors
        positions_tensor = torch.FloatTensor(np.array(positions))
        policies_tensor = torch.FloatTensor(np.array(policies))
        values_tensor = torch.FloatTensor(np.array(values)).unsqueeze(1)
        
        return positions_tensor, policies_tensor, values_tensor
    
    def sample_indices(self, batch_size: int) -> List[int]:
        """Sample random indices from buffer.
        
        Args:
            batch_size: Number of indices to sample
            
        Returns:
            List of random indices
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if batch_size > len(self.buffer):
            raise ValueError(f"Batch size {batch_size} exceeds buffer size {len(self.buffer)}")
        
        return random.sample(range(len(self.buffer)), batch_size)
    
    def get_examples_by_indices(self, indices: List[int]) -> List[TrainingExample]:
        """Get training examples by indices.
        
        Args:
            indices: List of indices to retrieve
            
        Returns:
            List of TrainingExample objects
        """
        examples = []
        for idx in indices:
            if not 0 <= idx < len(self.buffer):
                raise IndexError(f"Index {idx} out of range [0, {len(self.buffer)})")
            examples.append(self.buffer[idx])
        
        return examples
    
    def clear(self) -> None:
        """Clear all examples from buffer."""
        self.buffer.clear()
    
    def is_full(self) -> bool:
        """Check if buffer is at maximum capacity.
        
        Returns:
            True if buffer is full, False otherwise
        """
        return len(self.buffer) >= self.max_size
    
    def get_memory_usage(self) -> int:
        """Get approximate memory usage in bytes.
        
        Returns:
            Approximate memory usage in bytes
        """
        if len(self.buffer) == 0:
            return 0
        
        # Estimate based on first example
        example_size = self.buffer[0].get_memory_size()
        return example_size * len(self.buffer)
    
    def get_statistics(self) -> dict:
        """Get buffer statistics.
        
        Returns:
            Dictionary with buffer statistics
        """
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'max_size': self.max_size,
                'utilization': 0.0,
                'memory_usage_mb': 0.0,
                'avg_value': 0.0,
                'value_std': 0.0
            }
        
        values = [example.value for example in self.buffer]
        
        return {
            'size': len(self.buffer),
            'max_size': self.max_size,
            'utilization': len(self.buffer) / self.max_size,
            'memory_usage_mb': self.get_memory_usage() / (1024 * 1024),
            'avg_value': np.mean(values),
            'value_std': np.std(values)
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save buffer contents to file.
        
        Args:
            filepath: Path to save buffer data
        """
        import pickle
        
        data = {
            'max_size': self.max_size,
            'examples': list(self.buffer)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_from_file(self, filepath: str) -> None:
        """Load buffer contents from file.
        
        Args:
            filepath: Path to load buffer data from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Validate loaded data
        if 'max_size' not in data or 'examples' not in data:
            raise ValueError("Invalid buffer file format")
        
        # Update buffer
        self.max_size = data['max_size']
        self.buffer = deque(data['examples'], maxlen=self.max_size)
    
    def __len__(self) -> int:
        """Get number of examples in buffer."""
        return len(self.buffer)
    
    def __getitem__(self, index: int) -> TrainingExample:
        """Get example by index."""
        if not 0 <= index < len(self.buffer):
            raise IndexError(f"Index {index} out of range [0, {len(self.buffer)})")
        return self.buffer[index]
    
    def __iter__(self):
        """Iterate over examples in buffer."""
        return iter(self.buffer)
    
    def __repr__(self) -> str:
        """String representation of buffer."""
        return (f"ReplayBuffer(size={len(self.buffer)}, "
                f"max_size={self.max_size}, "
                f"utilization={len(self.buffer)/self.max_size:.1%})")