"""Training example data structure for self-play games."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pickle
import gzip


@dataclass
class TrainingExample:
    """Single training example from self-play.
    
    Contains the position, improved policy from MCTS, and final game result.
    Used to train the neural network on self-play data.
    """
    
    position: np.ndarray  # Shape (8, 8, 12) - board tensor
    policy: np.ndarray    # Shape (4096,) - improved policy from MCTS
    value: float          # Game result from position player's perspective: {-1, 0, 1}
    
    def __post_init__(self):
        """Validate data after initialization."""
        # Validate position shape
        if not isinstance(self.position, np.ndarray):
            raise TypeError("Position must be numpy array")
        if self.position.shape != (8, 8, 12):
            raise ValueError(f"Position must have shape (8, 8, 12), got {self.position.shape}")
        if self.position.dtype != np.float32:
            self.position = self.position.astype(np.float32)
        
        # Validate policy shape
        if not isinstance(self.policy, np.ndarray):
            raise TypeError("Policy must be numpy array")
        if self.policy.shape != (4096,):
            raise ValueError(f"Policy must have shape (4096,), got {self.policy.shape}")
        if self.policy.dtype != np.float32:
            self.policy = self.policy.astype(np.float32)
        
        # Validate policy is probability distribution
        if np.any(self.policy < 0):
            raise ValueError("Policy probabilities must be non-negative")
        if not np.allclose(np.sum(self.policy), 1.0, atol=1e-6):
            raise ValueError(f"Policy must sum to 1.0, got {np.sum(self.policy)}")
        
        # Validate value
        if not isinstance(self.value, (int, float, np.number)):
            raise TypeError("Value must be numeric")
        if not -1.0 <= self.value <= 1.0:
            raise ValueError(f"Value must be in range [-1, 1], got {self.value}")
        self.value = float(self.value)
    
    def to_bytes(self) -> bytes:
        """Serialize training example to compressed bytes.
        
        Returns:
            Compressed bytes representation of the training example
        """
        data = {
            'position': self.position,
            'policy': self.policy,
            'value': self.value
        }
        
        # Serialize with pickle and compress with gzip
        pickled_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        compressed_data = gzip.compress(pickled_data)
        
        return compressed_data
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'TrainingExample':
        """Deserialize training example from compressed bytes.
        
        Args:
            data: Compressed bytes representation
            
        Returns:
            TrainingExample instance
        """
        # Decompress and unpickle
        decompressed_data = gzip.decompress(data)
        unpickled_data = pickle.loads(decompressed_data)
        
        return cls(
            position=unpickled_data['position'],
            policy=unpickled_data['policy'],
            value=unpickled_data['value']
        )
    
    def get_memory_size(self) -> int:
        """Get approximate memory size in bytes.
        
        Returns:
            Approximate memory usage in bytes
        """
        position_size = self.position.nbytes
        policy_size = self.policy.nbytes
        value_size = 8  # float64
        
        return position_size + policy_size + value_size
    
    def copy(self) -> 'TrainingExample':
        """Create a deep copy of the training example.
        
        Returns:
            New TrainingExample instance with copied data
        """
        return TrainingExample(
            position=self.position.copy(),
            policy=self.policy.copy(),
            value=self.value
        )
    
    def __eq__(self, other) -> bool:
        """Check equality with another TrainingExample."""
        if not isinstance(other, TrainingExample):
            return False
        
        return (np.array_equal(self.position, other.position) and
                np.array_equal(self.policy, other.policy) and
                abs(self.value - other.value) < 1e-9)
    
    def __repr__(self) -> str:
        """String representation of training example."""
        return (f"TrainingExample(position_shape={self.position.shape}, "
                f"policy_sum={np.sum(self.policy):.6f}, value={self.value})")