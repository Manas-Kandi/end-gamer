"""Unit tests for TrainingExample class."""

import pytest
import numpy as np
import pickle
import gzip
from src.training.training_example import TrainingExample


class TestTrainingExample:
    """Test cases for TrainingExample class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create valid test data
        self.position = np.random.rand(8, 8, 12).astype(np.float32)
        
        # Create valid policy (probability distribution)
        policy = np.random.rand(4096).astype(np.float32)
        self.policy = policy / np.sum(policy)  # Normalize to sum to 1
        
        self.value = 0.5
        
        self.example = TrainingExample(
            position=self.position,
            policy=self.policy,
            value=self.value
        )
    
    def test_initialization_valid(self):
        """Test valid initialization."""
        example = TrainingExample(
            position=self.position,
            policy=self.policy,
            value=self.value
        )
        
        assert np.array_equal(example.position, self.position)
        assert np.array_equal(example.policy, self.policy)
        assert example.value == self.value
    
    def test_initialization_auto_dtype_conversion(self):
        """Test automatic dtype conversion."""
        # Test with different dtypes
        position_int = np.ones((8, 8, 12), dtype=np.int32)
        policy_float64 = np.ones(4096, dtype=np.float64) / 4096
        
        example = TrainingExample(
            position=position_int,
            policy=policy_float64,
            value=1
        )
        
        assert example.position.dtype == np.float32
        assert example.policy.dtype == np.float32
        assert isinstance(example.value, float)
    
    def test_position_validation_type(self):
        """Test position type validation."""
        with pytest.raises(TypeError, match="Position must be numpy array"):
            TrainingExample(
                position=[[1, 2], [3, 4]],  # List instead of numpy array
                policy=self.policy,
                value=self.value
            )
    
    def test_position_validation_shape(self):
        """Test position shape validation."""
        with pytest.raises(ValueError, match="Position must have shape \\(8, 8, 12\\)"):
            TrainingExample(
                position=np.random.rand(8, 8, 6),  # Wrong shape
                policy=self.policy,
                value=self.value
            )
    
    def test_policy_validation_type(self):
        """Test policy type validation."""
        with pytest.raises(TypeError, match="Policy must be numpy array"):
            TrainingExample(
                position=self.position,
                policy=[0.1, 0.2, 0.7],  # List instead of numpy array
                value=self.value
            )
    
    def test_policy_validation_shape(self):
        """Test policy shape validation."""
        with pytest.raises(ValueError, match="Policy must have shape \\(4096,\\)"):
            TrainingExample(
                position=self.position,
                policy=np.random.rand(64),  # Wrong shape
                value=self.value
            )
    
    def test_policy_validation_sum(self):
        """Test policy probability sum validation."""
        invalid_policy = np.random.rand(4096) * 2  # Won't sum to 1
        
        with pytest.raises(ValueError, match="Policy must sum to 1.0"):
            TrainingExample(
                position=self.position,
                policy=invalid_policy,
                value=self.value
            )
    
    def test_policy_validation_negative(self):
        """Test policy negative values validation."""
        invalid_policy = np.ones(4096) / 4096
        invalid_policy[0] = -0.1  # Negative value
        
        with pytest.raises(ValueError, match="Policy probabilities must be non-negative"):
            TrainingExample(
                position=self.position,
                policy=invalid_policy,
                value=self.value
            )
    
    def test_value_validation_type(self):
        """Test value type validation."""
        with pytest.raises(TypeError, match="Value must be numeric"):
            TrainingExample(
                position=self.position,
                policy=self.policy,
                value="invalid"  # String instead of number
            )
    
    def test_value_validation_range(self):
        """Test value range validation."""
        with pytest.raises(ValueError, match="Value must be in range \\[-1, 1\\]"):
            TrainingExample(
                position=self.position,
                policy=self.policy,
                value=2.0  # Out of range
            )
        
        with pytest.raises(ValueError, match="Value must be in range \\[-1, 1\\]"):
            TrainingExample(
                position=self.position,
                policy=self.policy,
                value=-1.5  # Out of range
            )
    
    def test_value_boundary_values(self):
        """Test boundary values for value field."""
        # Test valid boundary values
        example_min = TrainingExample(
            position=self.position,
            policy=self.policy,
            value=-1.0
        )
        assert example_min.value == -1.0
        
        example_max = TrainingExample(
            position=self.position,
            policy=self.policy,
            value=1.0
        )
        assert example_max.value == 1.0
        
        example_zero = TrainingExample(
            position=self.position,
            policy=self.policy,
            value=0.0
        )
        assert example_zero.value == 0.0
    
    def test_to_bytes_serialization(self):
        """Test serialization to bytes."""
        serialized = self.example.to_bytes()
        
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0
        
        # Should be compressed (smaller than uncompressed pickle)
        uncompressed_pickle = pickle.dumps({
            'position': self.position,
            'policy': self.policy,
            'value': self.value
        })
        assert len(serialized) <= len(uncompressed_pickle)
    
    def test_from_bytes_deserialization(self):
        """Test deserialization from bytes."""
        serialized = self.example.to_bytes()
        deserialized = TrainingExample.from_bytes(serialized)
        
        assert np.array_equal(deserialized.position, self.example.position)
        assert np.array_equal(deserialized.policy, self.example.policy)
        assert deserialized.value == self.example.value
    
    def test_serialization_roundtrip(self):
        """Test complete serialization roundtrip."""
        # Serialize and deserialize
        serialized = self.example.to_bytes()
        deserialized = TrainingExample.from_bytes(serialized)
        
        # Should be equal to original
        assert deserialized == self.example
    
    def test_serialization_with_different_values(self):
        """Test serialization with various value types."""
        test_cases = [
            (np.zeros((8, 8, 12)), np.ones(4096) / 4096, -1.0),
            (np.ones((8, 8, 12)), np.ones(4096) / 4096, 0.0),
            (np.random.rand(8, 8, 12), np.random.rand(4096), 1.0)
        ]
        
        for position, policy, value in test_cases:
            policy = policy / np.sum(policy)  # Normalize
            example = TrainingExample(
                position=position.astype(np.float32),
                policy=policy.astype(np.float32),
                value=value
            )
            
            # Test roundtrip
            serialized = example.to_bytes()
            deserialized = TrainingExample.from_bytes(serialized)
            assert deserialized == example
    
    def test_get_memory_size(self):
        """Test memory size calculation."""
        memory_size = self.example.get_memory_size()
        
        expected_size = (
            self.position.nbytes +  # 8*8*12*4 = 3072 bytes
            self.policy.nbytes +    # 4096*4 = 16384 bytes
            8                       # float64 for value
        )
        
        assert memory_size == expected_size
        assert memory_size > 0
    
    def test_copy(self):
        """Test deep copy functionality."""
        copied = self.example.copy()
        
        # Should be equal but not the same object
        assert copied == self.example
        assert copied is not self.example
        
        # Arrays should be copies, not references
        assert copied.position is not self.example.position
        assert copied.policy is not self.example.policy
        
        # Modifying copy shouldn't affect original
        copied.position[0, 0, 0] = 999.0
        assert not np.array_equal(copied.position, self.example.position)
    
    def test_equality(self):
        """Test equality comparison."""
        # Same data should be equal
        other = TrainingExample(
            position=self.position.copy(),
            policy=self.policy.copy(),
            value=self.value
        )
        assert self.example == other
        
        # Different position should not be equal
        different_position = TrainingExample(
            position=np.zeros((8, 8, 12), dtype=np.float32),
            policy=self.policy.copy(),
            value=self.value
        )
        assert self.example != different_position
        
        # Different policy should not be equal
        different_policy = np.ones(4096, dtype=np.float32) / 4096
        different_policy_example = TrainingExample(
            position=self.position.copy(),
            policy=different_policy,
            value=self.value
        )
        assert self.example != different_policy_example
        
        # Different value should not be equal
        different_value = TrainingExample(
            position=self.position.copy(),
            policy=self.policy.copy(),
            value=-0.5
        )
        assert self.example != different_value
        
        # Different type should not be equal
        assert self.example != "not a training example"
        assert self.example != None
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.example)
        
        assert "TrainingExample" in repr_str
        assert "position_shape=(8, 8, 12)" in repr_str
        assert "policy_sum=" in repr_str
        assert f"value={self.value}" in repr_str
    
    def test_with_chess_like_data(self):
        """Test with realistic chess-like data."""
        # Create chess-like position (binary values)
        position = np.zeros((8, 8, 12), dtype=np.float32)
        position[0, 4, 6] = 1.0  # Black king on e8
        position[7, 4, 0] = 1.0  # White king on e1
        position[6, 4, 5] = 1.0  # White pawn on e2
        
        # Create sparse policy (only a few legal moves)
        policy = np.zeros(4096, dtype=np.float32)
        policy[796] = 0.7   # e2e4
        policy[540] = 0.2   # e2e3
        policy[1575] = 0.1  # Ke1f1
        
        example = TrainingExample(
            position=position,
            policy=policy,
            value=0.3
        )
        
        # Should work without issues
        assert example.position.shape == (8, 8, 12)
        assert example.policy.shape == (4096,)
        assert abs(np.sum(example.policy) - 1.0) < 1e-6
        
        # Test serialization with sparse data
        serialized = example.to_bytes()
        deserialized = TrainingExample.from_bytes(serialized)
        assert deserialized == example
    
    def test_large_batch_creation(self):
        """Test creating multiple training examples efficiently."""
        batch_size = 100
        examples = []
        
        for i in range(batch_size):
            position = np.random.rand(8, 8, 12).astype(np.float32)
            policy = np.random.rand(4096).astype(np.float32)
            policy = policy / np.sum(policy)
            value = np.random.uniform(-1, 1)
            
            example = TrainingExample(
                position=position,
                policy=policy,
                value=value
            )
            examples.append(example)
        
        assert len(examples) == batch_size
        
        # All should be valid
        for example in examples:
            assert example.position.shape == (8, 8, 12)
            assert example.policy.shape == (4096,)
            assert -1.0 <= example.value <= 1.0
    
    def test_memory_efficiency(self):
        """Test memory usage is reasonable."""
        # Single example should use reasonable memory
        memory_size = self.example.get_memory_size()
        
        # Should be around 19KB (3KB position + 16KB policy + 8B value)
        assert 15000 <= memory_size <= 25000
        
        # Serialized should be smaller due to compression
        serialized_size = len(self.example.to_bytes())
        assert serialized_size < memory_size  # Should be compressed