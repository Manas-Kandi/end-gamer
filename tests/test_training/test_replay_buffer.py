"""Unit tests for ReplayBuffer class."""

import pytest
import numpy as np
import torch
import tempfile
import os
from src.training.replay_buffer import ReplayBuffer
from src.training.training_example import TrainingExample


class TestReplayBuffer:
    """Test cases for ReplayBuffer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.buffer = ReplayBuffer(max_size=100, seed=42)
        
        # Create sample training examples
        self.examples = []
        for i in range(10):
            position = np.random.rand(8, 8, 12).astype(np.float32)
            policy = np.random.rand(4096).astype(np.float32)
            policy = policy / np.sum(policy)  # Normalize
            value = np.random.uniform(-1, 1)
            
            example = TrainingExample(
                position=position,
                policy=policy,
                value=value
            )
            self.examples.append(example)
    
    def test_initialization_default(self):
        """Test default initialization."""
        buffer = ReplayBuffer()
        assert buffer.max_size == 100000
        assert len(buffer) == 0
        assert not buffer.is_full()
    
    def test_initialization_custom_size(self):
        """Test initialization with custom size."""
        buffer = ReplayBuffer(max_size=50)
        assert buffer.max_size == 50
        assert len(buffer) == 0
    
    def test_initialization_invalid_size(self):
        """Test initialization with invalid size."""
        with pytest.raises(ValueError, match="Buffer max_size must be positive"):
            ReplayBuffer(max_size=0)
        
        with pytest.raises(ValueError, match="Buffer max_size must be positive"):
            ReplayBuffer(max_size=-10)
    
    def test_add_example_single(self):
        """Test adding single example."""
        example = self.examples[0]
        self.buffer.add_example(example)
        
        assert len(self.buffer) == 1
        assert self.buffer[0] == example
    
    def test_add_example_invalid_type(self):
        """Test adding invalid type as example."""
        with pytest.raises(TypeError, match="Example must be TrainingExample instance"):
            self.buffer.add_example("not an example")
    
    def test_add_examples_multiple(self):
        """Test adding multiple examples."""
        self.buffer.add_examples(self.examples[:5])
        
        assert len(self.buffer) == 5
        for i in range(5):
            assert self.buffer[i] == self.examples[i]
    
    def test_add_examples_invalid_type(self):
        """Test adding invalid type as examples list."""
        with pytest.raises(TypeError, match="Examples must be a list"):
            self.buffer.add_examples("not a list")
    
    def test_add_examples_invalid_element_type(self):
        """Test adding list with invalid element types."""
        invalid_list = [self.examples[0], "not an example", self.examples[1]]
        
        with pytest.raises(TypeError, match="All examples must be TrainingExample instances"):
            self.buffer.add_examples(invalid_list)
    
    def test_buffer_overflow(self):
        """Test buffer behavior when exceeding max size."""
        small_buffer = ReplayBuffer(max_size=3)
        
        # Add more examples than buffer size
        small_buffer.add_examples(self.examples[:5])
        
        # Should only keep last 3 examples
        assert len(small_buffer) == 3
        assert small_buffer.is_full()
        
        # Should contain examples 2, 3, 4 (last 3 added)
        for i in range(3):
            assert small_buffer[i] == self.examples[i + 2]
    
    def test_sample_batch_basic(self):
        """Test basic batch sampling."""
        self.buffer.add_examples(self.examples)
        
        positions, policies, values = self.buffer.sample_batch(batch_size=5)
        
        # Check tensor shapes
        assert positions.shape == (5, 12, 8, 8)  # Note: transposed to (C, H, W)
        assert policies.shape == (5, 4096)
        assert values.shape == (5, 1)
        
        # Check tensor types
        assert isinstance(positions, torch.Tensor)
        assert isinstance(policies, torch.Tensor)
        assert isinstance(values, torch.Tensor)
        
        # Check data types
        assert positions.dtype == torch.float32
        assert policies.dtype == torch.float32
        assert values.dtype == torch.float32
    
    def test_sample_batch_invalid_size(self):
        """Test sampling with invalid batch size."""
        self.buffer.add_examples(self.examples[:3])
        
        # Batch size too large
        with pytest.raises(ValueError, match="Batch size 5 exceeds buffer size 3"):
            self.buffer.sample_batch(batch_size=5)
        
        # Batch size zero
        with pytest.raises(ValueError, match="Batch size must be positive"):
            self.buffer.sample_batch(batch_size=0)
        
        # Negative batch size
        with pytest.raises(ValueError, match="Batch size must be positive"):
            self.buffer.sample_batch(batch_size=-1)
    
    def test_sample_batch_empty_buffer(self):
        """Test sampling from empty buffer."""
        with pytest.raises(ValueError, match="Batch size 1 exceeds buffer size 0"):
            self.buffer.sample_batch(batch_size=1)
    
    def test_sample_batch_reproducibility(self):
        """Test that sampling is reproducible with same seed."""
        # Create buffers with same seed and reset random state
        import random
        import numpy as np
        
        # Test reproducibility by sampling twice from same buffer with reset seed
        self.buffer.add_examples(self.examples)
        
        # Set seed and sample
        random.seed(42)
        np.random.seed(42)
        pos1, pol1, val1 = self.buffer.sample_batch(batch_size=5)
        
        # Reset seed and sample again
        random.seed(42)
        np.random.seed(42)
        pos2, pol2, val2 = self.buffer.sample_batch(batch_size=5)
        
        assert torch.allclose(pos1, pos2)
        assert torch.allclose(pol1, pol2)
        assert torch.allclose(val1, val2)
    
    def test_sample_indices(self):
        """Test sampling indices."""
        self.buffer.add_examples(self.examples)
        
        indices = self.buffer.sample_indices(batch_size=5)
        
        assert len(indices) == 5
        assert all(isinstance(idx, int) for idx in indices)
        assert all(0 <= idx < len(self.buffer) for idx in indices)
        assert len(set(indices)) == 5  # All unique
    
    def test_get_examples_by_indices(self):
        """Test getting examples by indices."""
        self.buffer.add_examples(self.examples)
        
        indices = [0, 2, 4]
        retrieved = self.buffer.get_examples_by_indices(indices)
        
        assert len(retrieved) == 3
        assert retrieved[0] == self.examples[0]
        assert retrieved[1] == self.examples[2]
        assert retrieved[2] == self.examples[4]
    
    def test_get_examples_by_indices_invalid(self):
        """Test getting examples with invalid indices."""
        self.buffer.add_examples(self.examples[:3])
        
        with pytest.raises(IndexError, match="Index 5 out of range"):
            self.buffer.get_examples_by_indices([0, 1, 5])
        
        with pytest.raises(IndexError, match="Index -1 out of range"):
            self.buffer.get_examples_by_indices([-1])
    
    def test_clear(self):
        """Test clearing buffer."""
        self.buffer.add_examples(self.examples)
        assert len(self.buffer) > 0
        
        self.buffer.clear()
        assert len(self.buffer) == 0
        assert not self.buffer.is_full()
    
    def test_is_full(self):
        """Test is_full method."""
        small_buffer = ReplayBuffer(max_size=3)
        
        assert not small_buffer.is_full()
        
        small_buffer.add_examples(self.examples[:2])
        assert not small_buffer.is_full()
        
        small_buffer.add_example(self.examples[2])
        assert small_buffer.is_full()
        
        # Adding more should still be full
        small_buffer.add_example(self.examples[3])
        assert small_buffer.is_full()
    
    def test_get_memory_usage(self):
        """Test memory usage calculation."""
        # Empty buffer
        assert self.buffer.get_memory_usage() == 0
        
        # Add examples
        self.buffer.add_examples(self.examples[:5])
        memory_usage = self.buffer.get_memory_usage()
        
        assert memory_usage > 0
        # Should be approximately 5 * example_size
        expected_size = 5 * self.examples[0].get_memory_size()
        assert abs(memory_usage - expected_size) < 1000  # Allow some tolerance
    
    def test_get_statistics_empty(self):
        """Test statistics for empty buffer."""
        stats = self.buffer.get_statistics()
        
        expected = {
            'size': 0,
            'max_size': 100,
            'utilization': 0.0,
            'memory_usage_mb': 0.0,
            'avg_value': 0.0,
            'value_std': 0.0
        }
        
        assert stats == expected
    
    def test_get_statistics_with_data(self):
        """Test statistics with data."""
        self.buffer.add_examples(self.examples)
        stats = self.buffer.get_statistics()
        
        assert stats['size'] == len(self.examples)
        assert stats['max_size'] == 100
        assert stats['utilization'] == len(self.examples) / 100
        assert stats['memory_usage_mb'] > 0
        assert isinstance(stats['avg_value'], (int, float))
        assert isinstance(stats['value_std'], (int, float))
        
        # Check value statistics are reasonable
        values = [ex.value for ex in self.examples]
        assert abs(stats['avg_value'] - np.mean(values)) < 1e-6
        assert abs(stats['value_std'] - np.std(values)) < 1e-6
    
    def test_indexing(self):
        """Test buffer indexing."""
        self.buffer.add_examples(self.examples[:5])
        
        # Valid indices
        assert self.buffer[0] == self.examples[0]
        assert self.buffer[4] == self.examples[4]
        
        # Invalid indices
        with pytest.raises(IndexError):
            _ = self.buffer[5]
        
        with pytest.raises(IndexError):
            _ = self.buffer[-1]
    
    def test_iteration(self):
        """Test buffer iteration."""
        self.buffer.add_examples(self.examples[:5])
        
        iterated_examples = list(self.buffer)
        
        assert len(iterated_examples) == 5
        for i, example in enumerate(iterated_examples):
            assert example == self.examples[i]
    
    def test_len(self):
        """Test len() function."""
        assert len(self.buffer) == 0
        
        self.buffer.add_examples(self.examples[:3])
        assert len(self.buffer) == 3
        
        self.buffer.add_examples(self.examples[3:7])
        assert len(self.buffer) == 7
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.buffer)
        assert "ReplayBuffer" in repr_str
        assert "size=0" in repr_str
        assert "max_size=100" in repr_str
        assert "utilization=0.0%" in repr_str
        
        self.buffer.add_examples(self.examples[:10])
        repr_str = repr(self.buffer)
        assert "size=10" in repr_str
        assert "utilization=10.0%" in repr_str
    
    def test_save_load_file(self):
        """Test saving and loading buffer to/from file."""
        # Add some data
        self.buffer.add_examples(self.examples[:5])
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            self.buffer.save_to_file(tmp_path)
            
            # Create new buffer and load
            new_buffer = ReplayBuffer(max_size=50)  # Different size initially
            new_buffer.load_from_file(tmp_path)
            
            # Should match original
            assert len(new_buffer) == len(self.buffer)
            assert new_buffer.max_size == self.buffer.max_size
            
            for i in range(len(self.buffer)):
                assert new_buffer[i] == self.buffer[i]
        
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_load_invalid_file(self):
        """Test loading from invalid file."""
        # Create invalid file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_file.write("invalid data")
            tmp_path = tmp_file.name
        
        try:
            with pytest.raises(Exception):  # Could be pickle error or ValueError
                self.buffer.load_from_file(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_position_tensor_transformation(self):
        """Test that position tensors are correctly transformed."""
        self.buffer.add_examples(self.examples[:1])
        
        positions, _, _ = self.buffer.sample_batch(batch_size=1)
        
        # Original position is (8, 8, 12), should be transformed to (12, 8, 8)
        original_position = self.examples[0].position  # (8, 8, 12)
        sampled_position = positions[0]  # (12, 8, 8)
        
        # Check transformation is correct
        expected_position = np.transpose(original_position, (2, 0, 1))
        assert torch.allclose(sampled_position, torch.from_numpy(expected_position))
    
    def test_large_buffer_performance(self):
        """Test performance with larger buffer."""
        large_buffer = ReplayBuffer(max_size=1000)
        
        # Create many examples
        large_examples = []
        for i in range(500):
            position = np.random.rand(8, 8, 12).astype(np.float32)
            policy = np.random.rand(4096).astype(np.float32)
            policy = policy / np.sum(policy)
            value = np.random.uniform(-1, 1)
            
            example = TrainingExample(
                position=position,
                policy=policy,
                value=value
            )
            large_examples.append(example)
        
        # Add all examples
        large_buffer.add_examples(large_examples)
        
        assert len(large_buffer) == 500
        
        # Sample large batch
        positions, policies, values = large_buffer.sample_batch(batch_size=100)
        
        assert positions.shape == (100, 12, 8, 8)
        assert policies.shape == (100, 4096)
        assert values.shape == (100, 1)
    
    def test_buffer_with_different_seeds(self):
        """Test that different seeds produce different samples."""
        buffer1 = ReplayBuffer(max_size=100, seed=42)
        buffer2 = ReplayBuffer(max_size=100, seed=123)
        
        # Add same examples
        buffer1.add_examples(self.examples)
        buffer2.add_examples(self.examples)
        
        # Sample should be different
        pos1, _, _ = buffer1.sample_batch(batch_size=5)
        pos2, _, _ = buffer2.sample_batch(batch_size=5)
        
        # Should not be identical (very unlikely with different seeds)
        assert not torch.allclose(pos1, pos2)