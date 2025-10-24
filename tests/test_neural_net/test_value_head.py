"""Tests for ValueHead module."""

import torch
import pytest
from src.neural_net.value_head import ValueHead


class TestValueHead:
    """Test cases for ValueHead."""
    
    def test_output_shape(self):
        """Test that output shape is correct for value prediction."""
        num_filters = 256
        batch_size = 4
        
        head = ValueHead(num_filters)
        input_tensor = torch.randn(batch_size, num_filters, 8, 8)
        
        output = head(input_tensor)
        
        assert output.shape == (batch_size, 1)
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        num_filters = 256
        head = ValueHead(num_filters)
        
        for batch_size in [1, 2, 8, 16, 32]:
            input_tensor = torch.randn(batch_size, num_filters, 8, 8)
            output = head(input_tensor)
            assert output.shape == (batch_size, 1)
    
    def test_different_filter_sizes(self):
        """Test with different numbers of input filters."""
        batch_size = 4
        
        for num_filters in [64, 128, 256, 512]:
            head = ValueHead(num_filters)
            input_tensor = torch.randn(batch_size, num_filters, 8, 8)
            output = head(input_tensor)
            assert output.shape == (batch_size, 1)
    
    def test_output_range(self):
        """Test that output is bounded to [-1, 1] range."""
        num_filters = 256
        batch_size = 10
        
        head = ValueHead(num_filters)
        
        # Test with various input ranges
        for _ in range(10):
            input_tensor = torch.randn(batch_size, num_filters, 8, 8) * 10  # Large values
            output = head(input_tensor)
            
            # All outputs should be in [-1, 1] due to tanh
            assert (output >= -1.0).all()
            assert (output <= 1.0).all()
    
    def test_gradient_flow(self):
        """Test that gradients flow through the head correctly."""
        num_filters = 256
        batch_size = 2
        
        head = ValueHead(num_filters)
        input_tensor = torch.randn(batch_size, num_filters, 8, 8, requires_grad=True)
        
        output = head(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        assert input_tensor.grad is not None
        assert input_tensor.grad.shape == input_tensor.shape
        
        # Check that all parameters have gradients
        for param in head.parameters():
            assert param.grad is not None
            assert param.grad.shape == param.shape
    
    def test_forward_pass_values(self):
        """Test that forward pass produces reasonable values."""
        num_filters = 256
        batch_size = 4
        
        head = ValueHead(num_filters)
        input_tensor = torch.randn(batch_size, num_filters, 8, 8)
        
        output = head(input_tensor)
        
        # Output should be finite
        assert torch.isfinite(output).all()
        
        # Output should be in correct range
        assert (output >= -1.0).all()
        assert (output <= 1.0).all()
        
        # Output should have the right dtype
        assert output.dtype == torch.float32
    
    def test_training_vs_eval_mode(self):
        """Test head behavior in training vs eval mode."""
        num_filters = 256
        batch_size = 2
        
        head = ValueHead(num_filters)
        input_tensor = torch.randn(batch_size, num_filters, 8, 8)
        
        # Training mode
        head.train()
        output_train = head(input_tensor)
        
        # Eval mode
        head.eval()
        output_eval = head(input_tensor)
        
        # Outputs should have same shape and range
        assert output_train.shape == output_eval.shape
        assert output_train.shape == (batch_size, 1)
        assert (output_train >= -1.0).all() and (output_train <= 1.0).all()
        assert (output_eval >= -1.0).all() and (output_eval <= 1.0).all()
    
    def test_extreme_inputs(self):
        """Test head with extreme input values."""
        num_filters = 256
        batch_size = 4
        
        head = ValueHead(num_filters)
        
        # Test with very large positive values
        large_input = torch.ones(batch_size, num_filters, 8, 8) * 100
        output_large = head(large_input)
        assert (output_large >= -1.0).all() and (output_large <= 1.0).all()
        
        # Test with very large negative values
        small_input = torch.ones(batch_size, num_filters, 8, 8) * -100
        output_small = head(small_input)
        assert (output_small >= -1.0).all() and (output_small <= 1.0).all()
        
        # Test with zeros
        zero_input = torch.zeros(batch_size, num_filters, 8, 8)
        output_zero = head(zero_input)
        assert (output_zero >= -1.0).all() and (output_zero <= 1.0).all()
    
    def test_deterministic_output(self):
        """Test that same input produces same output."""
        num_filters = 256
        batch_size = 2
        
        head = ValueHead(num_filters)
        head.eval()  # Set to eval mode for deterministic behavior
        
        input_tensor = torch.randn(batch_size, num_filters, 8, 8)
        
        output1 = head(input_tensor)
        output2 = head(input_tensor)
        
        # Should be identical in eval mode
        assert torch.allclose(output1, output2)
    
    def test_tanh_saturation(self):
        """Test that tanh activation properly saturates at extremes."""
        num_filters = 256
        batch_size = 4
        
        head = ValueHead(num_filters)
        
        # Create inputs that should lead to saturation
        # We'll modify the final layer weights to create extreme values before tanh
        with torch.no_grad():
            # Set fc2 weights to large values to test saturation
            head.fc2.weight.fill_(10.0)
            head.fc2.bias.fill_(0.0)
        
        input_tensor = torch.randn(batch_size, num_filters, 8, 8)
        output = head(input_tensor)
        
        # Even with extreme weights, output should still be bounded
        assert (output >= -1.0).all()
        assert (output <= 1.0).all()
        
        # Some outputs should be close to saturation values
        assert (torch.abs(output) > 0.9).any()  # At least some should be near ±1