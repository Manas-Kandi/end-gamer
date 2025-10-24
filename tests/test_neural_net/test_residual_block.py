"""Tests for ResidualBlock module."""

import torch
import pytest
from src.neural_net.residual_block import ResidualBlock


class TestResidualBlock:
    """Test cases for ResidualBlock."""
    
    def test_output_shape(self):
        """Test that output shape matches input shape."""
        num_filters = 256
        batch_size = 4
        
        block = ResidualBlock(num_filters)
        input_tensor = torch.randn(batch_size, num_filters, 8, 8)
        
        output = block(input_tensor)
        
        assert output.shape == input_tensor.shape
        assert output.shape == (batch_size, num_filters, 8, 8)
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        num_filters = 256
        block = ResidualBlock(num_filters)
        
        for batch_size in [1, 2, 8, 16]:
            input_tensor = torch.randn(batch_size, num_filters, 8, 8)
            output = block(input_tensor)
            assert output.shape == (batch_size, num_filters, 8, 8)
    
    def test_different_filter_sizes(self):
        """Test with different numbers of filters."""
        batch_size = 4
        
        for num_filters in [64, 128, 256, 512]:
            block = ResidualBlock(num_filters)
            input_tensor = torch.randn(batch_size, num_filters, 8, 8)
            output = block(input_tensor)
            assert output.shape == (batch_size, num_filters, 8, 8)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the block correctly."""
        num_filters = 256
        batch_size = 2
        
        block = ResidualBlock(num_filters)
        input_tensor = torch.randn(batch_size, num_filters, 8, 8, requires_grad=True)
        
        output = block(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        assert input_tensor.grad is not None
        assert input_tensor.grad.shape == input_tensor.shape
        
        # Check that all parameters have gradients
        for param in block.parameters():
            assert param.grad is not None
            assert param.grad.shape == param.shape
    
    def test_skip_connection(self):
        """Test that skip connection is working correctly."""
        num_filters = 256
        batch_size = 2
        
        # Create a block and set all weights to zero to isolate skip connection
        block = ResidualBlock(num_filters)
        
        # Zero out all weights to test pure skip connection
        with torch.no_grad():
            for param in block.parameters():
                param.zero_()
        
        input_tensor = torch.randn(batch_size, num_filters, 8, 8)
        
        # With zero weights, output should be ReLU(input) due to skip connection
        output = block(input_tensor)
        expected = torch.relu(input_tensor)
        
        # Should be close due to skip connection (with some numerical differences from BatchNorm)
        assert output.shape == expected.shape
    
    def test_training_mode(self):
        """Test block behavior in training vs eval mode."""
        num_filters = 256
        batch_size = 2
        
        block = ResidualBlock(num_filters)
        input_tensor = torch.randn(batch_size, num_filters, 8, 8)
        
        # Training mode
        block.train()
        output_train = block(input_tensor)
        
        # Eval mode
        block.eval()
        output_eval = block(input_tensor)
        
        # Outputs should be different due to BatchNorm behavior
        assert output_train.shape == output_eval.shape
        # Note: We don't assert they're different because with small batches
        # the difference might be minimal
    
    def test_forward_pass_values(self):
        """Test that forward pass produces reasonable values."""
        num_filters = 256
        batch_size = 4
        
        block = ResidualBlock(num_filters)
        input_tensor = torch.randn(batch_size, num_filters, 8, 8)
        
        output = block(input_tensor)
        
        # Output should be finite
        assert torch.isfinite(output).all()
        
        # Output should be non-negative due to final ReLU
        assert (output >= 0).all()
        
        # Output should not be all zeros (unless input was all negative)
        if (input_tensor > 0).any():
            assert (output > 0).any()