"""Tests for PolicyHead module."""

import torch
import pytest
from src.neural_net.policy_head import PolicyHead


class TestPolicyHead:
    """Test cases for PolicyHead."""
    
    def test_output_shape(self):
        """Test that output shape is correct for policy logits."""
        num_filters = 256
        batch_size = 4
        
        head = PolicyHead(num_filters)
        input_tensor = torch.randn(batch_size, num_filters, 8, 8)
        
        output = head(input_tensor)
        
        assert output.shape == (batch_size, 4096)
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        num_filters = 256
        head = PolicyHead(num_filters)
        
        for batch_size in [1, 2, 8, 16, 32]:
            input_tensor = torch.randn(batch_size, num_filters, 8, 8)
            output = head(input_tensor)
            assert output.shape == (batch_size, 4096)
    
    def test_different_filter_sizes(self):
        """Test with different numbers of input filters."""
        batch_size = 4
        
        for num_filters in [64, 128, 256, 512]:
            head = PolicyHead(num_filters)
            input_tensor = torch.randn(batch_size, num_filters, 8, 8)
            output = head(input_tensor)
            assert output.shape == (batch_size, 4096)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the head correctly."""
        num_filters = 256
        batch_size = 2
        
        head = PolicyHead(num_filters)
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
        
        head = PolicyHead(num_filters)
        input_tensor = torch.randn(batch_size, num_filters, 8, 8)
        
        output = head(input_tensor)
        
        # Output should be finite
        assert torch.isfinite(output).all()
        
        # Output should have the right range (logits can be any real number)
        assert output.dtype == torch.float32
    
    def test_training_vs_eval_mode(self):
        """Test head behavior in training vs eval mode."""
        num_filters = 256
        batch_size = 2
        
        head = PolicyHead(num_filters)
        input_tensor = torch.randn(batch_size, num_filters, 8, 8)
        
        # Training mode
        head.train()
        output_train = head(input_tensor)
        
        # Eval mode
        head.eval()
        output_eval = head(input_tensor)
        
        # Outputs should have same shape
        assert output_train.shape == output_eval.shape
        assert output_train.shape == (batch_size, 4096)
    
    def test_policy_logits_range(self):
        """Test that policy logits are in reasonable range."""
        num_filters = 256
        batch_size = 4
        
        head = PolicyHead(num_filters)
        input_tensor = torch.randn(batch_size, num_filters, 8, 8)
        
        output = head(input_tensor)
        
        # Logits should be finite
        assert torch.isfinite(output).all()
        
        # After softmax, should sum to 1 (approximately)
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-6)
        
        # All probabilities should be non-negative
        assert (probs >= 0).all()
        assert (probs <= 1).all()
    
    def test_deterministic_output(self):
        """Test that same input produces same output."""
        num_filters = 256
        batch_size = 2
        
        head = PolicyHead(num_filters)
        head.eval()  # Set to eval mode for deterministic behavior
        
        input_tensor = torch.randn(batch_size, num_filters, 8, 8)
        
        output1 = head(input_tensor)
        output2 = head(input_tensor)
        
        # Should be identical in eval mode
        assert torch.allclose(output1, output2)