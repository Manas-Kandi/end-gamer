"""Tests for ChessNet main network."""

import torch
import pytest
from src.neural_net.chess_net import ChessNet


class TestChessNet:
    """Test cases for ChessNet."""
    
    def test_output_shapes(self):
        """Test that output shapes are correct."""
        batch_size = 4
        
        net = ChessNet(num_res_blocks=3, num_filters=256)
        input_tensor = torch.randn(batch_size, 12, 8, 8)
        
        policy, value = net(input_tensor)
        
        assert policy.shape == (batch_size, 4096)
        assert value.shape == (batch_size, 1)
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        net = ChessNet(num_res_blocks=3, num_filters=256)
        
        for batch_size in [1, 2, 8, 16, 32]:
            input_tensor = torch.randn(batch_size, 12, 8, 8)
            policy, value = net(input_tensor)
            
            assert policy.shape == (batch_size, 4096)
            assert value.shape == (batch_size, 1)
    
    def test_different_architectures(self):
        """Test with different network architectures."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, 12, 8, 8)
        
        # Test different numbers of residual blocks
        for num_blocks in [1, 2, 3, 5]:
            net = ChessNet(num_res_blocks=num_blocks, num_filters=256)
            policy, value = net(input_tensor)
            
            assert policy.shape == (batch_size, 4096)
            assert value.shape == (batch_size, 1)
        
        # Test different numbers of filters
        for num_filters in [64, 128, 256, 512]:
            net = ChessNet(num_res_blocks=3, num_filters=num_filters)
            policy, value = net(input_tensor)
            
            assert policy.shape == (batch_size, 4096)
            assert value.shape == (batch_size, 1)
    
    def test_value_range(self):
        """Test that value output is in correct range [-1, 1]."""
        batch_size = 10
        
        net = ChessNet(num_res_blocks=3, num_filters=256)
        
        # Test with various input ranges
        for _ in range(5):
            input_tensor = torch.randn(batch_size, 12, 8, 8) * 10
            policy, value = net(input_tensor)
            
            # Value should be bounded by tanh
            assert (value >= -1.0).all()
            assert (value <= 1.0).all()
    
    def test_gradient_flow(self):
        """Test that gradients flow through the entire network."""
        batch_size = 2
        
        net = ChessNet(num_res_blocks=3, num_filters=256)
        input_tensor = torch.randn(batch_size, 12, 8, 8, requires_grad=True)
        
        policy, value = net(input_tensor)
        loss = policy.sum() + value.sum()
        loss.backward()
        
        # Check that input has gradients
        assert input_tensor.grad is not None
        assert input_tensor.grad.shape == input_tensor.shape
        
        # Check that all parameters have gradients
        for param in net.parameters():
            assert param.grad is not None
            assert param.grad.shape == param.shape
    
    def test_parameter_count(self):
        """Test that parameter count is approximately 4M."""
        net = ChessNet(num_res_blocks=3, num_filters=256)
        param_count = net.count_parameters()
        
        # Should be approximately 4M parameters (within reasonable range)
        assert 3_500_000 <= param_count <= 5_000_000
        print(f"Parameter count: {param_count:,}")
    
    def test_model_size(self):
        """Test that model size is approximately 16MB."""
        net = ChessNet(num_res_blocks=3, num_filters=256)
        size_mb = net.get_model_size_mb()
        
        # Should be approximately 16MB (within reasonable range)
        assert 12.0 <= size_mb <= 20.0
        print(f"Model size: {size_mb:.2f} MB")
    
    def test_forward_pass_values(self):
        """Test that forward pass produces reasonable values."""
        batch_size = 4
        
        net = ChessNet(num_res_blocks=3, num_filters=256)
        input_tensor = torch.randn(batch_size, 12, 8, 8)
        
        policy, value = net(input_tensor)
        
        # All outputs should be finite
        assert torch.isfinite(policy).all()
        assert torch.isfinite(value).all()
        
        # Value should be in correct range
        assert (value >= -1.0).all()
        assert (value <= 1.0).all()
        
        # Policy logits can be any real number, but should be finite
        assert torch.isfinite(policy).all()
        
        # After softmax, policy should sum to 1
        probs = torch.softmax(policy, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-6)
    
    def test_training_vs_eval_mode(self):
        """Test network behavior in training vs eval mode."""
        batch_size = 2
        
        net = ChessNet(num_res_blocks=3, num_filters=256)
        input_tensor = torch.randn(batch_size, 12, 8, 8)
        
        # Training mode
        net.train()
        policy_train, value_train = net(input_tensor)
        
        # Eval mode
        net.eval()
        policy_eval, value_eval = net(input_tensor)
        
        # Outputs should have same shape
        assert policy_train.shape == policy_eval.shape
        assert value_train.shape == value_eval.shape
        
        # Values should still be in correct range
        assert (value_train >= -1.0).all() and (value_train <= 1.0).all()
        assert (value_eval >= -1.0).all() and (value_eval <= 1.0).all()
    
    def test_deterministic_output(self):
        """Test that same input produces same output in eval mode."""
        batch_size = 2
        
        net = ChessNet(num_res_blocks=3, num_filters=256)
        net.eval()
        
        input_tensor = torch.randn(batch_size, 12, 8, 8)
        
        policy1, value1 = net(input_tensor)
        policy2, value2 = net(input_tensor)
        
        # Should be identical in eval mode
        assert torch.allclose(policy1, policy2)
        assert torch.allclose(value1, value2)
    
    def test_different_input_patterns(self):
        """Test network with different input patterns."""
        batch_size = 4
        
        net = ChessNet(num_res_blocks=3, num_filters=256)
        
        # Test with zeros
        zero_input = torch.zeros(batch_size, 12, 8, 8)
        policy_zero, value_zero = net(zero_input)
        assert policy_zero.shape == (batch_size, 4096)
        assert value_zero.shape == (batch_size, 1)
        assert (value_zero >= -1.0).all() and (value_zero <= 1.0).all()
        
        # Test with ones
        ones_input = torch.ones(batch_size, 12, 8, 8)
        policy_ones, value_ones = net(ones_input)
        assert policy_ones.shape == (batch_size, 4096)
        assert value_ones.shape == (batch_size, 1)
        assert (value_ones >= -1.0).all() and (value_ones <= 1.0).all()
        
        # Test with random binary pattern (more chess-like)
        binary_input = torch.randint(0, 2, (batch_size, 12, 8, 8)).float()
        policy_binary, value_binary = net(binary_input)
        assert policy_binary.shape == (batch_size, 4096)
        assert value_binary.shape == (batch_size, 1)
        assert (value_binary >= -1.0).all() and (value_binary <= 1.0).all()
    
    def test_save_load_model(self):
        """Test that model can be saved and loaded correctly."""
        batch_size = 2
        
        # Create and test original model
        net1 = ChessNet(num_res_blocks=3, num_filters=256)
        net1.eval()  # Set to eval mode before testing
        input_tensor = torch.randn(batch_size, 12, 8, 8)
        
        # Save model state
        state_dict = net1.state_dict()
        
        # Create new model and load state
        net2 = ChessNet(num_res_blocks=3, num_filters=256)
        net2.load_state_dict(state_dict)
        net2.eval()  # Set to eval mode
        
        # Test that outputs are identical
        policy1, value1 = net1(input_tensor)
        policy2, value2 = net2(input_tensor)
        
        assert torch.allclose(policy1, policy2)
        assert torch.allclose(value1, value2)
    
    def test_small_network(self):
        """Test with smaller network configuration."""
        batch_size = 4
        
        # Small network for faster testing
        net = ChessNet(num_res_blocks=1, num_filters=64)
        input_tensor = torch.randn(batch_size, 12, 8, 8)
        
        policy, value = net(input_tensor)
        
        assert policy.shape == (batch_size, 4096)
        assert value.shape == (batch_size, 1)
        assert (value >= -1.0).all() and (value <= 1.0).all()
        
        # Should have fewer parameters
        param_count = net.count_parameters()
        assert param_count < 1_000_000  # Much smaller than full network