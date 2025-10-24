"""Unit tests for Trainer class."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from src.training.trainer import Trainer
from src.config.config import Config


class SimpleTestNet(nn.Module):
    """Simple neural network for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(12, 16, 3, padding=1)
        self.policy_fc = nn.Linear(16 * 8 * 8, 4096)
        self.value_fc = nn.Linear(16 * 8 * 8, 1)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.relu(self.conv(x))
        x = x.view(batch_size, -1)
        
        policy = self.policy_fc(x)
        value = torch.tanh(self.value_fc(x))
        
        return policy, value


class TestTrainer:
    """Test cases for Trainer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config(
            learning_rate=0.001,
            weight_decay=1e-4,
            gradient_clip_norm=1.0,
            lr_step_size=1000,
            lr_gamma=0.1,
            device='cpu'
        )
        
        self.neural_net = SimpleTestNet()
        self.trainer = Trainer(self.neural_net, self.config)
        
        # Create sample training data
        self.batch_size = 4
        self.positions = torch.randn(self.batch_size, 12, 8, 8)
        
        # Create valid policy targets (probability distributions)
        policy_logits = torch.randn(self.batch_size, 4096)
        self.target_policies = torch.softmax(policy_logits, dim=1)
        
        self.target_values = torch.randn(self.batch_size, 1)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.neural_net == self.neural_net
        assert self.trainer.config == self.config
        assert self.trainer.step_count == 0
        
        # Check optimizer configuration
        assert isinstance(self.trainer.optimizer, torch.optim.Adam)
        assert self.trainer.optimizer.param_groups[0]['lr'] == 0.001
        assert self.trainer.optimizer.param_groups[0]['weight_decay'] == 1e-4
        
        # Check scheduler configuration
        assert isinstance(self.trainer.scheduler, torch.optim.lr_scheduler.StepLR)
    
    def test_train_step_basic(self):
        """Test basic training step functionality."""
        # Get initial parameters for comparison
        initial_params = [p.clone() for p in self.neural_net.parameters()]
        
        # Perform training step
        losses = self.trainer.train_step(
            self.positions, self.target_policies, self.target_values
        )
        
        # Check that losses are returned
        assert isinstance(losses, dict)
        assert 'total_loss' in losses
        assert 'policy_loss' in losses
        assert 'value_loss' in losses
        assert 'learning_rate' in losses
        
        # Check that losses are reasonable
        assert losses['total_loss'] >= 0
        assert losses['policy_loss'] >= 0
        assert losses['value_loss'] >= 0
        assert losses['learning_rate'] > 0
        
        # Check that parameters were updated
        updated_params = list(self.neural_net.parameters())
        for initial, updated in zip(initial_params, updated_params):
            assert not torch.allclose(initial, updated, atol=1e-6)
        
        # Check step count incremented
        assert self.trainer.step_count == 1
    
    def test_train_step_loss_computation(self):
        """Test that losses are computed correctly."""
        losses = self.trainer.train_step(
            self.positions, self.target_policies, self.target_values
        )
        
        # Total loss should be sum of policy and value losses
        # Note: This is approximate due to potential numerical differences
        expected_total = losses['policy_loss'] + losses['value_loss']
        assert abs(losses['total_loss'] - expected_total) < 1e-5
    
    def test_train_step_gradient_clipping(self):
        """Test gradient clipping functionality."""
        # Create a scenario that might produce large gradients
        large_targets = torch.ones(self.batch_size, 1) * 100  # Large values
        
        # Mock gradient clipping to verify it's called
        with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            self.trainer.train_step(
                self.positions, self.target_policies, large_targets
            )
            
            # Verify gradient clipping was called
            mock_clip.assert_called_once()
            args, kwargs = mock_clip.call_args
            assert kwargs['max_norm'] == self.config.gradient_clip_norm
    
    def test_train_step_device_handling(self):
        """Test that tensors are moved to correct device."""
        # Test with CPU device (default in config)
        losses = self.trainer.train_step(
            self.positions, self.target_policies, self.target_values
        )
        
        # Should complete without errors
        assert isinstance(losses, dict)
    
    def test_multiple_train_steps(self):
        """Test multiple consecutive training steps."""
        losses_list = []
        
        for i in range(5):
            losses = self.trainer.train_step(
                self.positions, self.target_policies, self.target_values
            )
            losses_list.append(losses)
            
            # Check step count
            assert self.trainer.step_count == i + 1
        
        # All steps should complete successfully
        assert len(losses_list) == 5
        
        # Learning rate should remain constant (no scheduler step in this range)
        for losses in losses_list:
            assert losses['learning_rate'] == 0.001
    
    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling."""
        initial_lr = self.trainer.get_learning_rate()
        assert initial_lr == 0.001
        
        # Perform steps up to scheduler step size
        for _ in range(self.config.lr_step_size):
            self.trainer.train_step(
                self.positions, self.target_policies, self.target_values
            )
        
        # Learning rate should have decreased
        new_lr = self.trainer.get_learning_rate()
        expected_lr = initial_lr * self.config.lr_gamma
        assert abs(new_lr - expected_lr) < 1e-6
    
    def test_get_state_dict(self):
        """Test getting trainer state for checkpointing."""
        # Perform some training steps
        for _ in range(3):
            self.trainer.train_step(
                self.positions, self.target_policies, self.target_values
            )
        
        state_dict = self.trainer.get_state_dict()
        
        # Check state dict contents
        assert isinstance(state_dict, dict)
        assert 'optimizer_state_dict' in state_dict
        assert 'scheduler_state_dict' in state_dict
        assert 'step_count' in state_dict
        
        assert state_dict['step_count'] == 3
    
    def test_load_state_dict(self):
        """Test loading trainer state from checkpoint."""
        # Perform some training steps
        for _ in range(5):
            self.trainer.train_step(
                self.positions, self.target_policies, self.target_values
            )
        
        # Save state
        saved_state = self.trainer.get_state_dict()
        
        # Create new trainer
        new_trainer = Trainer(SimpleTestNet(), self.config)
        
        # Load state
        new_trainer.load_state_dict(saved_state)
        
        # Check that state was loaded correctly
        assert new_trainer.step_count == 5
        
        # Optimizer states should match
        old_lr = self.trainer.get_learning_rate()
        new_lr = new_trainer.get_learning_rate()
        assert abs(old_lr - new_lr) < 1e-6
    
    def test_set_learning_rate(self):
        """Test setting learning rate manually."""
        new_lr = 0.005
        self.trainer.set_learning_rate(new_lr)
        
        assert self.trainer.get_learning_rate() == new_lr
        
        # Verify it affects training
        losses = self.trainer.train_step(
            self.positions, self.target_policies, self.target_values
        )
        assert losses['learning_rate'] == new_lr
    
    def test_get_learning_rate(self):
        """Test getting current learning rate."""
        lr = self.trainer.get_learning_rate()
        assert lr == self.config.learning_rate
        
        # Change learning rate and test again
        new_lr = 0.002
        self.trainer.set_learning_rate(new_lr)
        assert self.trainer.get_learning_rate() == new_lr
    
    def test_eval_mode(self):
        """Test setting network to evaluation mode."""
        # Initially in training mode
        assert self.neural_net.training
        
        self.trainer.eval_mode()
        assert not self.neural_net.training
    
    def test_train_mode(self):
        """Test setting network to training mode."""
        # Set to eval mode first
        self.trainer.eval_mode()
        assert not self.neural_net.training
        
        # Set back to training mode
        self.trainer.train_mode()
        assert self.neural_net.training
    
    def test_train_step_with_different_batch_sizes(self):
        """Test training step with different batch sizes."""
        batch_sizes = [1, 2, 8, 16]
        
        for batch_size in batch_sizes:
            positions = torch.randn(batch_size, 12, 8, 8)
            policy_logits = torch.randn(batch_size, 4096)
            target_policies = torch.softmax(policy_logits, dim=1)
            target_values = torch.randn(batch_size, 1)
            
            losses = self.trainer.train_step(positions, target_policies, target_values)
            
            # Should work for all batch sizes
            assert isinstance(losses, dict)
            assert losses['total_loss'] >= 0
    
    def test_train_step_with_zero_gradients(self):
        """Test training step when gradients might be zero."""
        # Create targets that match current predictions exactly
        self.neural_net.eval()
        with torch.no_grad():
            pred_policies, pred_values = self.neural_net(self.positions)
        
        # Use predictions as targets (should give very small loss)
        target_policies = torch.softmax(pred_policies, dim=1)
        
        losses = self.trainer.train_step(
            self.positions, target_policies, pred_values
        )
        
        # Should complete without errors even with small gradients
        assert isinstance(losses, dict)
        assert losses['total_loss'] >= 0
    
    def test_trainer_with_different_configs(self):
        """Test trainer with different configuration parameters."""
        # Test with different learning rate
        config_high_lr = Config(
            learning_rate=0.01,
            weight_decay=1e-3,
            gradient_clip_norm=0.5,
            device='cpu'
        )
        
        trainer_high_lr = Trainer(SimpleTestNet(), config_high_lr)
        
        assert trainer_high_lr.get_learning_rate() == 0.01
        
        # Training step should work
        losses = trainer_high_lr.train_step(
            self.positions, self.target_policies, self.target_values
        )
        assert isinstance(losses, dict)
    
    def test_network_mode_during_training(self):
        """Test that network is in training mode during train_step."""
        # Set to eval mode initially
        self.trainer.eval_mode()
        assert not self.neural_net.training
        
        # Training step should set to training mode
        self.trainer.train_step(
            self.positions, self.target_policies, self.target_values
        )
        
        # Should be in training mode after train_step
        assert self.neural_net.training
    
    def test_parameter_updates_accumulate(self):
        """Test that parameter updates accumulate over multiple steps."""
        # Get initial parameters
        initial_params = [p.clone() for p in self.neural_net.parameters()]
        
        # Perform multiple training steps
        for _ in range(3):
            self.trainer.train_step(
                self.positions, self.target_policies, self.target_values
            )
        
        # Parameters should be different from initial
        final_params = list(self.neural_net.parameters())
        for initial, final in zip(initial_params, final_params):
            assert not torch.allclose(initial, final, atol=1e-5)
    
    def test_loss_values_are_finite(self):
        """Test that loss values are always finite."""
        # Test with various input ranges
        test_cases = [
            (torch.randn(2, 12, 8, 8), "normal"),
            (torch.randn(2, 12, 8, 8) * 10, "large"),
            (torch.randn(2, 12, 8, 8) * 0.1, "small"),
        ]
        
        for positions, case_name in test_cases:
            policy_logits = torch.randn(2, 4096)
            target_policies = torch.softmax(policy_logits, dim=1)
            target_values = torch.randn(2, 1)
            
            losses = self.trainer.train_step(positions, target_policies, target_values)
            
            # All losses should be finite
            for loss_name, loss_value in losses.items():
                if loss_name != 'learning_rate':  # Skip learning rate check
                    assert np.isfinite(loss_value), f"{loss_name} not finite in {case_name} case"
                    assert loss_value >= 0, f"{loss_name} negative in {case_name} case"