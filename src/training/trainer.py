"""Trainer class for neural network training."""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

from ..config.config import Config


class Trainer:
    """Manages neural network training with optimizer and scheduler.
    
    Handles gradient updates, loss computation, and learning rate scheduling
    for the chess neural network during self-play training. Supports mixed
    precision training for improved performance on modern GPUs.
    """
    
    def __init__(self, neural_net: nn.Module, config: Config):
        """Initialize trainer with neural network and configuration.
        
        Args:
            neural_net: Neural network to train
            config: Training configuration
        """
        self.neural_net = neural_net
        self.config = config
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            neural_net.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize learning rate scheduler
        self.scheduler = StepLR(
            self.optimizer,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma
        )
        
        # Initialize mixed precision training
        self.use_amp = config.mixed_precision and config.device == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Training step counter
        self.step_count = 0
        
        # Move network to device
        self.neural_net.to(config.device)
    
    def train_step(self, positions: torch.Tensor, target_policies: torch.Tensor,
                   target_values: torch.Tensor) -> Dict[str, float]:
        """Perform one training step with gradient update.
        
        Supports automatic mixed precision (AMP) training for improved performance
        on modern GPUs with minimal accuracy impact.
        
        Args:
            positions: Batch of board positions (batch_size, 12, 8, 8)
            target_policies: Target policy distributions (batch_size, 4096)
            target_values: Target position values (batch_size, 1)
            
        Returns:
            Dictionary with loss metrics
        """
        # Set network to training mode
        self.neural_net.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Move tensors to device
        positions = positions.to(self.config.device)
        target_policies = target_policies.to(self.config.device)
        target_values = target_values.to(self.config.device)
        
        if self.use_amp:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                # Forward pass
                pred_policies, pred_values = self.neural_net(positions)
                
                # Compute losses
                policy_loss = F.cross_entropy(pred_policies, target_policies)
                value_loss = F.mse_loss(pred_values, target_values)
                total_loss = policy_loss + value_loss
            
            # Backward pass with gradient scaling
            self.scaler.scale(total_loss).backward()
            
            # Gradient clipping (unscale first)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.neural_net.parameters(), 
                max_norm=self.config.gradient_clip_norm
            )
            
            # Update parameters
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard precision training
            # Forward pass
            pred_policies, pred_values = self.neural_net(positions)
            
            # Compute losses
            policy_loss = F.cross_entropy(pred_policies, target_policies)
            value_loss = F.mse_loss(pred_values, target_values)
            total_loss = policy_loss + value_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.neural_net.parameters(), 
                max_norm=self.config.gradient_clip_norm
            )
            
            # Update parameters
            self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        # Increment step counter
        self.step_count += 1
        
        # Return loss metrics
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def get_state_dict(self) -> Dict:
        """Get trainer state for checkpointing.
        
        Returns:
            Dictionary with optimizer, scheduler, and scaler state
        """
        state_dict = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step_count': self.step_count
        }
        
        # Include scaler state if using mixed precision
        if self.use_amp and self.scaler is not None:
            state_dict['scaler_state_dict'] = self.scaler.state_dict()
        
        return state_dict
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load trainer state from checkpoint.
        
        Args:
            state_dict: Dictionary with optimizer, scheduler, and scaler state
        """
        if 'optimizer_state_dict' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        
        if 'step_count' in state_dict:
            self.step_count = state_dict['step_count']
        
        # Load scaler state if using mixed precision
        if self.use_amp and self.scaler is not None and 'scaler_state_dict' in state_dict:
            self.scaler.load_state_dict(state_dict['scaler_state_dict'])
    
    def set_learning_rate(self, lr: float) -> None:
        """Set learning rate for optimizer.
        
        Args:
            lr: New learning rate
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_learning_rate(self) -> float:
        """Get current learning rate.
        
        Returns:
            Current learning rate
        """
        return self.optimizer.param_groups[0]['lr']
    
    def eval_mode(self) -> None:
        """Set network to evaluation mode."""
        self.neural_net.eval()
    
    def train_mode(self) -> None:
        """Set network to training mode."""
        self.neural_net.train()