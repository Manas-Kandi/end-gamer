"""Main ChessNet neural network implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .residual_block import ResidualBlock
from .policy_head import PolicyHead
from .value_head import ValueHead


class ChessNet(nn.Module):
    """AlphaZero-style neural network for chess endgames.
    
    Implements a convolutional neural network with residual blocks for chess position
    evaluation and move prediction. The network takes an 8x8x12 board representation
    and outputs both policy (move probabilities) and value (position evaluation).
    """
    
    def __init__(self, num_res_blocks: int = 3, num_filters: int = 256):
        """Initialize ChessNet.
        
        Args:
            num_res_blocks: Number of residual blocks in the backbone
            num_filters: Number of filters in convolutional layers
        """
        super().__init__()
        
        # Input convolution layer (12 -> num_filters, 3x3, BatchNorm, ReLU)
        self.input_conv = nn.Sequential(
            nn.Conv2d(12, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Policy and value heads
        self.policy_head = PolicyHead(num_filters)
        self.value_head = ValueHead(num_filters)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            x: Input board tensor of shape (batch, 12, 8, 8)
            
        Returns:
            Tuple of (policy_logits, value):
                - policy_logits: (batch, 4096) move probability logits
                - value: (batch, 1) position evaluation in range [-1, 1]
        """
        # Input convolution
        x = self.input_conv(x)
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Dual heads
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
    
    def count_parameters(self) -> int:
        """Count the total number of trainable parameters.
        
        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get approximate model size in MB.
        
        Returns:
            Model size in megabytes
        """
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb