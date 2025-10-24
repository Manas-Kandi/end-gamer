"""Residual block implementation for chess neural network."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with skip connection.
    
    Implements a residual block with two 3x3 convolutions, BatchNorm, and skip connection.
    The block follows the pattern: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> Add -> ReLU
    """
    
    def __init__(self, num_filters: int):
        """Initialize residual block.
        
        Args:
            num_filters: Number of filters for convolutions
        """
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection.
        
        Args:
            x: Input tensor of shape (batch, num_filters, 8, 8)
            
        Returns:
            Output tensor of same shape as input
        """
        residual = x
        
        # First convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second convolution block
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Add skip connection
        x += residual
        
        # Final activation
        x = F.relu(x)
        
        return x