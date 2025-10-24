"""Value head implementation for chess neural network."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueHead(nn.Module):
    """Value head for position evaluation.
    
    Implements a value head that takes feature maps and outputs a position evaluation.
    Uses 1x1 convolution with 1 filter followed by fully connected layers.
    """
    
    def __init__(self, num_filters: int):
        """Initialize value head.
        
        Args:
            num_filters: Number of input filters from the backbone network
        """
        super().__init__()
        self.conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to generate position value.
        
        Args:
            x: Input feature maps of shape (batch, num_filters, 8, 8)
            
        Returns:
            Position value of shape (batch, 1) in range [-1, 1]
        """
        # 1x1 convolution to reduce channels
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        
        # Flatten spatial dimensions
        x = x.view(x.size(0), -1)  # (batch, 8 * 8)
        
        # First fully connected layer with ReLU
        x = self.fc1(x)  # (batch, 256)
        x = F.relu(x)
        
        # Second fully connected layer with tanh activation
        x = self.fc2(x)  # (batch, 1)
        x = torch.tanh(x)  # Bound output to [-1, 1]
        
        return x