"""Policy head implementation for chess neural network."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """Policy head for move prediction.
    
    Implements a policy head that takes feature maps and outputs move probabilities.
    Uses 1x1 convolution with 2 filters followed by a fully connected layer.
    """
    
    def __init__(self, num_filters: int):
        """Initialize policy head.
        
        Args:
            num_filters: Number of input filters from the backbone network
        """
        super().__init__()
        self.conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * 8 * 8, 4096)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to generate policy logits.
        
        Args:
            x: Input feature maps of shape (batch, num_filters, 8, 8)
            
        Returns:
            Policy logits of shape (batch, 4096) representing move probabilities
        """
        # 1x1 convolution to reduce channels
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        
        # Flatten spatial dimensions
        x = x.view(x.size(0), -1)  # (batch, 2 * 8 * 8)
        
        # Fully connected layer to output move logits
        x = self.fc(x)  # (batch, 4096)
        
        return x