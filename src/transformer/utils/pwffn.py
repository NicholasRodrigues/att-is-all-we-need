import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FFNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) as described in "Attention Is All You Need".
    
    This module applies two linear transformations with a ReLU activation in between:
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    
    The FFN is applied to each position separately and identically. This consists
    of two linear transformations with a ReLU activation in between.
    """
    
    def __init__(self, 
                 d_model: int, 
                 d_ff: int, 
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize Position-wise Feed-Forward Network.
        
        Args:
            d_model (int): Model dimension (input and output dimension)
            d_ff (int): Hidden dimension of the feed-forward network
            dropout (float): Dropout probability applied after first linear layer.
                Default: 0.1
            activation (str): Activation function to use ('relu', 'gelu', 'swish').
                Default: 'relu'
                
        Raises:
            ValueError: If d_model or d_ff are not positive, or if dropout is not in [0, 1]
        """
        super(FFNetwork, self).__init__()
        
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got: {d_model}")
        if d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got: {d_ff}")
        if not 0 <= dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got: {dropout}")
        if activation not in ['relu', 'gelu', 'swish']:
            raise ValueError(f"activation must be one of ['relu', 'gelu', 'swish'], got: {activation}")
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_prob = dropout
        self.activation_name = activation
        
        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)
    
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'swish':
            self.activation = F.silu  # SiLU is equivalent to Swish
        
        logger.debug(f"Initialized FFNetwork with d_model={d_model}, d_ff={d_ff}, "
                    f"dropout={dropout}, activation={activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Position-wise Feed-Forward Network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
            
        Raises:
            ValueError: If input tensor has incorrect dimensions
        """
        if x.dim() != 3:
            raise ValueError(f"Input tensor must be 3-dimensional (batch, seq_len, d_model), "
                           f"got shape: {x.shape}")
        if x.size(-1) != self.d_model:
            raise ValueError(f"Input last dimension must be {self.d_model}, got: {x.size(-1)}")
        
        batch_size, seq_len, _ = x.size()
        residual = x
        x = self.linear2(self.activation(self.linear1(x)))
        x = self.dropout(x)
        
        x += residual
        
        x = self.norm(x)
       
        return x
    
    def extra_repr(self) -> str:
        """
        Extra representation for debugging and printing.
        
        Returns:
            str: String representation of module parameters
        """
        return (f'd_model={self.d_model}, d_ff={self.d_ff}, '
                f'dropout={self.dropout_prob}, activation={self.activation_name}')
