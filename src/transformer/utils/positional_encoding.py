import math
import torch
import torch.nn as nn
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PositionalEncoder(nn.Module):
    """
    Positional Encoding module as described in "Attention Is All You Need".
    
    Since the model contains no recurrence and no convolution, in order for the 
    model to make use of the order of the sequence, we must inject some information
    about the relative or absolute position of the tokens in the sequence.
    
    The positional encodings use sine and cosine functions of different frequencies:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        """
        Initialize Positional Encoder.
        
        Args:
            d_model (int): Model dimension
            max_seq_len (int): Maximum sequence length to precompute encodings for.
                Default: 5000
            dropout (float): Dropout probability applied to positional encodings.
                Default: 0.1
                
        Raises:
            ValueError: If d_model is not positive or if dropout is not in [0, 1]
        """
        super(PositionalEncoder, self).__init__()
        
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got: {d_model}")
        if not 0 <= dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got: {dropout}")
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_seq_len, d_model)
        
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        # div_term = exp(2i * -log(10000) / d_model) for i in [0, 1, ..., d_model//2-1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)
        
        logger.debug(f"Initialized PositionalEncoder with d_model={d_model}, "
                    f"max_seq_len={max_seq_len}, dropout={dropout}")
        logger.debug(f"Positional encoding shape: {pe.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Input with added positional encoding, same shape as input
            
        Raises:
            ValueError: If input tensor has incorrect dimensions or sequence length 
                       exceeds max_seq_len
        """
        if x.dim() != 3:
            raise ValueError(f"Input tensor must be 3-dimensional (batch, seq_len, d_model), "
                           f"got shape: {x.shape}")
        
        batch_size, seq_len, model_dim = x.size()
        
        if model_dim != self.d_model:
            raise ValueError(f"Input model dimension must be {self.d_model}, got: {model_dim}")
        
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        
        logger.debug(f"Adding positional encoding to input shape: {x.shape}")
        
        # Add positional encoding (broadcasting handles batch dimension)
        # x: (batch_size, seq_len, d_model)
        # self.pe: (1, max_seq_len, d_model) -> slice to (1, seq_len, d_model)
        pe_tensor = getattr(self, 'pe')
        pe_slice = pe_tensor[:, :seq_len, :]
        x = x + pe_slice
        
        x = self.dropout(x)
        
        logger.debug(f"Positional encoding applied successfully")
        
        return x
    
    def get_positional_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Get positional encoding for a specific sequence length.
        
        Args:
            seq_len (int): Desired sequence length
            
        Returns:
            torch.Tensor: Positional encoding of shape (1, seq_len, d_model)
            
        Raises:
            ValueError: If seq_len exceeds max_seq_len
        """
        if seq_len > self.max_seq_len:
            raise ValueError(f"Requested sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        
        pe_tensor = getattr(self, 'pe')
        return pe_tensor[:, :seq_len, :]
    
    def extra_repr(self) -> str:
        """
        Extra representation for debugging and printing.
        
        Returns:
            str: String representation of module parameters
        """
        return (f'd_model={self.d_model}, max_seq_len={self.max_seq_len}, '
                f'dropout={self.dropout.p}')