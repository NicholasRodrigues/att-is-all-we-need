import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, Optional, Union
from ..utils.softmax import softmax

logger = logging.getLogger(__name__)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism as described in "Attention Is All You Need".
    
    This implementation computes attention weights using the formula:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    Where:
    - Q: Query matrix
    - K: Key matrix  
    - V: Value matrix
    - d_k: Dimension of the key vectors
    """
    
    def __init__(self, temperature: float = 1.0, attention_dropout: float = 0.1):
        """
        Initialize Scaled Dot-Product Attention module.
        
        Args:
            temperature (float): Temperature parameter to scale the attention scores.
                Higher temperature makes the attention distribution more uniform.
                Default: 1.0
            attention_dropout (float): Dropout probability applied to attention weights.
                Default: 0.1
                
        Raises:
            ValueError: If temperature <= 0 or attention_dropout < 0 or > 1
        """
        super(ScaledDotProductAttention, self).__init__()
        
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got: {temperature}")
        if not 0 <= attention_dropout <= 1:
            raise ValueError(f"Attention dropout must be in [0, 1], got: {attention_dropout}")
        
        self.temperature = temperature
        self.attention_dropout = nn.Dropout(attention_dropout)
        
        logger.debug(f"Initialized ScaledDotProductAttention with temperature={temperature}, "
                    f"dropout={attention_dropout}")
    
    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Scaled Dot-Product Attention.
        
        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, seq_len_q, d_k)
            K (torch.Tensor): Key tensor of shape (batch_size, seq_len_k, d_k)  
            V (torch.Tensor): Value tensor of shape (batch_size, seq_len_v, d_v)
            mask (Optional[torch.Tensor]): Optional attention mask of shape 
                (batch_size, seq_len_q, seq_len_k). Default: None
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - output: Attention output of shape (batch_size, seq_len_q, d_v)
                - attention_weights: Attention weights of shape (batch_size, seq_len_q, seq_len_k)
                
        Raises:
            ValueError: If input tensors have incompatible dimensions
        """
        batch_size, seq_len_q, d_k = Q.size()
        _, seq_len_k, _ = K.size()
        _, seq_len_v, d_v = V.size()
        
        # Validate input dimensions
        if K.size(2) != d_k:
            raise ValueError(f"Query and Key must have same feature dimension. "
                           f"Got Q: {d_k}, K: {K.size(2)}")
        if seq_len_k != seq_len_v:
            raise ValueError(f"Key and Value must have same sequence length. "
                           f"Got K: {seq_len_k}, V: {seq_len_v}")
        
        logger.debug(f"Processing attention with Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
        
        # Compute scaled dot-product attention scores
        # QK^T / sqrt(d_k) 
        sqrt_dk = np.sqrt(d_k)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (sqrt_dk * self.temperature)
        
        logger.debug(f"Computed attention scores with shape: {attention_scores.shape}")
        
        # Apply mask if provided (set masked positions to large negative value)
        if mask is not None:
            if mask.shape != attention_scores.shape:
                raise ValueError(f"Mask shape {mask.shape} doesn't match attention scores shape "
                               f"{attention_scores.shape}")
            attention_scores.masked_fill_(mask == 0, -1e9)
            logger.debug("Applied attention mask")
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout to attention weights
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)
        
        logger.debug(f"Attention computation completed. Output shape: {output.shape}")
        
        return output, attention_weights
        
    def extra_repr(self) -> str:
        """
        Extra representation for debugging and printing.
        
        Returns:
            str: String representation of module parameters
        """
        return f'temperature={self.temperature}, dropout={self.attention_dropout.p}'