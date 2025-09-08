import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional
import einops
from .scaled_dot_product import ScaledDotProductAttention

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as described in "Attention Is All You Need".
    
    This module performs attention computation in parallel across multiple 
    representation subspaces (heads), then concatenates and linearly transforms
    the results.
    
    The multi-head attention is computed as:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, 
                 d_model: int,
                 n_heads: int = 8,
                 attention_dropout: float = 0.1, 
                 temperature: float = 1.0):
        """
        Initialize Multi-Head Attention module.
        
        Args:
            d_model (int): Model dimension (must be divisible by n_heads)
            n_heads (int): Number of attention heads. Default: 8
            attention_dropout (float): Dropout probability applied to attention weights.
                Default: 0.1
            temperature (float): Temperature parameter to scale attention scores.
                Default: 1.0
                
        Raises:
            ValueError: If d_model is not divisible by n_heads, or if parameters are invalid
        """
        super(MultiHeadAttention, self).__init__()
        
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got: {n_heads}")
        if not 0 <= attention_dropout <= 1:
            raise ValueError(f"attention_dropout must be in [0, 1], got: {attention_dropout}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got: {temperature}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.attention_dropout = attention_dropout
        self.temperature = temperature
        self.d_k = d_model // n_heads  # Dimension per head for keys/queries
        self.d_v = d_model // n_heads  # Dimension per head for values

        # Linear projection layers for Q, K, V
        self.proj_wq = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False)
        self.proj_wk = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False) 
        self.proj_wv = nn.Linear(self.d_model, self.n_heads * self.d_v, bias=False)
        
        # Output projection layer
        self.proj_wo = nn.Linear(self.n_heads * self.d_v, d_model, bias=False)

        # Dropout layer
        self.dropout = nn.Dropout(attention_dropout)

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(
            temperature=self.temperature, 
            attention_dropout=self.attention_dropout
        )
        
        logger.debug(f"Initialized MultiHeadAttention with d_model={d_model}, "
                    f"n_heads={n_heads}, d_k={self.d_k}, d_v={self.d_v}")

    def split_into_heads(self, X: torch.Tensor) -> torch.Tensor:
        """
        Split input tensor into multiple attention heads using einops.
        
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Reshaped tensor of shape (batch_size, n_heads, seq_len, d_k)
        """
        return einops.rearrange(X, "b l (h d) -> b h l d", h=self.n_heads)
    
    def concat_heads(self, X: torch.Tensor) -> torch.Tensor:
        """
        Concatenate multiple attention heads back into single tensor using einops.
        
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, n_heads, seq_len, d_v)
            
        Returns:
            torch.Tensor: Concatenated tensor of shape (batch_size, seq_len, n_heads * d_v)
        """
        return einops.rearrange(X, "b h l d -> b l (h d)")

    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Multi-Head Attention.
        
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len_q, d_model)
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len_k, d_model)
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len_v, d_model)
            mask (Optional[torch.Tensor]): Attention mask of shape 
                (batch_size, seq_len_q, seq_len_k). Default: None
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - output: Multi-head attention output of shape (batch_size, seq_len_q, d_model)
                - attention_weights: Average attention weights across heads of shape 
                  (batch_size, seq_len_q, seq_len_k)
                  
        Raises:
            ValueError: If input tensor dimensions are incompatible
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)
        
        # Validate input dimensions
        if query.size(2) != self.d_model:
            raise ValueError(f"Query last dimension must be {self.d_model}, got: {query.size(2)}")
        if key.size(2) != self.d_model:
            raise ValueError(f"Key last dimension must be {self.d_model}, got: {key.size(2)}")
        if value.size(2) != self.d_model:
            raise ValueError(f"Value last dimension must be {self.d_model}, got: {value.size(2)}")
        if seq_len_k != seq_len_v:
            raise ValueError(f"Key and Value sequence lengths must match. Got K: {seq_len_k}, V: {seq_len_v}")
        
        logger.debug(f"Processing multi-head attention with Q: {query.shape}, "
                    f"K: {key.shape}, V: {value.shape}")

        # Linear projections for Q, K, V
        Q = self.proj_wq(query)  # (batch_size, seq_len_q, n_heads * d_k)
        K = self.proj_wk(key)    # (batch_size, seq_len_k, n_heads * d_k)
        V = self.proj_wv(value)  # (batch_size, seq_len_v, n_heads * d_v)

        # Split into multiple heads
        Q = self.split_into_heads(Q)  # (batch_size, n_heads, seq_len_q, d_k)
        K = self.split_into_heads(K)  # (batch_size, n_heads, seq_len_k, d_k)
        V = self.split_into_heads(V)  # (batch_size, n_heads, seq_len_v, d_v)
        
        logger.debug(f"Split into {self.n_heads} heads with shapes Q: {Q.shape}, "
                    f"K: {K.shape}, V: {V.shape}")

        # Expand mask for multiple heads if provided
        if mask is not None:
            # mask: (batch_size, seq_len_q, seq_len_k) -> (batch_size, n_heads, seq_len_q, seq_len_k)
            mask = mask.unsqueeze(1).expand(batch_size, self.n_heads, seq_len_q, seq_len_k)

        # Apply scaled dot-product attention
        attention_output, attention_weights = self.attention(Q, K, V, mask)
        # attention_output: (batch_size, n_heads, seq_len_q, d_v)
        # attention_weights: (batch_size, n_heads, seq_len_q, seq_len_k)

        # Concatenate heads
        attention_output = self.concat_heads(attention_output)  # (batch_size, seq_len_q, n_heads * d_v)
        
        logger.debug(f"Concatenated attention output shape: {attention_output.shape}")

        # Apply dropout
        attention_output = self.dropout(attention_output)

        # Final linear projection
        output = self.proj_wo(attention_output)  # (batch_size, seq_len_q, d_model)

        # Average attention weights across heads for visualization
        avg_attention_weights = attention_weights.mean(dim=1)  # (batch_size, seq_len_q, seq_len_k)
        
        logger.debug(f"Multi-head attention computation completed. Output shape: {output.shape}")

        return output, avg_attention_weights
    
        