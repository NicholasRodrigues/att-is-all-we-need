from ast import Tuple
from typing import Optional
import torch
import torch.nn as nn

from src.transformer.attention.multihead import MultiHeadAttention
from src.transformer.utils.pwffn import FFNetwork

class DecoderLayer(nn.Modules):
    def __init__(self,
                d_model: int,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 activation: str = 'relu',
                 norm_first: bool = True):
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.norm_first = norm_first
        
        self.self_attention = MultiHeadAttention(
        d_model=d_model,
        n_heads=n_heads,
        attention_dropout=attention_dropout
        )
        
        self.feed_forward = FFNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, 
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        res1 = x 
        if self.norm_first:
            x_norm = self.norm1(x)
            attention_output, attention_weights = self.self_attention(x_norm, x_norm, x_norm, attention_mask)
            x = x + self.dropout1(attention_output) + res1
        else:
            attention_output, attention_weights = self.self_attention(x, x, x, attention_mask)
            x = self.norm1(x + self.dropout1(attention_output)) + res1            
        
        res2 = x        
        if self.norm_first:
            x_norm = self.norm2(x)
            attention_output, attention_weights = self.self_attention(x_norm, x_norm, x_norm, attention_mask)
            x = x + self.dropout2(attention_output) + res2
        else:
            attention_output, attention_weights = self.self_attention(x_norm, x_norm, x_norm, attention_mask)
            x = x + self.dropout2(attention_output) + res2

        res3 = x 
        if self.norm_first:
            x_norm = self.norm3(x)
            ff_output = self.feed_forward(x_norm)
            x = x + self.dropout3(ff_output) + res3
        else:
            ff_output = self.feed_forward(x)
            x = self.norm3(x + self.dropout3(ff_output)) + res3
        
        return x, attention_weights
