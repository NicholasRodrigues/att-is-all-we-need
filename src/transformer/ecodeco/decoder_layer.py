from ast import Tuple
from typing import Optional
import torch
import torch.nn as nn

from src.transformer.attention.multihead import MultiHeadAttention
from src.transformer.utils.pwffn import FFNetwork

class DecoderLayer(nn.Module):
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
        self.encode_attention = MultiHeadAttention(
        d_model=d_model,
        n_heads=n_heads,
        attention_dropout=attention_dropout
        )
        self.position_feed_forward = FFNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )

    def forward(self, 
            x: torch.Tensor,
            encoder_out: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        decoder_attention_output, decoder_attention_weights = self.self_attention(x, x, x, attention_mask)
        cross_attention_output, cross_attention_weights = self.encode_attention(x, encoder_out, encoder_out, attention_mask)
        ff_output = self.position_feed_forward(x)
        
        return x, attention_weights, cross_attention_weights
