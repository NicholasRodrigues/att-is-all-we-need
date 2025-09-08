import torch
import torch.nn as nn

from ..attention.multihead import MultiHeadAttention
from ..utils.positional_encoding import PositionalEncoder
from ..utils.pwffn import FFNetwork

class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, w):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model

        self.multi_head = MultiHeadAttention(self.d_model)
        self.positional_encoder = PositionalEncoder(len(w), d_model)
        self.ffn = FFNetwork(w, d_model)

    def forward(self, x):
        x = self.multi_head(x)
        x = self.positional_encoder(x)
        x = self.ffn(x)
        return x
