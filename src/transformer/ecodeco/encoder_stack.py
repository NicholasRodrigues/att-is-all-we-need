import torch 
import torch.nn as nn

from src.transformer.utils.positional_encoding import PositionalEncoder

from ..ecodeco.encoder_layer import EncoderLayer

class EncoderStack(nn.Module):
    def __init__(self, d_model, vocab_len, n_layers: int = 6):
        super(EncoderStack, self).__init__()
        self.n_layers = n_layers
        self.embeddings = nn.Embedding(vocab_len, d_model)
        self.positional = PositionalEncoder(d_model)

        self.layer = EncoderLayer(d_model)

    def forward(self, x):
        x = self.positional(self.embeddings(x))

        for _ in range(self.n_layers):
            x = self.layer(x)
        return x        
