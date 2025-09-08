import torch 
import torch.nn as nn

from ..ecodeco.encoder_layer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, d_model, w, n_layers: int = 6):
        super(Encoder, self).__init__()
