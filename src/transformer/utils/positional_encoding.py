from typing import Literal
from numpy import swapaxes
import torch
import torch.nn as nn
class PositionalEncoder(nn.Module):
    def __init__(self, seq_len, d_model):
       self.seq_len = seq_len 
       self.d_model = d_model
       self.pe = torch.zeros(seq_len)

       positions = torch.arange(0, seq_len)
       dims = torch.arange(0,seq_len)
       # e ^ 2i/d ln(10000) 
       div_term = positions/torch.exp(2*dims/d_model * torch.log(torch.Tensor(10000)))
       self.pe[:, 1::2] = torch.cos(div_term)
       self.pe[:, 0::2] = torch.sin(div_term)
 
    def encode(self, X):
        return X + self.pe.reshape(X)