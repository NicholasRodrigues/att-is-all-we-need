import torch
import torch.nn as nn 
import numpy as np
from src.transformer.utils.positional_encoding import PositionalEncoder
from transformer.ecodeco.encoder_stack import EncoderStack
from transformer.ecodeco.decoder_stack import DecoderStack


class Encoder(nn.Module):
  def __init__(self, n_vocab, d_vocab_vec, n_layers, n_head, dk, dv, d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):
    super().__init__()
    self.src_vocab_embedding = nn.Embedding(n_vocab, d_vocab_vec, padding_idx=pad_idx)
    self.position_encoder = PositionalEncoder(d_model)
    self.drop = nn.Dropout(dropout)
    self.l_stack = nn.ModuleList([EncoderStack(d_model,n_vocab,n_head)
                                  for _ in range(n_layers)])
    self.norm = nn.LayerNorm(d_model, eps=1e-6)
    self.scale_emb = scale_emb
    self.d_model = d_model
    
  def forward(self, vocab_seq, mask, return_attn: bool = False):
    attn_list = []
    enc_output = self.src_vocab_embedding(vocab_seq)
    enc_output = self.drop(self.position_encoder(enc_output))
    enc_output = self.norm(enc_output)

    for layer in self.l_stack:
      enc_output, slf_attn = layer(enc_output, mask)
      slf_attn_list += [slf_attn]
