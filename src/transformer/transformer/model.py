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
      attn_list += [slf_attn] if return_attn else []

    if return_attn:
      return enc_output, attn_list
    return enc_output

class Decoder(nn.Module):
  def __init__(self, n_trg_vocab, d_vocab_vec, n_layers, n_head, dk, dv, d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):
    self.trg_vocab_embedding = nn.Embedding(n_trg_vocab, d_vocab_vec, padding_idx=pad_idx)
    self.position_encoder = PositionalEncoder(d_model)
    self.drop = nn.Dropout(dropout)

    self.l_stack = nn.ModuleList([DecoderStack(d_model, n_trg_vocab,n_head)
                                  for _ in range(n_layers)])
    self.norm = nn.LayerNorm(d_model, eps=1e-6)
    self.scale_emb = scale_emb
    self.d_model = d_model

  def forward(self, x, trg_mask, enc_output, src_mask, return_attn: bool = False):
    dec_attn_list, dec_enc_attn_list = [], []
    dec_output = self.trg_vocab_embedding(x)
    dec_output = self.drop(self.position_encoder(dec_output))
    dec_output = self.norm(dec_output)

    for layer in self.l_stack:
      dec_output, slf_attn, dec_enc_output= layer(dec_output, enc_output, trg_mask, src_mask)
      dec_attn_list = [slf_attn] if return_attn else []
      dec_enc_attn_list = [dec_enc_output] if return_attn else []

    if return_attn:
      return dec_output, dec_attn_list, dec_enc_attn_list
    return dec_output 

class Transformer(nn.Module):
  def __init__(self, n_in_vocab, n_out_vocab, in_pad_idx, out_pad_idx, d_word_vec=512, d_model=512, d_inner=2048, n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200, out_emb_prj_weight_sharing=True, in_emb_prj_weight_sharing=True):
    super().__init__()
    self.in_pad_idx = in_pad_idx
    self.out_pad_idx = out_pad_idx
    self.d_model = d_model

    self.encoder = Encoder(
      n_vocab=n_in_vocab,
      d_vocab_vec=d_word_vec,
      n_layers=n_layers,
      n_head=n_head,
      dk=d_k,
      dv=d_v,
      d_model=d_model,
      d_inner=d_inner,
      pad_idx=in_pad_idx,
      dropout=dropout,
      n_position=n_position,
      scale_emb=False,
    )

    self.decoder = Decoder(
      n_trg_vocab=n_out_vocab,
      d_vocab_vec=d_word_vec,
      n_layers=n_layers,
      n_head=n_head,
      dk=d_k,
      dv=d_v,
      d_model=d_model,
      d_inner=d_inner,
      pad_idx=out_pad_idx,
      dropout=dropout,
      n_position=n_position,
      scale_emb=False,
    )

    self.trg_word_prj = nn.Linear(d_model, n_out_vocab, bias=False)

    if out_emb_prj_weight_sharing:
      self.trg_word_prj.weight = self.decoder.trg_vocab_embedding.weight

    if in_emb_prj_weight_sharing:
      self.decoder.trg_vocab_embedding.weight = self.encoder.src_vocab_embedding.weight

    self.x_logit_scale = 1.0 / np.sqrt(d_model)

  def forward(self, src_seq, tgt_seq, src_mask=None, tgt_mask=None, return_attn: bool = False):
    enc_out = self.encoder(src_seq, src_mask, return_attn=False)

    dec_out = self.decoder(tgt_seq, tgt_mask, enc_out, src_mask, return_attn=False)

    logits = self.trg_word_prj(dec_out) * self.x_logit_scale

    return logits
