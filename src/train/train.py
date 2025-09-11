import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from transformer.transformer.model import Transformer
from utils.config import Config
from utils.logger import setup_logger
from utils.seed import set_seed, get_device

def get_dummy_dataloader(config, device):
    """Creates a dummy dataloader for training."""
    batch_size = config.get("training.batch_size")
    max_seq_len = config.get("model.max_seq_len")
    vocab_size = config.get("model.vocab_size")
    

    num_samples = batch_size * 5
    src_seq = torch.randint(1, vocab_size, (num_samples, max_seq_len), device=device, dtype=torch.long)
    tgt_seq = torch.randint(1, vocab_size, (num_samples, max_seq_len), device=device, dtype=torch.long)
    
    dataset = TensorDataset(src_seq, tgt_seq)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def main():
    """Main training function."""
    config = Config()
    logger = setup_logger(config.get("logging"))
    logger.info("Starting training script...")

    seed = config.get("system.seed")
    set_seed(seed)
    device = get_device(config.get("system.device"))
    logger.info(f"Using device: {device}")

    train_loader = get_dummy_dataloader(config, device)
    
    model_config = config.get("model")
    model = Transformer(
        n_in_vocab=model_config['vocab_size'],
        n_out_vocab=model_config['vocab_size'],
        in_pad_idx=0, # Assuming 0 is pad
        out_pad_idx=0, # Assuming 0 is pad
        d_word_vec=model_config['d_model'],
        d_model=model_config['d_model'],
        d_inner=model_config['d_ff'],
        n_layers=model_config['n_layers'],
        n_head=model_config['n_heads'],
        d_k=model_config['d_model'] // model_config['n_heads'],
        d_v=model_config['d_model'] // model_config['n_heads'],
        dropout=model_config['dropout'],
        n_position=model_config['max_seq_len'],
        out_emb_prj_weight_sharing=model_config['tie_weights'],
        in_emb_prj_weight_sharing=model_config['tie_weights'],
    ).to(device)
    logger.info(f"Transformer model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    optimizer_config = config.get("optimizer")
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=optimizer_config['lr'], 
        betas=optimizer_config['betas'], 
        eps=optimizer_config['eps']
    )


    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=config.get("training.label_smoothing"))

    # --- 5. Training Loop ---
    max_epochs = config.get("training.max_epochs")
    log_interval = config.get("logging.log_interval")
    
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            src_seq, tgt_seq = batch
            
            tgt_input = tgt_seq[:, :-1]
            tgt_output = tgt_seq[:, 1:]

            optimizer.zero_grad()
            
            logits = model(src_seq, tgt_input)
            
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("training.gradient_clip"))
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % log_interval == 0:
                avg_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                logger.info(f'Epoch [{epoch+1}/{max_epochs}], Step [{i+1}/{len(train_loader)}], ' \
                            f'Loss: {avg_loss:.4f}, Elapsed: {elapsed:.2f}s')
                total_loss = 0
                start_time = time.time()

    logger.info("Training finished.")

if __name__ == "__main__":
    main()
