#!/usr/bin/env python3
"""
Smoke test for the Attention Is All You Need implementation.

This test verifies:
- System setup (CUDA/CPU detection)
- Random seed configuration
- Dummy batch creation and tensor operations
- Basic project structure

Run with: python -m tests.smoke
"""

import sys
import os
from pathlib import Path

# Add src to path to import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np

from utils.config import Config
from utils.logger import setup_logger
from utils.seed import set_seed, get_device, get_device_info, print_system_info


def create_dummy_batch(config: Config, device: torch.device):
    """
    Create a dummy batch for testing tensor operations
    """
    batch_size = config.get("training.batch_size", 32)
    seq_len = config.get("model.max_seq_len", 128)
    vocab_size = config.get("model.vocab_size", 32000)
    
    # Create dummy input tensors
    src_tokens = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    tgt_tokens = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    
    # Create attention masks (random padding masks)
    src_padding_mask = torch.rand(batch_size, seq_len, device=device) > 0.1
    tgt_padding_mask = torch.rand(batch_size, seq_len, device=device) > 0.1
    
    # Create causal mask for decoder
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    
    batch = {
        "src_tokens": src_tokens,
        "tgt_tokens": tgt_tokens,
        "src_padding_mask": src_padding_mask,
        "tgt_padding_mask": tgt_padding_mask,
        "causal_mask": causal_mask,
    }
    
    return batch


def test_basic_operations(batch: dict, device: torch.device):
    """
    Test basic tensor operations on the dummy batch
    """
    src_tokens = batch["src_tokens"]
    
    # Test basic tensor operations
    assert src_tokens.device.type == device.type, f"Tensor not on correct device type: {src_tokens.device.type} != {device.type}"
    
    # Test embedding-like operation
    d_model = 512
    embedding_table = torch.randn(32000, d_model, device=device)
    embedded = torch.embedding(embedding_table, src_tokens)
    
    assert embedded.shape == (src_tokens.shape[0], src_tokens.shape[1], d_model), \
        f"Embedding shape incorrect: {embedded.shape}"
    
    # Test attention-like operation (simplified)
    batch_size, seq_len = src_tokens.shape
    q = torch.randn(batch_size, seq_len, d_model, device=device)
    k = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Scaled dot-product attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_model ** 0.5)
    attention_weights = torch.softmax(scores, dim=-1)
    
    assert attention_weights.shape == (batch_size, seq_len, seq_len), \
        f"Attention weights shape incorrect: {attention_weights.shape}"
    
    return True


def run_smoke_test():
    """
    Run the complete smoke test
    """
    print("🚀 Starting Attention Is All You Need - Smoke Test")
    print("=" * 60)
    
    # 1. Load configuration
    try:
        config = Config()
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False
    
    # 2. Setup logging
    try:
        logger = setup_logger(config=config.get("logging", {}))
        print("✓ Logger setup successfully")
    except Exception as e:
        print(f"✗ Failed to setup logger: {e}")
        return False
    
    # 3. Set seed for reproducibility
    seed = config.get("system.seed", 42)
    set_seed(seed)
    logger.info(f"🎲 Random seed set to: {seed}")
    print(f"✓ Random seed set to: {seed}")
    
    # 4. Device detection and setup
    device_config = config.get("system.device", "auto")
    device = get_device(device_config)
    device_info = get_device_info(device)
    
    logger.info(f"🖥️  Using device: {device}")
    print(f"✓ Device selected: {device}")
    
    # Print detailed system information
    print("\n📊 System Information:")
    print("-" * 30)
    print_system_info()
    
    print(f"\n🔧 Device Details:")
    print("-" * 20)
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # 5. Create and test dummy batch
    try:
        batch = create_dummy_batch(config, device)
        logger.info("📦 Dummy batch created successfully")
        print(f"✓ Dummy batch created")
        
        # Print batch information
        print(f"\n📦 Batch Information:")
        print("-" * 25)
        for key, tensor in batch.items():
            if torch.is_tensor(tensor):
                print(f"  {key}: {tensor.shape} ({tensor.dtype}) on {tensor.device}")
            else:
                print(f"  {key}: {tensor}")
                
    except Exception as e:
        logger.error(f"Failed to create dummy batch: {e}")
        print(f"✗ Failed to create dummy batch: {e}")
        return False
    
    # 6. Test basic operations
    try:
        test_basic_operations(batch, device)
        logger.info("🧪 Basic tensor operations test passed")
        print("✓ Basic tensor operations test passed")
    except Exception as e:
        logger.error(f"Basic operations test failed: {e}")
        print(f"✗ Basic operations test failed: {e}")
        return False
    
    # 7. Memory information (if CUDA)
    if device.type == "cuda" and torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**2    # MB
        print(f"\n💾 GPU Memory Usage:")
        print("-" * 25)
        print(f"  Allocated: {memory_allocated:.1f} MB")
        print(f"  Reserved:  {memory_reserved:.1f} MB")
    
    print("\n" + "=" * 60)
    print("🎉 Smoke test completed successfully!")
    print("✅ Project setup is working correctly")
    
    # Cleanup
    logger.close()
    
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)