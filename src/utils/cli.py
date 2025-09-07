import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from .config import Config


def create_base_parser() -> argparse.ArgumentParser:
    """
    Create base argument parser with common arguments
    """
    parser = argparse.ArgumentParser(
        description="Attention Is All You Need - Transformer Implementation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration file (YAML)"
    )
    
    # System settings
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default=None,
        help="Device to use for computation"
    )
    
    # Model settings
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["base", "large", "small"],
        default=None,
        help="Pre-defined model size configuration"
    )
    
    # Training settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=None,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Maximum number of training epochs"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file"
    )
    
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        help="TensorBoard logging directory"
    )
    
    # Checkpoints
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints"
    )
    
    return parser


def create_train_parser() -> argparse.ArgumentParser:
    """
    Create parser for training command
    """
    parser = create_base_parser()
    parser.prog = "train"
    parser.description = "Train the Transformer model"
    
    # Training-specific arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        required=False,
        help="Path to training data directory"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation, no training"
    )
    
    return parser


def create_inference_parser() -> argparse.ArgumentParser:
    """
    Create parser for inference command
    """
    parser = create_base_parser()
    parser.prog = "inference"
    parser.description = "Run inference with the Transformer model"
    
    # Inference-specific arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Input file for batch inference"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for results"
    )
    
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam size for beam search"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum generation length"
    )
    
    return parser


def merge_args_with_config(args: argparse.Namespace, config: Config) -> Config:
    """
    Merge command line arguments with configuration file
    Command line arguments take precedence over config file
    """
    # Create a mapping from arg names to config paths
    arg_to_config = {
        "seed": "system.seed",
        "device": "system.device",
        "batch_size": "training.batch_size",
        "learning_rate": "optimizer.lr",
        "max_epochs": "training.max_epochs",
        "log_level": "logging.log_level",
        "tensorboard_dir": "logging.tensorboard_dir",
        "beam_size": "inference.beam_size",
        "max_length": "inference.max_decode_length",
    }
    
    # Apply model size presets
    if hasattr(args, 'model_size') and args.model_size:
        model_presets = {
            "small": {
                "model.d_model": 256,
                "model.n_heads": 4,
                "model.n_layers": 4,
                "model.d_ff": 1024,
            },
            "base": {
                "model.d_model": 512,
                "model.n_heads": 8,
                "model.n_layers": 6,
                "model.d_ff": 2048,
            },
            "large": {
                "model.d_model": 1024,
                "model.n_heads": 16,
                "model.n_layers": 12,
                "model.d_ff": 4096,
            }
        }
        
        if args.model_size in model_presets:
            for key, value in model_presets[args.model_size].items():
                config.set(key, value)
    
    # Override config with command line arguments
    for arg_name, config_path in arg_to_config.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            config.set(config_path, arg_value)
    
    return config


def setup_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Setup configuration and other components from parsed arguments
    
    Returns:
        Dict containing config, logger, device info, etc.
    """
    # Load configuration
    if args.config:
        config = Config(args.config)
    else:
        config = Config()  # Use default config
    
    # Merge command line arguments
    config = merge_args_with_config(args, config)
    
    # Additional setup can be added here (logger, device, etc.)
    setup = {
        "config": config,
        "args": args,
    }
    
    return setup