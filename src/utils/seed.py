import random
import os
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms (slower but more reproducible)
    """
    # Python random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    if deterministic:
        # Make CuDNN deterministic (can impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set PyTorch to use deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation
    
    Args:
        device: Device preference ("auto", "cpu", "cuda", "mps")
        
    Returns:
        torch.device: The selected device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def get_device_info(device: torch.device) -> dict:
    """
    Get information about the device
    
    Args:
        device: The device to get info about
        
    Returns:
        dict: Device information
    """
    info = {
        "device": str(device),
        "type": device.type,
    }
    
    if device.type == "cuda":
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(device.index or 0),
                "device_capability": torch.cuda.get_device_capability(device.index or 0),
                "total_memory": torch.cuda.get_device_properties(device.index or 0).total_memory,
            })
    elif device.type == "mps":
        info.update({
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
        })
    
    return info


def print_system_info(logger=None):
    """
    Print system information including PyTorch version, device info, etc.
    
    Args:
        logger: Logger instance to use for printing
    """
    info_lines = [
        f"PyTorch version: {torch.__version__}",
        f"NumPy version: {np.__version__}",
        f"Python random seed support: Available",
        f"CUDA available: {torch.cuda.is_available()}",
    ]
    
    if torch.cuda.is_available():
        info_lines.extend([
            f"CUDA version: {torch.version.cuda}",
            f"CUDA device count: {torch.cuda.device_count()}",
        ])
    
    if hasattr(torch.backends, 'mps'):
        info_lines.append(f"MPS available: {torch.backends.mps.is_available()}")
    
    for line in info_lines:
        if logger:
            logger.info(line)
        else:
            print(line)