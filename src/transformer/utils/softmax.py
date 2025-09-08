import numpy as np
import logging
from typing import Union

logger = logging.getLogger(__name__)


def softmax(Z: Union[np.ndarray, float]) -> np.ndarray:
    """
    Compute the softmax function for the input array.
    
    The softmax function is defined as:
    softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    
    This implementation includes numerical stability by subtracting the maximum
    value to prevent overflow.
    
    Args:
        Z (Union[np.ndarray, float]): Input array or scalar value.
            Can be a vector or matrix of real numbers.
    
    Returns:
        np.ndarray: Softmax output with the same shape as input.
            Each element is in the range (0, 1) and the sum equals 1.
    
    Raises:
        ValueError: If input contains NaN or infinite values.
    
    Examples:
        >>> softmax(np.array([1, 2, 3]))
        array([0.09003057, 0.24472847, 0.66524096])
        
        >>> softmax(np.array([[1, 2], [3, 4]]))
        array([[0.26894142, 0.73105858],
               [0.26894142, 0.73105858]])
    """
    Z = np.array(Z, dtype=np.float32)
    
    if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
        logger.error(f"Invalid input to softmax: contains NaN or infinite values")
        raise ValueError("Input contains NaN or infinite values")
    
    logger.debug(f"Computing softmax for input shape: {Z.shape}")
    
    Z_stable = Z - np.max(Z, axis=-1, keepdims=True)
    
    exp_Z = np.exp(Z_stable)
    
    softmax_output = exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)
    
    logger.debug(f"Softmax computation completed successfully")
    
    return softmax_output