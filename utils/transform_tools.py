import numpy as np
import torch
from typing import Union, Tuple, Any

def to_tensor(x, *, device, dtype=torch.float32, copy=False):
    """Fast conversion to torch tensor on specified device.
    
    Usage:
        buffers: to_t(obs, device=self.device)  # zero-copy
        runners: to_t(states, device=device, copy=True)  # safe copy for async env
    """
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype, non_blocking=True)
    
    #  # Add debug info for non-writable arrays
    # if isinstance(x, np.ndarray) and not x.flags.writeable:
    #     print(f"Warning: Non-writable array detected:")
    #     print(f"Shape: {x.shape}")
    #     print(f"Type: {x.dtype}")
    #     print(f"Memory flags: {x.flags}")
    #     # Stack trace to identify caller
    #     import traceback
    #     print("Call stack:")
    #     traceback.print_stack()
    #     copy = True

    if copy or not x.flags.writeable: # Always copy when the NumPy array is not writeable
        return torch.tensor(x, device=device, dtype=dtype)  # always copies
    return torch.as_tensor(x, device=device, dtype=dtype)   # may alias, no copy

def flatten_first_dims(data: Union[np.ndarray, torch.Tensor], n_dims: int = 2) -> Union[np.ndarray, torch.Tensor]:
    """
    Efficiently flatten the first n dimensions of an array/tensor while preserving the rest.
    Much faster than np.concatenate or torch.cat for this purpose.
    
    Args:
        data: Input array/tensor with shape (D1, D2, ..., Dn, *remaining_dims)
        n_dims: Number of first dimensions to flatten. Default is 2 for common case of (batch, agents, *features)
    
    Returns:
        Reshaped array/tensor with shape (D1*D2*...*Dn, *remaining_dims)
    
    Examples:
        >>> obs = np.zeros((32, 4, 256))  # (batch_size, n_agents, obs_dim)
        >>> flat_obs = flatten_first_dims(obs)  # shape: (128, 256)
        
        >>> states = torch.zeros(16, 8, 4, 64)  # (timesteps, batch, agents, hidden_dim)
        >>> flat_states = flatten_first_dims(states, n_dims=3)  # shape: (512, 64)
    
    Note:
        This is significantly faster than using np.concatenate/torch.cat as it only changes
        the view of the data without copying. Use this instead of:
        - np.concatenate(data)
        - np.vstack(data)
        - torch.cat(data)
        when the data is already in a single array/tensor.
    """
    if not isinstance(data, (np.ndarray, torch.Tensor)):
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(data)}")
    
    new_shape = (-1, *data.shape[n_dims:])
    return data.reshape(new_shape)     

def unflatten_first_dim(data: Union[np.ndarray, torch.Tensor], 
                       dims_to_unflatten: Tuple[int, ...]) -> Union[np.ndarray, torch.Tensor]:
    """
    Unflatten only specified dimensions while keeping the rest unchanged.
    
    Args:
        data: Flattened input with shape (D1*D2*...*Dn, *remaining_dims)
        dims_to_unflatten: Dimensions to unflatten into (e.g., (16, 8) to unflatten first dim into (16, 8))
    
    Returns:
        Reshaped array/tensor
    
    Examples:
        >>> x = torch.zeros(128, 4, 64)  # (batch*n_agents, obs_dim)
        >>> unflat_x = unflatten_first_dim(x, (16, 8))  # shape: (16, 8, 4, 64)
    """
    if isinstance(data, (np.ndarray, torch.Tensor)):
        return data.reshape(dims_to_unflatten + data.shape[1:])
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

def flatten_time_batch(T: int, N: int, x: torch.Tensor) -> torch.Tensor:
    """Flatten time and batch dimensions while preserving the rest.

    Memory-efficient flattening of first two dimensions using PyTorch's view operation.
    Commonly used in RL for reshaping (timesteps, batch, *features) -> (timesteps*batch, *features).

    Args:
        T (int): Time dimension size
        N (int): Batch dimension size
        x (torch.Tensor): Input tensor with shape (T, N, *features)

    Returns:
        torch.Tensor: Reshaped tensor with shape (T*N, *features)

    Examples:
        >>> states = torch.zeros(32, 8, 256)  # (timesteps, batch_size, hidden_dim)
        >>> flat_states = flatten_time_batch(32, 8, states)  # shape: (256, 256)

    Raises:
        ValueError: If x.shape[:2] doesn't match (T, N)
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Input must be torch tensor, got {type(x)}")
    
    if x.shape[0] != T or x.shape[1] != N:
        raise ValueError(f"Shape mismatch. Expected first two dims ({T}, {N}), got {x.shape[:2]}")
    
    return x.reshape(T * N, *x.shape[2:])