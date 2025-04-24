import numpy as np
import torch
from typing import Union, Tuple, Any

def n2t(x, *, dtype=torch.float32, device=None):
    """
    Behaves like torch.as_tensor but silently copies once if the NumPy
    array is read-only.  No warning, no unnecessary reallocations.
    """
    if isinstance(x, np.ndarray) and not x.flags.writeable:
        x = np.array(x, copy=True)              # one cheap copy
    return torch.as_tensor(x, dtype=dtype, device=device)

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
    if isinstance(data, np.ndarray):
        # Calculate new shape: product of first n_dims, followed by remaining dims
        new_shape = (-1,) + data.shape[n_dims:]
        return data.reshape(new_shape)
    elif isinstance(data, torch.Tensor):
        new_shape = (-1,) + tuple(data.shape[n_dims:])
        return data.view(new_shape)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected numpy.ndarray or torch.Tensor")

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