import numpy as np
import torch

from buffers.rollout_storage import RolloutStorage
from utils.transform_tools import flatten_time_batch, to_tensor

def _transform_data(data_np: np.ndarray, device: torch.device, sequence_first: bool = False) -> torch.Tensor:
    """
    Transform data to be used in RNN.

    Args:
        data_np: Input numpy array with shape [T, N, feat_dim] or [T, N, num_layers, M, feat_dim] for RNN states
        device: Target device for tensor
        sequence_first: If True, keeps sequence dimension first [T, N, feat_dim] -> [T, N, feat_dim]
                      If False, transposes sequence and batch [T, N, feat_dim] -> [N, T, feat_dim] -> [N*T, feat_dim]
                      So we get steps sorted by rollout threads and agents.

    Returns:
        Transformed torch tensor
    """
    if sequence_first:
        # Keep original sequence ordering
        return to_tensor(data_np, device=device, copy=True)
    else:
        # Original behavior: transpose and flatten
        # Use copy() to ensure contiguous memory layout in NumPy
        reshaped_data = data_np.transpose(1, 0, 2).reshape(-1, *data_np.shape[2:])
        return to_tensor(reshaped_data,  device=device, copy=True)

class AgentRolloutView:
    """Zero-copy view of rollout storage data for a specific agent.
    
    This class provides an efficient interface to access a single agent's data
    from the joint rollout storage. It uses numpy/torch stride tricks to avoid
    data copying, making it memory-efficient for HAPPO training.

    Shapes:
        T: Number of timesteps
        N: Number of parallel environments (n_rollout_threads)
        ...: Variable number of feature dimensions

    Properties:
        obs (np.ndarray): Observations, shape (T+1, N, ...)
        actions (np.ndarray): Actions taken, shape (T, N, ...)
        logp_old (np.ndarray): Log probabilities, shape (T, N, 1)
    """
    def __init__(self, parent: "RolloutStorage", idx: int):
        """Initialize view for specific agent's data.

        Args:
            parent (RolloutStorage): Parent storage containing all agents' data
            idx (int): Agent index to create view for

        Raises:
            IndexError: If idx is out of valid range
        """
        if not 0 <= idx < parent.n_agents:
            raise IndexError(f"Agent index {idx} out of range [0, {parent.n_agents})")
        
        self._p = parent
        self.idx = idx
        self.n_rollout_threads = parent.n_rollout_threads
        self.n_stpes = parent.n_steps
        self.device = parent.device

    @property
    def obs(self):      
        """Agent's observations across all timesteps and environments."""
        return self._p.obs[:, :, self.idx]          # (T+1, N, obs_dim)
    
    @property
    def rewards(self): 
        """Rewards received by the agent."""
        return self._p.rewards[:, :, self.idx]      # (T, N, 1)
    
    @property
    def actions(self):  
        """Actions taken by the agent."""
        return self._p.actions[:, :, self.idx]      # (T, N, action_dim)
    
    @property
    def action_log_probs(self): 
        """Log probabilities of taken actions."""
        return self._p.action_log_probs[:, :, self.idx]     # (T,   N, 1)

    @property
    def masks(self): 
        """Masks indicating episode termination."""
        return self._p.masks[:, :, self.idx]        # (T+1, N, 1)
    
    @property
    def active_masks(self): 
        """Masks indicating agent's active status."""
        return self._p.active_masks[:, :, self.idx] # (T+1, N, 1)
    
    @property
    def truncated(self): 
        """Masks indicating episode truncation."""
        return self._p.truncated[:, :, self.idx]    # (T+1, N, 1)
    
    @property
    def available_actions(self): 
        """Available actions for the agent."""
        return self._p.available_actions[:, :, self.idx] if self._p.available_actions is not None else None# (T+1, N, action_dim)
    
    @property
    def actor_rnn_states(self): 
        """RNN states for the agent's policy."""
        # (T+1, N, num_layers, hidden_size)
        return self._p.actor_rnn_states[:, :, self.idx] if self._p.use_rnn else None 
    
    @property
    def advantages(self): 
        """Advantages for the agent's actions."""
        return self._p.advantages[:, :, self.idx]   # (T, N, action_dim)

    def get_minibatches(self, num_mini_batch = 1, mini_batch_size=None):
        """
        Create minibatches for training - not RNN version

        Args:
            num_mini_batch (int): Number of minibatches to create (default: 1)
            mini_batch_size (int, optional): Size of each minibatch, if None will be calculated
                                            based on num_mini_batch

        Returns:
            Generator yielding minibatches for training
        """
        if self._p.use_rnn:
            raise ValueError("RNN is enabled, cannot use get_minibatches")
        
        batch_size = self._p.n_steps * self._p.n_rollout_threads  # e.g., T * N = 400 * 8 = 3200

        if mini_batch_size is None:
            mini_batch_size = batch_size // num_mini_batch # e.g., 3200 // 1 = 3200
        
        # Create random indices for minibatches
        batch_inds = np.random.permutation(batch_size)

        # Preshape data to improve performance (only do this once)
        # Batch size is [T, N, feat_dim] -> [T*N, feat_dim]
        data = {
            'obs': self.obs[:-1].reshape(-1, *self.obs.shape[2:]),
            'actions': self.actions.reshape(-1, self.actions.shape[-1]),
            'masks': self.masks[:-1].reshape(-1, 1),
            'active_masks': self.active_masks[:-1].reshape(-1, 1),
            'old_action_log_probs': self.action_log_probs.reshape(-1, 1),
            'advantages': self.advantages.reshape(-1, 1),
        }

        # Handle available actions specially
        if self.available_actions is not None:
            data['available_actions'] = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[2:])

        #Yield minibatches
        start_ind = 0
        for _ in range(num_mini_batch):
            end_ind = min(start_ind + mini_batch_size, batch_size)
            if end_ind - start_ind < 1:  # Skip empty batches
                continue

            batch_inds_subset = batch_inds[start_ind:end_ind]

            batch = {
                   key: torch.tensor(data[key][batch_inds_subset], dtype=torch.float32).to(self.device)
                for key in data.keys()
            }

            yield (
                batch['obs'],
                None, #actor_rnn_states,
                batch['actions'],
                batch['masks'],
                batch['active_masks'],
                batch['old_action_log_probs'],
                batch['advantages'],
                batch['available_actions'] if 'available_actions' in batch else None# Handle optional key
            ), batch_inds_subset

            start_ind = end_ind
    
    def  get_minibatches_seq_first(self, num_mini_batch = 1, data_chunk_length = 10):
        """
        Create minibatches for training RNN, flattening num_agents into total steps.
        Returns sequences with shape [seq_len*batch_size, feat_dim] and RNN states
        with shape [num_layers, batch_size, hidden_dim].

        Args:
            num_mini_batch (int): Number of minibatches to create (default: 1)
            data_chunk_length (int): Length of data chunk to use for training, default is 10

        Returns:
            Generator yielding minibatches as tuples with the following keys:
            - obs: [seq_len*batch_size, obs_dim]
            - actor_rnn_states: [num_layers, batch_size, hidden_size]
            - actions, masks, active_masks, etc.: [seq_len*batch_size, dim]
        """
        if not self._p.use_rnn:
            raise ValueError("RNN is not enabled, cannot use get_minibatches_seq_first")

        total_steps = self._p.n_steps # e.g., [T, N, feat_dim] -> T
        if total_steps < data_chunk_length:
            raise ValueError(f"n_steps ({total_steps}) must be >= data_chunk_length ({data_chunk_length})")

        # Calculate chunks and batch sizes
        rollout_threads = self._p.n_rollout_threads
        total_agent_steps = total_steps * rollout_threads # e.g., T * N = 400 * 8 = 3200
        max_data_chunks = total_agent_steps // data_chunk_length  # e.g., 3200 // 10 = 320

        # Adjust mini_batch count if needed
        num_mini_batch = min(num_mini_batch, max_data_chunks) # e.g., 1
        mini_batch_size = max(1, max_data_chunks // num_mini_batch)  # e.g., 320 // 1 = 320

        # Pre-convert and flatten data, 
        # [T, N, feat_dim] -> [N, T, feat_dim] -> [N*T, feat_dim]
        data = {
            'obs': _transform_data(self.obs[:-1], self.device),
            'actions': _transform_data(self.actions, self.device),
            'masks': _transform_data(self.masks[:-1], self.device),
            'active_masks': _transform_data(self.active_masks[:-1], self.device),
            'old_action_log_probs': _transform_data(self.action_log_probs, self.device),
            'advantages': _transform_data(self.advantages, self.device),
        }

        # Handle available actions specially
        if self.available_actions is not None:
            data['available_actions'] = _transform_data(self.available_actions[:-1], self.device)

        # Process RNN states - maintain num_layers dimension while flattening agents
        # [T, N, num_layers, hidden_size] -> [N, T, num_layers, hidden_size] -> [N*T, num_layers, hidden_size]
        actor_rnn_states  =  self.actor_rnn_states[:-1].transpose(1, 0, 2, 3).reshape(-1,
                                                                                    self.actor_rnn_states.shape[-2],
                                                                                    self.actor_rnn_states.shape[-1])
        data['actor_rnn_states'] = torch.tensor(actor_rnn_states, dtype=torch.float32, device=self.device)

        # This is just [0, 1, 2, …, T*N-1] on CPU; cheap to allocate once.
        flat_index_master = torch.arange(
            self._p.n_steps * self._p.n_rollout_threads,
            dtype=torch.long
        )

        # Generate chunk start indices over total_agent_steps
        all_starts = np.arange(0, total_agent_steps - data_chunk_length + 1, data_chunk_length)
        if len(all_starts) > max_data_chunks:
            all_starts = all_starts[:max_data_chunks]
        np.random.shuffle(all_starts)

        # Generate minibatches
        for batch_start in range(0, len(all_starts), mini_batch_size): #(0, 320, 320)
            batch_chunk_starts = all_starts[batch_start:batch_start + mini_batch_size] #all_starts[0:320]
            if not batch_chunk_starts.size:
                continue

            # Collect sequences
            sequences = {key: [] for key in data.keys()}
            batch_indices  = [] 

            for start_idx in batch_chunk_starts:
                end_idx = start_idx + data_chunk_length

                # Collect sequences for each data type
                for key in data.keys():
                    if key not in ['actor_rnn_states']:
                        sequences[key].append(data[key][start_idx:end_idx])

                # Get initial RNN states
                sequences['actor_rnn_states'].append(data['actor_rnn_states'][start_idx])

                # Note: start_idx/end_idx are already in flattened space (T*N)
                batch_indices.append(flat_index_master[start_idx:end_idx])

            stack_dims = {
                'actor_rnn_states': 0,
            }

            # Stack sequences into proper shapes (seq_len, batch_size, feat_dim) or (batch_size, num_layers, hidden_dim)
            batch = {
                key: torch.stack(sequences[key], dim=stack_dims.get(key, 1))
                for key in sequences
            }

            batch_indices = torch.cat(batch_indices, dim=0) # shape (T*N_mini ,)
            
            T, N = data_chunk_length, mini_batch_size
            
            # Yield the minibatch as a tuple
            yield (
                flatten_time_batch(T, N, batch['obs']),
                batch['actor_rnn_states'],
                flatten_time_batch(T, N, batch['actions']),
                flatten_time_batch(T, N, batch['masks']),
                flatten_time_batch(T, N, batch['active_masks']),
                flatten_time_batch(T, N, batch['old_action_log_probs']),
                flatten_time_batch(T, N, batch['advantages']),
                flatten_time_batch(T, N, batch['available_actions']) if 'available_actions' in batch else None # Handle optional key
            ), batch_indices
