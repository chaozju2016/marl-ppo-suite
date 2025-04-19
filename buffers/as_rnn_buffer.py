import numpy as np
import torch

from utils.env_tools import get_shape_from_obs_space, get_shape_from_act_space

def _transform_data(data_np: np.ndarray, device: torch.device, sequence_first: bool = False) -> torch.Tensor:
    """
    Transform data to be used in RNN.
    
    Args:
        data_np: Input numpy array with shape [T, N, M, feat_dim] or [T, N, num_layers, M, feat_dim] for RNN states
        device: Target device for tensor
        sequence_first: If True, keeps sequence dimension first [T, N, M, feat_dim] -> [T, N, M, feat_dim]
                      If False, transposes sequence and batch [T, N, M, feat_dim] -> [N, M, T, feat_dim] -> [N*M*T, feat_dim]
                      So we get steps sorted by rollout threads and agents.
    
    Returns:
        Transformed torch tensor
    """
    if sequence_first:
        # Keep original sequence ordering
        return torch.tensor(data_np, dtype=torch.float32).to(device)
    else:
        # Original behavior: transpose and flatten
        reshaped_data = data_np.transpose(1, 2, 0, 3).reshape(-1, *data_np.shape[2:])
        return torch.tensor(reshaped_data, dtype=torch.float32).to(device)


class AgentSpecificRecurrentRolloutStorage:
    """
    Rollout storage for collecting multi-agent experiences during training.
    Designed for MAPPO with n-step returns and RNN-based policies.
    Support for agent-specific global state and multiple parallel environments.
    """
    def __init__(self, n_steps, n_rollout_threads, n_agents, obs_space, action_space, state_space, hidden_size,
                 agent_specific_global_state=False,
                 num_rnn_layers=1, device='cpu'):
        """
        Initialize rollout storage for collecting experiences.  
        
        Args:
            n_steps (int): Number of steps to collect before update (can be different from episode length)
            n_rollout_threads (int): Number of parallel environments
            n_agents (int): Number of agents in the environment
            obs_space: Observation space
            action_space: Action space
            state_space: State space
            agent_specific_global_state: Whether to use agent-specific global state
            hidden_size (int): Dimension of hidden state  
            num_rnn_layers (int): Number of layers in the RNN  
            device (str): Device for storage ('cpu' for numpy-based implementation)
        """
        self.n_steps = n_steps
        self.n_rollout_threads = n_rollout_threads
        self.n_agents = n_agents
        self.num_rnn_layers = num_rnn_layers
        self.agent_specific_global_state = agent_specific_global_state
        self.device = torch.device(device)
        
        # Current position in the buffer
        self.step = 0
        # (obs_0, state_0) → action_0 / action_log_prob_0 → (reward_0, obs_1, state_1, mask_1, trunc_1)
        # delta = reward_0 + gamma * value_1 * mask_1 - value_0

        obs_shape = get_shape_from_obs_space(obs_space)
        action_shape = get_shape_from_act_space(action_space)
        state_shape = get_shape_from_obs_space(state_space)

        # Core storage buffers - using numpy arrays for efficiency
        self.obs = np.zeros((n_steps + 1, n_rollout_threads, n_agents, *obs_shape), dtype=np.float32)
        self.global_state = np.zeros((n_steps + 1, n_rollout_threads, n_agents, *state_shape), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_rollout_threads, n_agents, 1), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_rollout_threads, n_agents, action_shape), dtype=np.int64)
        self.action_log_probs = np.zeros((n_steps, n_rollout_threads, n_agents, 1), dtype=np.float32)
        self.values = np.zeros((n_steps + 1, n_rollout_threads, n_agents, 1), dtype=np.float32)
        self.masks = np.ones((n_steps + 1, n_rollout_threads, n_agents, 1), dtype=np.float32) # 0 if episode done, 1 otherwise
        self.truncated = np.zeros((n_steps + 1, n_rollout_threads, n_agents, 1), dtype=np.bool_) # 1 if episode truncated, 0 otherwise
        
        # Initialize available actions buffer if shape is Discrete
        if action_space.__class__.__name__ == "Discrete":
            self.available_actions = np.ones((n_steps+1, n_rollout_threads, n_agents, action_space.n), dtype=np.float32)
        else:
            self.available_actions = None

        # RNN hidden states
        # Shape: [n_steps + 1, n_rollout_threads, n_agents, num_layers, hidden_size]
        self.actor_rnn_states = np.zeros(
            (n_steps + 1, n_rollout_threads, n_agents, num_rnn_layers, hidden_size), 
            dtype=np.float32)
        self.critic_rnn_states = np.zeros_like(self.actor_rnn_states)
        
        # Extra buffers for the algorithm
        self.returns = np.zeros((n_steps + 1, n_rollout_threads, n_agents, 1), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_rollout_threads, n_agents, 1), dtype=np.float32)
        
    def insert(self, obs, global_state, actions, 
        action_log_probs, values, rewards, 
        masks, truncates, available_actions,
        actor_rnn_states, critic_rnn_states):
        """
        Insert a new transition into the buffer.
        
        Args:
            obs: Agent observations [n_rollout_threads, n_agents, obs_shape]
            global_state: Global state if available [n_rollout_threads, n_agents, state_shape]
            actions: Actions taken by agents [n_rollout_threads, n_agents, action_shape]
            action_log_probs: Log probs of actions [n_rollout_threads, n_agents, 1]
            values: Value predictions [n_rollout_threads, n_agents, 1]
            rewards: Rewards received [n_rollout_threads, n_agents, 1]
            masks: Episode termination masks [n_rollout_threads, n_agents], 0 if episode done, 1 otherwise
            truncates: Boolean array indicating if episode was truncated (e.g., due to time limit) 
                      rather than terminated [n_rollout_threads, n_agents]
            available_actions: Available actions mask [n_rollout_threads, n_agents, n_actions]
            actor_rnn_states: RNN states [n_rollout_threads, num_layers, n_agents, hidden_size]
            critic_rnn_states: RNN states [n_rollout_threads, num_layers, n_agents, hidden_size]
        """
        self.obs[self.step + 1] = obs.copy()
        self.global_state[self.step + 1] = global_state.copy()
            
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.values[self.step] = values.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.truncated[self.step + 1] = truncates.copy()
        self.available_actions[self.step + 1] = available_actions.copy()
        
        self.actor_rnn_states[self.step + 1] = actor_rnn_states.copy()
        self.critic_rnn_states[self.step + 1] = critic_rnn_states.copy()
        
        self.step += 1
        
    def compute_returns_and_advantages(self, next_values, gamma=0.99, lambda_=0.95, use_gae=True):
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).
        Properly handles truncated episodes by incorporating next state values.

        Args:
            next_values: Value estimates for the next observations [n_rollout_threads, n_agents, 1]
            gamma: Discount factor
            lambda_: GAE lambda parameter for advantage weighting
            use_gae: Whether to use GAE or just n-step returns
        """
        # Set the value of the next observation
        self.values[-1] = next_values
        
        # Create arrays for storing returns and advantages
        advantages = np.zeros_like(self.rewards)
        returns = np.zeros_like(self.returns)
        
        if use_gae:
            # GAE advantage computation with vectorized operations for better performance
            gae = 0
            for step in reversed(range(self.n_steps)):       
                # For truncated episodes, we adjust rewards directly
                adjusted_rewards = self.rewards[step].copy() # [n_agents]
                
                # Identify truncated episodes (done but not terminated)
                truncated_mask = (self.masks[step + 1] == 0) & (self.truncated[step + 1] == 1) # [n_agents]
                if np.any(truncated_mask):
                    # Add bootstrapped value only for truncated episodes
                    adjusted_rewards[truncated_mask] += gamma * self.values[step + 1][truncated_mask]
                    
                # Calculate delta (TD error) with adjusted rewards
                delta = (
                    adjusted_rewards + 
                    gamma * self.values[step + 1] * self.masks[step + 1] - 
                    self.values[step]
                ) # [n_agents]
                
                # Standard GAE calculation 
                gae = delta + gamma * lambda_ * self.masks[step + 1] * gae # [n_agents]
                advantages[step] = gae # [n_agents]
                
            # Compute returns as advantages + values
            returns[:-1] = advantages + self.values[:-1] # [n_agents]
            returns[-1] = next_values # [n_agents]
            
        else:
            # N-step returns without GAE (more efficient calculation)
            returns[-1] = next_values
            for step in reversed(range(self.n_steps)):
                # Adjust rewards for truncated episodes
                adjusted_rewards = self.rewards[step].copy()
                
                # Identify truncated episodes
                truncated_mask = (self.masks[step + 1] == 0) & (self.truncated[step + 1] == 1)
                
                # For truncated episodes, add discounted bootstrapped value directly to rewards
                if np.any(truncated_mask):
                    adjusted_rewards[truncated_mask] += gamma * returns[step + 1][truncated_mask]

                # Calculate returns with proper masking
                returns[step] = adjusted_rewards + gamma * returns[step + 1] * self.masks[step + 1]
                
            # Calculate advantages
            advantages = returns[:-1] - self.values[:-1]
        
        # Store results
        self.returns = returns
        
        # Normalize advantages (helps with training stability)
        # Use stable normalization with small epsilon
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        self.advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        
        return self.advantages, self.returns

    def after_update(self):
        """Copy the last observation and masks to the beginning for the next update."""
        self.obs[0] = self.obs[-1].copy()
        self.global_state[0] = self.global_state[-1].copy()    
        self.masks[0] = self.masks[-1].copy()
        self.truncated[0] = self.truncated[-1].copy()
        self.available_actions[0] = self.available_actions[-1].copy()

        # Copy RNN states
        self.actor_rnn_states[0] = self.actor_rnn_states[-1].copy()
        self.critic_rnn_states[0] = self.critic_rnn_states[-1].copy()

        # Reset step counter
        self.step = 0

    def get_minibatches_seq_first(self, num_mini_batch, data_chunk_length = 10):
        """
        Create minibatches for training RNN, flattening num_agents into total steps.
        Returns sequences with shape [seq_len, batch_size, feat_dim] and RNN states 
        with shape [num_layers, batch_size, hidden_dim].

        Args:
            num_mini_batch (int): Number of minibatches to create
            data_chunk_length (int): Length of data chunk to use for training, default is 10
        
        Returns:
            Generator yielding minibatches as tuples with the following keys:
            - obs: [seq_len, batch_size, obs_dim]
            - global_state: [seq_len, batch_size, state_dim]
            - actor_rnn_states: [num_layers, batch_size, hidden_size]
            - critic_rnn_states: [num_layers, batch_size, hidden_size]
            - actions, values, returns, etc.: [seq_len, batch_size, dim]
        """
        total_steps = self.n_steps # e.g., [T, N, M, feat_dim] -> T
        if total_steps < data_chunk_length:
            raise ValueError(f"n_steps ({total_steps}) must be >= data_chunk_length ({data_chunk_length})")
        
        # Calculate chunks and batch sizes
        rollout_threads = self.n_rollout_threads
        num_agents = self.n_agents
        total_agent_steps = total_steps * num_agents * rollout_threads # e.g., T * M * N = 400 * 5 * 8 = 16000
        max_data_chunks = total_agent_steps // data_chunk_length  # e.g., 16000 // 10 = 1600
        
        # Adjust mini_batch count if needed
        num_mini_batch = min(num_mini_batch, max_data_chunks) # e.g., 1
        mini_batch_size = max(1, max_data_chunks // num_mini_batch)  # e.g., 1600 // 1 = 1600

        # Pre-convert and flatten data, collapsing num_agents into the sequence
        data = {
            'obs': _transform_data(self.obs[:-1], self.device),
            'global_state': _transform_data(self.global_state[:-1], self.device),
            'actions': _transform_data(self.actions, self.device),
            'values': _transform_data(self.values[:-1], self.device),
            'returns': _transform_data(self.returns[:-1], self.device),
            'masks': _transform_data(self.masks[:-1], self.device),
            'old_action_log_probs': _transform_data(self.action_log_probs, self.device),
            'advantages': _transform_data(self.advantages, self.device),
            'available_actions': _transform_data(self.available_actions[:-1], self.device),
        }

        # Process RNN states - maintain num_layers dimension while flattening agents
        # [T, N, M, num_layers, hidden_size] -> [N, M, T, num_layers, hidden_size] -> [N*M*T, num_layers, hidden_size]
        actor_rnn_states  =  self.actor_rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, 
                                                                                      self.actor_rnn_states.shape[-2], 
                                                                                      self.actor_rnn_states.shape[-1])
        critic_rnn_states =  self.critic_rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, 
                                                                                       self.critic_rnn_states.shape[-2], 
                                                                                    self.critic_rnn_states.shape[-1])
 
        data['actor_rnn_states'] = torch.tensor(actor_rnn_states, dtype=torch.float32, device=self.device)
        data['critic_rnn_states'] = torch.tensor(critic_rnn_states, dtype=torch.float32, device=self.device)

        # Generate chunk start indices over total_agent_steps
        all_starts = np.arange(0, total_agent_steps - data_chunk_length + 1, data_chunk_length)
        if len(all_starts) > max_data_chunks:
            all_starts = all_starts[:max_data_chunks]
        np.random.shuffle(all_starts)

        # Generate minibatches
        for batch_start in range(0, len(all_starts), mini_batch_size):
            batch_chunk_starts = all_starts[batch_start:batch_start + mini_batch_size]
            if not batch_chunk_starts.size:
                continue
            
            # Collect sequences
            sequences = {key: [] for key in data.keys()}

            for start_idx in batch_chunk_starts:
                end_idx = start_idx + data_chunk_length

                # Collect sequences for each data type
                for key in data.keys():
                    if key not in ['actor_rnn_states', 'critic_rnn_states']:
                        sequences[key].append(data[key][start_idx:end_idx])

                # Get initial RNN states
                sequences['actor_rnn_states'].append(data['actor_rnn_states'][start_idx])
                sequences['critic_rnn_states'].append(data['critic_rnn_states'][start_idx])

            # Stack sequences into proper shapes (seq_len, batch_size, feat_dim) or (num_layers, batch_size, hidden_dim)
            batch = {
                key: torch.stack(sequences[key], dim=1)
                for key in sequences
            }

            # Yield the minibatch as a tuple
            yield ( 
                batch['obs'],
                batch['global_state'],
                batch['actor_rnn_states'],
                batch['critic_rnn_states'],
                batch['actions'],
                batch['values'],
                batch['returns'],
                batch['masks'],
                batch['old_action_log_probs'],
                batch['advantages'],
                batch['available_actions']
            )

