"""
Code fot simple single environment algos/mappo_rnn algorithm for SMAC experiments
- 
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

def _orthogonal_init(layer, gain=1.0, bias_const=0.0):
    """Enhanced orthogonal initialization with configurable gain and bias."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.constant_(layer.bias, bias_const)
    elif isinstance(layer, nn.LayerNorm):
        nn.init.constant_(layer.weight, 1.0)
        nn.init.constant_(layer.bias, 0.0)

class GRUModule(nn.Module):
    """Reusable GRU module for sequence processing with masking."""

    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        """
        Initialize the reusable GRU module.

        Args:
            input_dim (int): Dimension of the input features
            hidden_dim (int): Dimension of the hidden states
            num_layers (int, optional): Number of layers in the GRU. Defaults to 1.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(GRUModule, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Initialize GRU
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=False,# (seq_len, batch, input_size)
            dropout=dropout if num_layers > 1 else 0.0)

        # Layer norm after GRU
        self.gru_layer_norm = nn.LayerNorm(hidden_dim)

        # Initialize the weights with orthogonal initialization with higher gain
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=1.4)  # Higher gain for better gradient flow
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

        # Initialize layer norms
        nn.init.constant_(self.gru_layer_norm.weight, 1.0)
        nn.init.constant_(self.gru_layer_norm.bias, 0.0)

    def forward(self, x, rnn_states, masks):
        """
        Forward pass through the GRU module with masking.

        Args:
            x: (T, B, input_dim) or (B, input_dim) for single step
            rnn_states: (num_layers, B, hidden_dim)
            masks: (T, B, 1) or (B, 1) for single step
        """
        # Assert input shapes
        if x.dim() == 2:
            # Single step case: (B, input_dim)
            assert x.size(1) == self.input_dim, f"Expected input dimension {self.input_dim}, got {x.size(1)}"
            x = x.view(1, -1, self.input_dim) # (1, B, input_dim)
        else:
            # Multi-step case: (T, B, input_dim)
            assert x.dim() == 3, f"Expected 3D input (T, B, input_dim), got {x.dim()}D"
            assert x.size(2) == self.input_dim, f"Expected input dimension {self.input_dim}, got {x.size(2)}"

        # Assert RNN states shape
        assert rnn_states.size(0) == self.num_layers, f"Expected {self.num_layers} RNN layers, got {rnn_states.size(0)}"
        assert rnn_states.size(2) == self.hidden_dim, f"Expected hidden dimension {self.hidden_dim}, got {rnn_states.size(2)}"

        # Assert batch sizes match
        batch_size = x.size(1)
        assert rnn_states.size(1) == batch_size, f"Batch size mismatch: x has {batch_size}, rnn_states has {rnn_states.size(1)}"

        # Handle single timestep (evaluation/rollout) case
        is_single_step = x.size(0) == 1

        if is_single_step:
            # Apply mask to RNN states
            temp_states = (rnn_states * masks.view(1, -1, 1)).contiguous() # (num_layers, batch_size, hidden_size)
            x, rnn_states = self.gru(x, temp_states)
            x = x.squeeze(0) # (B, hidden_size)

            return self.gru_layer_norm(x), rnn_states

        # Handle Multi-step case (training)
        seq_len, batch_size = x.shape[:2]
        # Find sequences of zeros in masks for efficient processing
        masks = masks.view(seq_len, batch_size).contiguous() # (T, B, 1) -> (T, B)

        # Using trick from iKostrikov to process sequences in chunks.
        #
        # The trick works by:
        # 1. Finding timesteps where masks contain zeros (episode boundaries)
        # 2. Processing all timesteps between zeros as a single chunk
        # 3. Resetting RNN states at episode boundaries (where masks=0)
        # This avoids unnecessary sequential processing of each timestep
        has_zeros = ((masks[1:] == 0.0)
             .any(dim=-1) # (T-1)
             .nonzero() # (num_true, 1)
             .squeeze()) # (num_true)
        has_zeros = [has_zeros.item() + 1] if has_zeros.dim() == 0 else [(idx + 1) for idx in has_zeros.tolist()]
        has_zeros = [0] + has_zeros + [seq_len]

        # Process sequences between zero masks
        outputs = []
        for i in range(len(has_zeros) - 1):
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]

            # Apply mask to current RNN states (num_layers, batch, hidden_dim)
            temp_states = (rnn_states * masks[start_idx].view(1, -1, 1)).contiguous()

            # Process current sequence
            out, rnn_states = self.gru(x[start_idx:end_idx], temp_states)
            outputs.append(out)

        # Combine outputs and apply layer norm
        x = torch.cat(outputs, dim=0) # (T, B, hidden_dim)
        x = self.gru_layer_norm(x)

        return x, rnn_states


    def init_hidden(self, batch_size, device):
        """Initialize hidden states."""
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_dim,
            device=device, dtype=torch.float32
        )

class Actor_RNN(nn.Module):
    """
    Actor network for MAPPO.
    """
    def __init__(self, input_dim, action_dim, hidden_size, rnn_layers=1, use_feature_normalization=False, output_gain=0.01):
        """
        Initialize the actor network.

        Args:
            input_dim (int): Dimension of the input.
            action_dim (int): Dimension of the action.
            hidden_size (int): Hidden size of the network.
            rnn_layers (int): Number of RNN layers.
            use_feature_normalization (bool): Whether to use feature normalization.
            output_gain (float): Gain for the output layer.
        """
        super(Actor_RNN, self).__init__()

        self._use_feature_normalization = use_feature_normalization

        if self._use_feature_normalization:
            self.layer_norm = nn.LayerNorm(input_dim)

        # MLP layers before RNN
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
        )

        # RNN layer (GRU)
        self.gru = GRUModule(hidden_size, hidden_size, num_layers=rnn_layers)

        # Output layer
        self.output = nn.Linear(hidden_size, action_dim)

        # Initialize with specific gain
        gain  = nn.init.calculate_gain('relu')

        # Initialize MLP layers
        self.apply(lambda module: _orthogonal_init(module, gain=gain))

        # Initialize the output layer
        _orthogonal_init(self.output, gain=output_gain)

    def forward(self, x, rnn_states, masks):
        """
        Forward pass of the actor network.
        Args:
            x (torch.Tensor): Input tensor (num_agents, input_dim) or (seq_len, batch_size, input_dim)
            rnn_states (torch.Tensor): RNN hidden state tensor of shape (num_layers, num_agents, hidden_size) or (num_layers, batch_size, hidden_size)
            masks (torch.Tensor): Mask tensor of shape (num_agents, 1) or (seq_len, batch_size, 1)

        Returns:
            logits: action logits
            rnn_states_out: updated RNN states
        """

        if self._use_feature_normalization:
            x = self.layer_norm(x)

        x = self.mlp(x)
        x, rnn_states_out = self.gru(x, rnn_states, masks)
        logits = self.output(x)  # [seq_len, batch_size, action_dim]

        return logits, rnn_states_out

    def get_actions(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """Get actions from the actor network.

        Args:
            obs: tensor of shape [n_agents, input_dim]
            rnn_states: tensor of shape [num_layers, n_agents, rnn_hidden_size]
            masks: tensor of shape [n_agents, 1]
            available_actions: tensor of shape [n_agents, action_dim]
            deterministic: bool, whether to use deterministic actions

        Returns:
            actions: tensor of shape [n_agents, 1]
            action_log_probs: tensor of shape [n_agents, 1]
            next_rnn_states: tensor of shape [num_layers, n_agents, rnn_hidden_size]
        """
        # Forward pass to get logits
        logits, rnn_states_out = self.forward(obs, rnn_states, masks)

        # Apply mask for available actions if provided
        if available_actions is not None:
            # Set unavailable actions to have a very small probability
            logits[available_actions == 0] = -1e10

        if deterministic:
            actions = torch.argmax(logits, dim=-1, keepdim=True)
            action_log_probs = None
        else:
            # Convert logits to action probabilities
            # action_probs = F.softmax(logits, dim=-1)
            action_dist = Categorical(logits=logits)
            actions = action_dist.sample().unsqueeze(-1) # (n_agents, 1)
            action_log_probs = action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1) # (n_agents, 1)

        return actions, action_log_probs, rnn_states_out


    def evaluate_actions(self, obs_seq, rnn_states, masks_seq, actions_seq, available_actions_seq):
        """Evaluate actions for training.

        Args:
            obs_seq: tensor of shape [seq_len, batch_size, input_dim]
            rnn_states: tensor of shape [n_agents, hidden_size] - initial hidden state
            masks_seq: tensor of shape [seq_len, batch_size, 1]
            actions_seq: tensor of shape [seq_len, batch_size, 1]
            available_actions_seq: tensor of shape [seq_len, batch_size, action_dim]

        Returns:
            action_log_probs: log probabilities of actions [batch_size, seq_len, 1]
            dist_entropy: entropy of action distribution [batch_size, seq_len, 1]
            rnn_states_out: updated RNN states [num_layers, batch_size, hidden_size]
        """
        logits, rnn_states_out = self.forward(obs_seq, rnn_states, masks_seq)
        # [seq_len, batch_size, action_dim], [num_layers, batch_size, hidden_size]

        if available_actions_seq is not None:
            # Set unavailable actions to have a very small probability
            logits[available_actions_seq == 0] = -1e10

        action_dist = Categorical(logits=logits)
        action_log_probs = action_dist.log_prob(actions_seq.squeeze(-1)).unsqueeze(-1) # [seq_len, batch_size, 1]
        dist_entropy = action_dist.entropy().unsqueeze(-1) # [seq_len, batch_size, 1]

        return action_log_probs, dist_entropy, rnn_states_out


class Critic_RNN(nn.Module):
    """
    Critic network for MAPPO.
    """
    def __init__(self, input_dim, hidden_size, rnn_layers=1, use_feature_normalization=False):
        """
        Initialize the critic network.

        Args:
            input_dim (int): Dimension of the input.
            hidden_size (int): Hidden size of the network.
            rnn_layers (int): Number of RNN layers.
            use_feature_normalization (bool): Whether to use feature normalization
        """
        super(Critic_RNN, self).__init__()

        self._use_feature_normalization = use_feature_normalization

        if self._use_feature_normalization:
            self.layer_norm = nn.LayerNorm(input_dim)

        # MLP layers before RNN
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
        )

        # RNN layer (GRU)
        self.gru = GRUModule(hidden_size, hidden_size, num_layers=rnn_layers)

        # Output layer
        self.output = nn.Linear(hidden_size, 1)

        # Initialize with specific gains for each layer type
        gain = nn.init.calculate_gain('relu')

        # Initialize MLP layers
        self.apply(lambda module: _orthogonal_init(module, gain=gain))

        # Initialize the output layer
        _orthogonal_init(self.output, gain=gain)

    def forward(self, x, rnn_states, masks):
        """Forward pass for critic network.

        Args:
            x (torch.Tensor): Input tensor (num_agents, input_dim) or (seq_len, num_agents, input_dim)
            rnn_states (torch.Tensor): RNN hidden state tensor of shape (num_agents, hidden_size) or (batch_size, hidden_size)
            masks (torch.Tensor): Mask tensor of shape (num_agents, 1) or (seq_len, batch_size, 1)

        Returns:
            values (torch.Tensor): Value predictions, shape [n_agents, 1] or
                                  [seq_len, batch_size, 1].
            rnn_states_out (torch.Tensor): Updated RNN states, shape [num_layers, n_agents, hidden_size] or
                                          [num_layers, batch_size, hidden_size]
        """

        if self._use_feature_normalization:
            x = self.layer_norm(x)

        x = self.mlp(x)
        x, rnn_states_out = self.gru(x, rnn_states, masks)
        values = self.output(x)  # [seq_len, batch_size, 1]

        return values, rnn_states_out