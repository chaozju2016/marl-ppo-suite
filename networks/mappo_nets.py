"""
Unified Code for the networks of the mappo implementation:
- working with vectorized environments
- supporting both MLP and RNN networks
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from networks.modules.rnn import GRUModule

from utils.env_tools import get_shape_from_obs_space

def _orthogonal_init(layer, gain=1.0, bias_const=0.0):
    """Enhanced orthogonal initialization with configurable gain and bias."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.constant_(layer.bias, bias_const)
    elif isinstance(layer, nn.LayerNorm):
        nn.init.constant_(layer.weight, 1.0)
        nn.init.constant_(layer.bias, 0.0)

class Actor(nn.Module):
    """
    Actor network for MAPPO.
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize the actor network.

        Args:
            args (argparse.Namespace): Arguments containing training hyperparameters
            obs_space (gymnasium.spaces.Box): Observation space for individual agents (Box)
            action_space (gymnasium.spaces.Discrete): Action space (Discrete)
            device (torch.device): Device to run the agent on
        """

        super(Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.use_rnn = args.use_rnn
        # self.use_layer_norm = args.use_layer_norm
        self.rnn_layers = args.rnn_layers
        self.fc_layers = args.fc_layers
        self.use_feature_normalization = args.use_feature_normalization
        self.actor_gain = args.actor_gain

        obs_shape = get_shape_from_obs_space(obs_space)
        obs_dim = obs_shape[0]

        action_type = action_space.__class__.__name__
        if action_type == "Discrete":
            self.action_dim = action_space.n
        else:
            raise NotImplementedError("Only discrete action space is supported")

        # Feature Normalization
        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        # MLP Layers
        layers = []
        in_dim = obs_dim
        for _ in range(self.fc_layers):
            layers += [
                nn.Linear(in_dim, self.hidden_size),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_size),
            ]
            in_dim = self.hidden_size
        self.mlp = nn.Sequential(*layers)

        # RNN Layer (GRU)
        if self.use_rnn:
            self.rnn = GRUModule(self.hidden_size,
                                 self.hidden_size,
                                 num_layers=self.rnn_layers)

        # Output Layer
        self.output = nn.Linear(self.hidden_size, self.action_dim)

        self.apply(lambda module: _orthogonal_init(module, gain=nn.init.calculate_gain('relu')))
        _orthogonal_init(self.output, gain=self.actor_gain)


        self.to(device)

    def forward(self, x, rnn_states=None, masks=None):
        """
        Forward pass of the actor network.
        Batch size is n_agents * n_rollout_threads

        Args:
            x (torch.Tensor): Input tensor (batch_size, input_dim) or (seq_len, batch_size, input_dim)
            rnn_states (torch.Tensor, optional): RNN hidden state tensor. Required when use_rnn=True,
                                            ignored otherwise. Shape: (batch_size, num_layers, hidden_size)
            masks (torch.Tensor, optional): Mask tensor. Required when use_rnn=True, ignored otherwise.
                                       Shape: (batch_size, 1) or (seq_len, batch_size, 1)
        Returns:
            logits: action logits
            rnn_states_out: updated RNN states if use_rnn=True, None otherwise
        """
        if self.use_rnn and (rnn_states is None or masks is None):
            raise ValueError("rnn_states and masks must be provided when use_rnn=True")

        if self.use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)
        if self.use_rnn:
            x, rnn_states_out = self.rnn(x, rnn_states, masks)
        else:
            rnn_states_out = None
        logits = self.output(x)  # [seq_len, batch_size, action_dim]

        return logits, rnn_states_out

    def get_actions(self, obs, rnn_states=None, masks=None, available_actions=None, deterministic=False):
        """Get actions from the actor network.
        Batch size is n_agents * n_rollout_threads

        Args:
            obs: tensor of shape [batch_size, input_dim]
            rnn_states: tensor of shape [batch_size, num_layers, rnn_hidden_size],
                required when use_rnn=True, can be None otherwise
            masks: tensor of shape [batch_size, 1], required when use_rnn=True,
                can be None otherwise
            available_actions: tensor of shape [batch_size, action_dim]
            deterministic: bool, whether to use deterministic actions

        Returns:
            actions: tensor of shape [n_agents, 1]
            action_log_probs: tensor of shape [batch_size, 1]
            next_rnn_states: tensor of shape [batch_size, num_layers, rnn_hidden_size]
                if use_rnn=True, None otherwise
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
            action_dist = Categorical(logits=logits)
            actions = action_dist.sample().unsqueeze(-1) # (batch_size, 1)
            action_log_probs = action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1) # (batch_size, 1)

        return actions, action_log_probs, rnn_states_out

    def evaluate_actions(self, obs, actions, rnn_states=None, masks=None, available_actions=None):
        """Evaluate actions for training.

        Args:
            obs_seq: tensor of shape [seq_len, batch_size, input_dim] or [batch_size, input_dim]
            actions: tensor of shape [seq_len, batch_size, 1] or [batch_size, 1]
            rnn_states: tensor of shape [batch_size, num_layers, hidden_size] - initial hidden state
                required when use_rnn=True, can be None otherwise
            masks: tensor of shape [seq_len, batch_size, 1] or [batch_size, 1]
                required when use_rnn=True, can be None otherwise
            available_actions: tensor of shape [seq_len, batch_size, action_dim] or [batch_size, action_dim]
                can be None, when all actions are available
        Returns:
            action_log_probs: log probabilities of actions [seq_len,batch_size, 1] or [batch_size, 1]
            dist_entropy: entropy of action distribution [seq_len, batch_size, 1] or [batch_size, 1]
            rnn_states_out: updated RNN states [batch_size, num_layers, hidden_size] or None
        """
        logits, rnn_states_out = self.forward(obs, rnn_states, masks)

        if available_actions is not None:
            # Set unavailable actions to have a very small probability
            logits[available_actions == 0] = -1e10

        action_dist = Categorical(logits=logits)
        action_log_probs = action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1) # [seq_len, batch_size, 1]
        dist_entropy = action_dist.entropy().unsqueeze(-1) # [seq_len, batch_size, 1]

        return action_log_probs, dist_entropy, rnn_states_out

class Critic(nn.Module):
        """
        Critic network for MAPPO.
        """
        def __init__(self, args, centralized_obs_space, device=torch.device("cpu")):
            """Initialize the actor network.

            Args:
                args (argparse.Namespace): Arguments containing training hyperparameters
                centralized_obs_space (gymnasium.spaces.Box): Centralized observation space for critic (Box)
                device (torch.device): Device to run the agent on
            """
            super(Critic, self).__init__()
            self.hidden_size = args.hidden_size
            self.use_rnn = args.use_rnn
            self.use_feature_normalization = args.use_feature_normalization
            self.rnn_layers = args.rnn_layers
            self.fc_layers = args.fc_layers

            cent_obs_shape = get_shape_from_obs_space(centralized_obs_space)
            cent_obs_dim = cent_obs_shape[0]

            # Feature Normalization
            if self.use_feature_normalization:
                self.feature_norm = nn.LayerNorm(cent_obs_dim)

            # MLP Layers
            layers = []
            in_dim = cent_obs_dim
            for _ in range(self.fc_layers):
                layers += [
                    nn.Linear(in_dim, self.hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(self.hidden_size),
                ]
                in_dim = self.hidden_size
            self.mlp = nn.Sequential(*layers)

            # RNN Layer (GRU)
            if self.use_rnn:
                self.rnn = GRUModule(self.hidden_size,
                                     self.hidden_size,
                                     num_layers=self.rnn_layers)

            # Output Layer
            self.output = nn.Linear(self.hidden_size, 1)

            # Initialize weights
            self.apply(lambda module: _orthogonal_init(module, gain=nn.init.calculate_gain('relu')))
            _orthogonal_init(self.output, gain=1.0)

            self.to(device)

        def forward(self, x, rnn_states=None, masks=None):
            """Forward pass for critic network.

            Args:
                x (torch.Tensor): Input tensor (batch_size, input_dim) or (seq_len, batch_size, input_dim)
                rnn_states (torch.Tensor, optional): RNN hidden state tensor. Required when use_rnn=True,
                    ignored when use_rnn=False. Shape: (batch_size, num_layers, hidden_size)
                masks (torch.Tensor, optional): Mask tensor. Required when use_rnn=True,
                    ignored when use_rnn=False. Shape: (batch_size, 1) or (seq_len, batch_size, 1)
            Returns:
                values (torch.Tensor): Value predictions, shape (batch_size, 1) or (seq_len, batch_size, 1).
                rnn_states_out (torch.Tensor): updated RNN states if use_rnn=True, None otherwise
            """
            # Validate inputs when RNN is used
            if self.use_rnn and (rnn_states is None or masks is None):
                raise ValueError("rnn_states and masks must be provided when use_rnn=True")

            if self.use_feature_normalization:
                x = self.feature_norm(x)

            x = self.mlp(x)

            if self.use_rnn:
                x, rnn_states_out = self.rnn(x, rnn_states, masks)
            else:
                rnn_states_out = None

            values = self.output(x)  # [seq_len, batch_size, 1]

            return values, rnn_states_out