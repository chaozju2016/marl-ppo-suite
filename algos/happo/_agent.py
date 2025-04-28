"""
HAPPO agent implementation.
"""
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium

from networks.mappo_nets import Actor

class _HAPPOAgent:
    """
    Internal PPO agent implementation used by the `HAPPO` (Heterogeneous-Agent PPO) algorithm.
    Each agent maintains its own policy and value networks, allowing for heterogeneous agent architectures.
    This class handles individual agent learning updates and should only be instantiated by the main HAPPO coordinator.

    Note:
        This is an internal implementation detail and should not be used directly.
        Instead, use the main `HAPPO` class which manages multiple `_HAPPOAgent` instances.
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """
        Initialize the HAPPO agent.

        Args:
            args (_type_): _description_
            obs_space (_type_): _description_
            action_space (_type_): _description_
            device (_type_, optional): _description_. Defaults to torch.device("cpu").
        """

        # Input validation
        self._validate_inputs(args, obs_space, action_space)

        self.device = device
        self.obs_space = obs_space
        self.action_space = action_space
        
        # Initialize hyperparameters
        self.use_rnn = args.use_rnn
        self.data_chunk_length = args.data_chunk_length
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.use_max_grad_norm = args.use_max_grad_norm

        self.actor = Actor(
            args,
            obs_space,
            action_space,
            device
        )

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=args.lr,
            eps=args.optimizer_eps
        )
    

    def _validate_inputs(self, args, obs_space: gymnasium.spaces.Box, 
                        action_space: gymnasium.spaces.Discrete) -> None:
        """Validate input parameters."""
        if not isinstance(obs_space, gymnasium.spaces.Box):
            raise TypeError(f"Expected Box observation space, got {type(obs_space)}")
        if not isinstance(action_space, gymnasium.spaces.Discrete):
            raise TypeError(f"Expected Discrete action space, got {type(action_space)}")

        obs_dim, action_dim = obs_space.shape[0], action_space.n
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError(f"Invalid dimensions: obs={obs_dim}, action={action_dim}")
        
        required_attrs = ['use_rnn', 'lr', 'clip_param', 'ppo_epoch','num_mini_batch']
        missing = [attr for attr in required_attrs if not hasattr(args, attr)]
        if missing:
            raise AttributeError(f"args missing required attributes: {missing}")
        
    def get_actions(self, 
                    obs:torch.Tensor, 
                    rnn_states:torch.Tensor=None, 
                    masks:torch.Tensor=None, 
                    available_actions:torch.Tensor=None, 
                    deterministic: bool=False):
        """
        Get actions from the policy network.
        Batched version -> Batch(B) = (n_rollout_threads) = n_rollout_threads

        Args:
            obs (torch.Tensor): Observation tensor #(batch_size, obs_shape)
            rnn_states (torch.Tensor, optional): RNN states tensor. Required when RNN is enabled,
                can be None otherwise. #(batch_size, num_layers hidden_size)
            masks (torch.Tensor, optional): Masks tensor. Required when RNN is enabled,
                can be None otherwise. #(batch_size, 1)
            available_actions (np.ndarray, optional): Available actions tensor #(batch_size, n_actions)
            deterministic (bool): Whether to use deterministic actions

        Returns:
            actions (torch.Tensor): Actions tensor #(batch_size, 1)
            action_log_probs (torch.Tensor): Action log probabilities tensor #(batch_size, 1)
            rnn_states_out (torch.Tensor): Updated RNN states tensor #(batch_size, num_layers, hidden_size)
                                        or None if RNN is disabled
        """
        with torch.no_grad():
            # Handle RNN states and masks based on whether RNN is enabled
            if self.use_rnn:
                if rnn_states is None or masks is None:
                    raise ValueError("rnn_states and masks must be provided when RNN is enabled")
            
            # Get actions
            actions, action_log_probs, rnn_states_out = self.actor.get_actions(
                obs, rnn_states, masks, available_actions, deterministic
            )

        return actions, action_log_probs, rnn_states_out
    
    def evaluate_actions(self, 
                         obs:torch.Tensor, 
                         actions:torch.Tensor,
                         actor_h0:torch.Tensor=None, 
                         masks:torch.Tensor=None,
                         available_actions:torch.Tensor=None):
        """
        Evaluate actions for a training.

        Args:
            state (torch.Tensor): State tensor #(seq_len, batch_size, n_state) or #(batch_size, n_state)
            obs (torch.Tensor): Observation tensor #(seq_len, batch_size, n_obs) or #(batch_size, n_obs)
            actions (torch.Tensor): Actions tensor #(seq_len, batch_size, 1) or #(batch_size, 1)
            actor_h0 (torch.Tensor): Initial actor RNN states tensor #(num_layers, batch_size, hidden_size) or None
            masks (torch.Tensor): Masks tensor #(seq_len, batch_size, 1) or #(batch_size, 1)
            available_actions (torch.Tensor): Available actions tensor #(seq_len, batch_size, action_dim) or #(batch_size, action_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (action_log_probs, dist_entropy)
        """
        action_log_probs, dist_entropy, _ = self.actor.evaluate_actions(
            obs,
            actions,
            actor_h0,
            masks,
            available_actions)

        return action_log_probs, dist_entropy    
    
    def update(self, mini_batch, factor_batch:torch.Tensor):
        """
        Update policy using a mini-batch of experiences.

        Args:
            mini_batch (dict): Dictionary containing mini-batch data
            factor_batch (torch.Tensor): Factor tensor #(batch_size, 1) (default: =1.0)

        Returns:
            tuple: (policy_loss, dist_entropy)
        """
        metrics = {}

        # Extract data from mini-batch
        (obs_batch, actor_h0_batch,
            actions_batch, masks_batch, active_masks_batch,
            old_action_log_probs_batch, advantages_batch, available_actions_batch) = mini_batch

        # Evaluate actions
        action_log_probs, dist_entropy = self.evaluate_actions(
            obs_batch, actions_batch, actor_h0_batch, masks_batch, available_actions_batch)
        
        # Calculate PPO ratio and KL divergence
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
        clip_ratio = (torch.abs(ratio - 1) > self.clip_param).float().mean().item()

        # Actor Loss
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        # We do sum across action dimension first (required for multi-discrete)
        policy_loss = -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        entropy_loss = -self.entropy_coef * torch.mean(dist_entropy)
        actor_loss = policy_loss + entropy_loss

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = self._clip_gradients(self.actor)
        self.actor_optimizer.step()

        # Update metrics
        metrics.update({
            'actor_loss': actor_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'approx_kl': approx_kl,
            'clip_ratio': clip_ratio,
            'actor_grad_norm': actor_grad_norm
        })

        return metrics
    
    def train(self, buffer, factor=None):
        """
        Perform a training update using minibatches of data from the buffer.
        
        Args:
            buffer: (AgentRolloutView) buffer containing training data related to agent
            factor: (torch.Tensor) factor used for considering updates made by previous agents (default: None)
        
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """

        train_info = {
            'actor_loss': 0,
            'entropy_loss': 0,
            'approx_kl': 0,
            'clip_ratio': 0,
            'actor_grad_norm': 0,
        }

        if factor is None:
            factor = torch.ones(self.buffer.n_steps, self.buffer.n_rollout_threads, 1).to(self.device)
        
        # Flatten factor 
        factor_flat = factor.view(-1, 1)

        # Train for ppo_epoch iterations
        for _ in range(self.ppo_epoch):
            
            if self.use_rnn:
                # Generate mini-batches
                mini_batches = buffer.get_minibatches_seq_first(
                    self.num_mini_batch,
                    data_chunk_length=self.data_chunk_length)
            else:
                # Generate mini-batches without RNN
                mini_batches = buffer.get_minibatches(self.num_mini_batch)

            # Update for each mini-batch
            for mini_batch, mb_idx in mini_batches:
                  
                # Get batch factor
                batch_factor = factor_flat[mb_idx].detach().clone() # (T*N_mini, 1)
                metrics = self.update(mini_batch, batch_factor)

                # Update training info
                for k, v in metrics.items():
                    if k in train_info:
                        train_info[k] += v

        # Calculate means
        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info


    def _clip_gradients(self, network: nn.Module) -> Optional[float]:
        """Clip gradients and return gradient norm."""
        if self.use_max_grad_norm:
            return nn.utils.clip_grad_norm_(
                network.parameters(),
                self.max_grad_norm
            )
        return None