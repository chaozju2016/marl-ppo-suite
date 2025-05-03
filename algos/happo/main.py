
"""HAPPO algorithm."""
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Sequence, Tuple
import gymnasium

from algos.happo._agent import _HAPPOAgent
from buffers.rollout_storage import RolloutStorage
from networks.mappo_nets import Critic
from utils.scheduler import LinearScheduler
from utils.value_normalizers import create_value_normalizer
from utils.transform_tools import to_tensor


def slice(arr: Optional[torch.Tensor], idx: int) -> Optional[torch.Tensor]:
        """Select a column from a 2D array."""
        return None if arr is None else arr[:, idx]

class HAPPO:
    """
    Heterogeneous-Agent Proximal Policy Optimization (HAPPO) algorithm.

    This class implements the HAPPO algorithm for training multi-agent reinforcement learning models.
    """
    def __init__(self, args, obs_space, state_space, action_space, device=torch.device("cpu")):
        """
        Initialize HAPPO algorithm.

        Args:
            args: Arguments containing training hyperparameters
            obs_space: Observation space for individual agents (Box)
            state_space: Centralized observation space for critic (Box)
            action_space: Action space (Discrete)
            device (torch.device): Device to run the agent on
        """
        #Input Validation
        self._validate_inputs(args, obs_space, state_space, action_space)

        self.args = args
        self.device = device

        # Initialize core components
        self._init_hyperparameters()
        
        # Create a list of HAPPO agents
        self.happo_agents = [
            _HAPPOAgent(args, obs_space, action_space, device)
            for _ in range(self.n_agents)
        ]

        # Create Centralized Critic 
        if self.state_type == "AS":
            # Concatenate observation and state spaces for AS state type
            concat_dim = obs_space.shape[0] + state_space.shape[0]
            state_space = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(concat_dim,),
                dtype=np.float32
            )

        self.critic = Critic(
            self.args,
            state_space,
            self.device
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.lr,
            eps=self.args.optimizer_eps
        )

        if self.args.use_linear_lr_decay:
            self.scheduler = LinearScheduler(
                self.lr,
                self.args.min_lr,
                self.args.max_steps
            )
        
        if self.use_value_norm:
            self.value_normalizer = create_value_normalizer(
                normalizer_type=self.args.value_norm_type,
                device=device
            )
        else:
            self.value_normalizer = None

    def _validate_inputs(self, args, obs_space: gymnasium.spaces.Box,
                         state_space: gymnasium.spaces.Box, 
                        action_space: gymnasium.spaces.Discrete) -> None:
        """Validate input parameters."""
        if not isinstance(obs_space, gymnasium.spaces.Box):
            raise TypeError(f"Expected Box observation space, got {type(obs_space)}")
        if not isinstance(state_space, gymnasium.spaces.Box):
            raise TypeError(f"Expected Box state space, got {type(state_space)}")
        if not isinstance(action_space, gymnasium.spaces.Discrete):
            raise TypeError(f"Expected Discrete action space, got {type(action_space)}")
        
        required_attrs = ['n_agents', 'state_type', 'use_rnn', 'fixed_order', 'lr', 'clip_param', 'ppo_epoch',
                         'num_mini_batch']
        missing = [attr for attr in required_attrs if not hasattr(args, attr)]
        if missing:
            raise AttributeError(f"args missing required attributes: {missing}")

    def _init_hyperparameters(self):
        """Initialize training hyperparameters."""

        # Architecture Parameters
        self.n_agents = self.args.n_agents
        self.state_type = self.args.state_type # [FP, EP, AS]
        self.use_rnn = self.args.use_rnn
        self.fixed_order = self.args.fixed_order # Heterogeneous Agent specific - use fixed agent order for updates
        
        # PPO specific parameters
        self.clip_param = self.args.clip_param
        self.ppo_epoch = self.args.ppo_epoch
        self.num_mini_batch = self.args.num_mini_batch
        self.data_chunk_length = self.args.data_chunk_length
        self.entropy_coef = self.args.entropy_coef
        
        # Gradient parameters
        self.max_grad_norm = self.args.max_grad_norm
        self.use_max_grad_norm = self.args.use_max_grad_norm
        
        # Value function parameters
        self.use_clipped_value_loss = self.args.use_clipped_value_loss
        self.use_value_norm= self.args.use_value_norm
        self.use_huber_loss = self.args.use_huber_loss
        self.huber_delta = self.args.huber_delta

        # Learning parameters
        self.lr = self.args.lr
        self.gamma = self.args.gamma
        self.use_gae = self.args.use_gae
        self.gae_lambda = self.args.gae_lambda
        
    def get_actions(
            self,
            obs: torch.Tensor,
            rnn_states: torch.Tensor=None,
            masks: torch.Tensor=None,
            available_actions: torch.Tensor = None,
            deterministic: bool = False
        ):
        """
        Get actions for each agent.
        B = (n_rollout_threads, n_agents)

        Args:
            obs (torch.Tensor): Observation tensor #(B, obs_shape)
            rnn_states (torch.Tensor, optional): RNN states tensor. Required when RNN is enabled,
                can be None otherwise. #(B, num_layers hidden_size)
            masks (torch.Tensor, optional): Masks tensor. Required when RNN is enabled,
                can be None otherwise. #(B, 1)
            available_actions (torch.Tensor, optional): Available actions tensor #(B, n_actions)
            deterministic (bool): Whether to use deterministic actions

        Returns:
            actions (torch.Tensor): Actions tensor #(B, 1)
            action_log_probs (torch.Tensor): Action log probabilities tensor #(B, 1)
            rnn_states_out (torch.Tensor): Updated RNN states tensor #(B, num_layers, hidden_size)
                                        or None if RNN is disabled
        """

        # per agent forward pass
        outputs: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = [
            agent.get_actions(
                slice(obs, i),
                slice(rnn_states, i),
                slice(masks, i),
                slice(available_actions, i),
                deterministic,
            )
            for i, agent in enumerate(self.happo_agents)
        ]
        
        actions, log_probs, rnn_states_out = zip(*outputs)

        actions = torch.stack(actions, axis=1)  # (n_rollout_threads, n_agents, act_dim)
        log_probs = torch.stack(log_probs, axis=1) if not deterministic else None  # (n_rollout_threads, n_agents, 1)
        rnn_states_out = torch.stack(rnn_states_out, axis=1) if self.use_rnn else None # (n_rollout_threads, n_agents, num_layers, hidden_size)

        return actions, log_probs, rnn_states_out

    def get_values(self, 
                   state:torch.Tensor, 
                   obs:torch.Tensor, 
                   active_masks:torch.Tensor, 
                   rnn_states:torch.Tensor=None, 
                   masks:torch.Tensor=None):
        """
        Get values from the critic network.
        Batched version -> Batch(B) = (n_rollout_threads, n_agents) = n_rollout_threads * n_agents

        Args:
            state (torch.Tensor): State tensor #(B, state_shape)
            obs (torch.Tensor): Observation tensor #(B, obs_shape)
            active_masks (torch.Tensor): Active masks tensor #(B, 1) - used for death masking in (AS)
            rnn_states (torch.Tensor, optional): RNN states tensor. Required when RNN is enabled,
                                          can be None otherwise. #(B, n_layers, hidden_size)
            masks (torch.Tensor, optional): Masks tensor. Required when RNN is enabled,
                                     can be None otherwise. #(B, 1)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (values, rnn_states_out) where rnn_states_out is None
                                      if RNN is disabled
        """
        if self.state_type == "AS":
            # Concatenate observation and state spaces for AS state type
            state = state * active_masks # (batch_size, n_state) # Mask out inactive agents
            state = torch.cat([obs, state], dim=-1)  
    
        # Handle RNN states and masks based on whether RNN is enabled
        if self.use_rnn:
            if rnn_states is None or masks is None:
                raise ValueError("rnn_states and masks must be provided when RNN is enabled")

        # Get values and states
        values, rnn_states_out = self.critic(
            state,
            rnn_states,
            masks
        )
        return values, rnn_states_out
    
    def compute_value_loss(self, values, value_preds_batch, returns_batch):
        """
        Compute value function loss with normalization.

        Args:
            values: Current value predictions
            value_preds_batch: Old value predictions
            return_batch: Return targets
        """
        if self.use_value_norm:
            returns = self.value_normalizer.normalize(returns_batch, update=True)
            # values = self.value_normalizer.normalize(values, update=False)
            # value_preds_batch = self.value_normalizer.normalize(value_preds_batch, update=False)
        else:
            returns = returns_batch

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + torch.clamp(
                values - value_preds_batch,
                -self.clip_param,
                self.clip_param
            )
            if self.use_huber_loss:
                # Compute Huber loss for clipped and unclipped predictions
                value_losses = F.huber_loss(values, returns, delta=self.huber_delta, reduction='none')
                value_losses_clipped = F.huber_loss(value_pred_clipped, returns, delta=self.huber_delta, reduction='none')
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_losses = (values - returns).pow(2)
                value_losses_clipped = (value_pred_clipped - returns).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            if self.use_huber_loss:
                value_loss = F.huber_loss(values, returns, delta=self.huber_delta, reduction='mean')
            else:
                value_loss = 0.5 * (values - returns).pow(2).mean()

        return value_loss
    
    def update_critic(self, buffer:RolloutStorage):
        """
        Perform a training update using minibatches of data from the buffer.

        Args:
            buffer: (RolloutStorage) buffer containing training data related to critic

        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {
            'critic_loss': 0,
            'critic_grad_norm': 0,
        }

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
            for mini_batch in mini_batches:

                (obs_batch, global_state_batch, actor_h0_batch, critic_h0_batch,
                    actions_batch, values_batch, returns_batch, masks_batch, active_masks_batch,
                    old_action_log_probs_batch, advantages_batch, available_actions_batch) = mini_batch
                
                values, _ = self.get_values(global_state_batch, 
                                            obs_batch, 
                                            active_masks_batch, 
                                            critic_h0_batch, 
                                            masks_batch)
                
                # Critic loss
                critic_loss = self.compute_value_loss(values, values_batch, returns_batch)
                
                # Update Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_grad_norm = self._clip_gradients(self.critic)
                self.critic_optimizer.step()
                

                train_info['critic_loss'] += critic_loss.item()
                train_info['critic_grad_norm'] += critic_grad_norm
            
            num_updates = self.ppo_epoch * self.num_mini_batch
            for k in train_info.keys():
                train_info[k] /= num_updates

        return train_info

    def train(self, buffer:RolloutStorage):
        """
        Train the HAPPO policy using experiences from the buffer.
        """

        # Define factor, used for considering updates made by previous agents
        factor = torch.ones(buffer.n_steps, buffer.n_rollout_threads, 1).to(self.device)

        # Define agent order
        if self.fixed_order:
            agent_order = list(range(self.n_agents))
        else:
            agent_order = np.random.permutation(self.n_agents)

        # Define agent train infos
        agent_train_infos = {}

        # Train for each agent
        for agent_id in agent_order:
            agent_buffer = buffer.for_agent(agent_id)

            #compute action log probs before update 
            old_action_log_probs, _ = self.happo_agents[agent_id].evaluate_actions(
                to_tensor(agent_buffer.obs[:-1].reshape(-1, *agent_buffer.obs.shape[2:]), device=self.device), 
                to_tensor(agent_buffer.actions.reshape(-1, *agent_buffer.actions.shape[2:]), device=self.device), 
                to_tensor(agent_buffer.actor_rnn_states[0:1].reshape(-1, *agent_buffer.actor_rnn_states.shape[2:]), device=self.device) if self.use_rnn else None, 
                to_tensor(agent_buffer.masks[:-1].reshape(-1, *agent_buffer.masks.shape[2:]), device=self.device), 
                to_tensor(agent_buffer.available_actions[:-1].reshape(-1, *agent_buffer.available_actions.shape[2:]), device=self.device) if agent_buffer.available_actions is not None else None
            )

            agent_train_info = self.happo_agents[agent_id].train(agent_buffer, factor)

            # compute action log probs for updated agent
            new_actions_logprob, _ = self.happo_agents[agent_id].evaluate_actions(
                to_tensor(agent_buffer.obs[:-1].reshape(-1, *agent_buffer.obs.shape[2:]), device=self.device), 
                to_tensor(agent_buffer.actions.reshape(-1, *agent_buffer.actions.shape[2:]), device=self.device), 
                to_tensor(agent_buffer.actor_rnn_states[0:1].reshape(-1, *agent_buffer.actor_rnn_states.shape[2:]), device=self.device) if self.use_rnn else None, 
                to_tensor(agent_buffer.masks[:-1].reshape(-1, *agent_buffer.masks.shape[2:]), device=self.device), 
                to_tensor(agent_buffer.available_actions[:-1].reshape(-1, *agent_buffer.available_actions.shape[2:]), device=self.device) if agent_buffer.available_actions is not None else None
            )

            # update factor for next agent (could be multidiscrete - many actions so last dim multiplied)
            factor = factor * torch.prod(
                    torch.exp(new_actions_logprob - old_action_log_probs), dim=-1
                ).reshape(
                    buffer.n_steps,
                    buffer.n_rollout_threads,
                    1,
                )

            agent_train_infos[agent_id] = agent_train_info

        # Update critic
        critic_train_info = self.update_critic(buffer)

        return agent_train_infos, critic_train_info

    def _clip_gradients(self, network: nn.Module) -> Optional[float]:
        """Clip gradients and return gradient norm."""
        if self.use_max_grad_norm:
            return nn.utils.clip_grad_norm_(
                network.parameters(),
                self.max_grad_norm
            )
        return None

    def update_learning_rate(self, current_step):
        """Update the learning rate based on the current step."""
        lr_now = self.scheduler.get_lr(current_step)
        for agent in self.happo_agents:
            for p in agent.actor_optimizer.param_groups:
                p['lr'] = lr_now

        for p in self.critic_optimizer.param_groups:
            p['lr'] = lr_now

        return {
            'actor_lr': self.happo_agents[0].actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_optimizer.param_groups[0]['lr']
        }
    
    def save(self, save_path, save_args=False):
        """Save both actor and critic networks."""
        models_dict = {}

        for i, agent in enumerate(self.happo_agents):
            # Save actor and critic models
            models_dict[f'actor_{i}_state_dict'] = agent.actor.state_dict()
            models_dict[f'actor_{i}_optimizer_state_dict'] = agent.actor_optimizer.state_dict()

        models_dict['critic_state_dict'] = self.critic.state_dict()
        models_dict['critic_optimizer_state_dict'] = self.critic_optimizer.state_dict()
        torch.save(models_dict, save_path)

        # Save args separately
        if save_args:
            args_path = save_path + '.args'
            torch.save({'args': self.args}, args_path)

    def load(self, model_path):
        """Load both actor and critic networks."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        # Load actors 
        for i, agent in enumerate(self.happo_agents):
            agent.actor.load_state_dict(checkpoint[f'actor_{i}_state_dict'])
            agent.actor_optimizer.load_state_dict(checkpoint[f'actor_{i}_optimizer_state_dict'])

        # Load critic models
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        # Load args separately if they exist
        args_path = model_path + '.args'
        if os.path.exists(args_path):
            args_dict = torch.load(args_path, weights_only=False)
            self.args = args_dict['args']