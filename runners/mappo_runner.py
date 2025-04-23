import os
import time
import numpy as np
import torch

from buffers.rollout_storage import RolloutStorage
from envs import make_vec_envs
from algos.mappo import MAPPO


from utils.logger import Logger
from utils.reward_normalization_new import StandardNormalizer, EMANormalizer
from utils.transform_tools import flatten_first_dims, unflatten_first_dim

# import wandb
# TODO: Add use case for the wandb
def normalise_shared_reward(rew: np.ndarray, norm):
    """
    rew : (n_env, n_agents, 1) – identical values along the agent axis
    norm: StandardNormalizer or EMANormalizer
    """
    r_env  = rew[:, 0, 0]
    r_norm = norm.normalize(r_env)[:, None, None] #  # (n_env,1,1) add broadcast dims
    return np.broadcast_to(r_norm, rew.shape) # view, no copy

class MAPPORunner:
    """
    Runner class to handle environment interactions and training for MAPPO with agent-specific states.

    This class manages the environment with agent-specific states, agent, buffer, and training process,
    collecting trajectories and updating the policy.
    """
    def __init__(self, args):
        """
        Initialize the runner.

        Args:
            args: Arguments containing training parameters
        """
        self.args = args

        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

        # Create training environment using the factory function
        self.envs = make_vec_envs(args, is_eval=False, num_processes=args.n_rollout_threads)
        self.eval_envs = make_vec_envs(args, is_eval=True, num_processes=args.n_eval_rollout_threads)

        # Store args for creating evaluation environment later
        self.args = args

        # Get environment info
        print(f"n_agents: {self.envs.n_agents}, "
              f"action_space: {self.envs.action_space}, "
              f"state_space: {self.envs.share_observation_space}, "
              f"obs_space: {self.envs.observation_space}, "
              f"episode_limit: {self.envs.episode_limit}")

        # Create agent
        self.agent = MAPPO(args,
                            self.envs.observation_space,
                            self.envs.share_observation_space,
                            self.envs.action_space,
                            self.device)

        # Create buffer
        self.buffer = RolloutStorage(
            args,
            self.envs.n_agents,
            self.envs.observation_space,
            self.envs.action_space,
            self.envs.share_observation_space,
            self.device)

        # Normalize rewards
        if args.use_reward_norm:
            if args.reward_norm_type == "ema":
                self.reward_norm = EMANormalizer(clip=None)
            else:
                self.reward_norm = StandardNormalizer(clip=None)

        # Initialize logger
        run_name = (
            f"lr{args.lr}_nenvs{args.n_rollout_threads}_nsteps{args.n_steps}_"
            f"minibatch{args.num_mini_batch}_epochs{args.ppo_epoch}_"
            f"gamma{args.gamma}_gae{args.gae_lambda}_"
            f"clip{args.clip_param}_state{args.state_type}_"
            f"rnn{args.use_rnn}_{int(time.time())}"
        )

        run_name = "".join(run_name)
        env_name = "sc2_" + args.map_name
        self.logger = Logger(run_name=run_name, env=env_name, algo="MAPPO")

        # Log hyperparameters
        self.logger.log_hyperparameters({
            "env_name": env_name,
            "map_name": args.map_name,
            "difficulty": args.difficulty,
            "use_rnn": args.use_rnn,
            "state_type": args.state_type,
            "use_fp_wrapper": args.use_fp_wrapper,
            "use_agent_specific_state": args.use_agent_specific_state,
            "lr": args.lr,
            "optimizer_eps": args.optimizer_eps,
            "use_linear_lr_decay": args.use_linear_lr_decay,
            "use_feature_normalization": args.use_feature_normalization,
            "use_value_norm": args.use_value_norm,
            "use_reward_norm": args.use_reward_norm,
            "actor_gain": args.actor_gain,
            "hidden_size": args.hidden_size,
            "rnn_layers": args.rnn_layers,
            "n_steps": args.n_steps,
            "data_chunk_length": args.data_chunk_length,
            "num_mini_batch": args.num_mini_batch,
            "ppo_epoch": args.ppo_epoch,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_param": args.clip_param,
            "use_clipped_value_loss": args.use_clipped_value_loss,
            "use_huber_loss": args.use_huber_loss,
            "huber_delta": args.huber_delta,
            "entropy_coef": args.entropy_coef,
            "use_gae": args.use_gae,
            "use_proper_time_limits": args.use_proper_time_limits,
            "use_max_grad_norm": args.use_max_grad_norm,
            "max_grad_norm": args.max_grad_norm,
            "use_eval": args.use_eval,
            "eval_interval": args.eval_interval,
            "eval_episodes": args.eval_episodes
        })


    def run(self):
        """
        Run the training process.
        """
        # Training stats
        self.total_steps = 0
        self.best_win_rate = 0
        self.last_battles_game = np.zeros((self.args.n_rollout_threads), dtype=np.float32)
        self.last_battles_won = np.zeros((self.args.n_rollout_threads), dtype=np.float32)
        self.episode_rewards = np.zeros((self.args.n_rollout_threads), dtype=np.float32)
        self.episode_length = np.zeros((self.args.n_rollout_threads), dtype=np.float32)
        evaluate_num = -1

        # Warmup
        self.warmup()

        # Training loop
        while self.total_steps < self.args.max_steps:
            # Decay learning rate
            if self.args.use_linear_lr_decay:
                self.agent.update_learning_rate(self.total_steps)

            # Evaluate agent
            if self.total_steps // self.args.eval_interval > evaluate_num:
                self.evaluate(self.args.eval_episodes)
                evaluate_num += 1

            # Collect trajectories
            last_infos, rollout_data = self.collect_rollouts()
            self.total_steps += self.args.n_steps * self.args.n_rollout_threads

            # Compute returns and advantages
            self.compute_returns()

            # Train agent
            if self.args.show_performance_metrics:
                train_start_time = time.time()
                train_info = self.agent.train(self.buffer)
                train_end_time = time.time()
                train_duration = train_end_time - train_start_time

                # Log training performance
                print(f"\nTraining Performance Metrics:")
                print(f"  Time to train on {self.args.n_steps} steps: {train_duration:.2f} seconds")
                print(f"  Training time per step: {train_duration/self.args.n_steps*1000:.2f} ms")
            else:
                train_info = self.agent.train(self.buffer)

            # Log training information
            self._log_rollout_outcome(last_infos, rollout_data, train_info, self.total_steps)

            # Reset buffer for next rollout
            self.buffer.after_update()

        # Final evaluation
        if self.args.use_eval:
            print(f"Final evaluation at {self.total_steps}/{self.args.max_steps}")
            self.evaluate(self.args.eval_episodes)

        self.envs.close()
        self.eval_envs.close()
        self.logger.close()

    def warmup(self):
        """
        Warmup the agent.
        """
        # Reset environment
        obs, states, available_actions = self.envs.reset()

        # Store initial observations and states
        self.buffer.obs[0] = np.array(obs)  # (rollout_threads, n_agents, obs_shape)
        self.buffer.global_state[0] = np.array(states)  # (rollout_threads, n_agents, state_shape)
        self.buffer.available_actions[0] = np.array(available_actions)  # (rollout_threads, n_agents, n_actions)

    def collect_rollouts(self):
        """
        Collect trajectories by interacting with the environment.

        Returns:
            np.ndarray: Information from the last step of the rollout
        """
        # Start timing for rollout collection if performance metrics are enabled
        rollout_start_time = time.time() if self.args.show_performance_metrics else None

        rollout_data = {
            'episode_lengths': [],
            'episode_rewards': []
        }

        for step in range(self.args.n_steps):
            # Prepare basic observations
            flatten_obs = flatten_first_dims(self.buffer.obs[step])  # (n_rollout_threads*n_agents, obs_shape)
            flatten_share_obs = flatten_first_dims(self.buffer.global_state[step])  # (n_rollout_threads*n_agents, state_shape)
            flatten_masks = flatten_first_dims(self.buffer.masks[step])  # (n_rollout_threads*n_agents, 1)

            # Handle available actions if present
            flatten_available_actions = (
                flatten_first_dims(self.buffer.available_actions[step])
                if self.buffer.available_actions is not None
                else None
            )

            # Prepare RNN states if using RNN
            if self.args.use_rnn:
                flatten_actor_rnn_states = flatten_first_dims(self.buffer.actor_rnn_states[step])
                flatten_critic_rnn_states = flatten_first_dims(self.buffer.critic_rnn_states[step])
            else:
                flatten_actor_rnn_states = None
                flatten_critic_rnn_states = None

            # Get actions and values
            actions, action_log_probs, actor_rnn_states = self.agent.get_actions(
                flatten_obs,
                flatten_actor_rnn_states,
                flatten_masks,
                flatten_available_actions,
                deterministic=False
            )

            values, critic_rnn_states = self.agent.get_values(
                flatten_share_obs,
                flatten_critic_rnn_states,
                flatten_masks
            )

            # Reshape actions and values
            shape = (self.args.n_rollout_threads, self.envs.n_agents)
            actions = unflatten_first_dim(actions, shape)
            action_log_probs = unflatten_first_dim(action_log_probs, shape)
            values = unflatten_first_dim(values, shape)

            # Reshape RNN states if using RNN
            if self.args.use_rnn:
                actor_rnn_states = unflatten_first_dim(actor_rnn_states, shape)
                critic_rnn_states = unflatten_first_dim(critic_rnn_states, shape)
            else:
                actor_rnn_states = None
                critic_rnn_states = None

            # Execute actions in environment
            obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
            # obs: (n_threads, n_agents, obs_dim)
            # share_obs: (n_threads, n_agents, share_obs_dim)
            # rewards: (n_threads, n_agents, 1)
            # dones: (n_threads, n_agents)
            # infos: (n_threads)
            # available_actions: None or (n_threads, n_agents, action_number)

            # Update episode stats
            self.episode_length += 1
            self.episode_rewards += rewards[:, 0, 0]

            # Normalize rewards if enabled
            if self.args.use_reward_norm:
                rewards = normalise_shared_reward(rewards, self.reward_norm)

            # Handle episode termination
            done_envs = np.all(dones, axis=1)
            if np.any(done_envs):
                self._check_episode_outcome(done_envs, self.total_steps + step*self.args.n_rollout_threads)
                # done_indices = np.where(done_envs)[0]
                # rollout_data['episode_lengths'].extend(self.episode_length[done_indices].tolist())
                # rollout_data['episode_rewards'].extend(self.episode_rewards[done_indices].tolist())
                # self.episode_length[done_indices] = 0
                # self.episode_rewards[done_indices] = 0
                      

            # Insert collected data
            data = (
                obs, share_obs, rewards, dones,
                infos, available_actions, values, actions,
                action_log_probs, actor_rnn_states, critic_rnn_states,
            )
            self.insert(data)

        # End timing and calculate statistics if performance metrics are enabled
        if self.args.show_performance_metrics and rollout_start_time is not None:
            rollout_end_time = time.time()
            rollout_duration = rollout_end_time - rollout_start_time
            steps_per_second = self.args.n_steps * self.args.n_rollout_threads / rollout_duration
            experiences_per_second = self.args.n_steps * self.args.n_rollout_threads * self.envs.n_agents / rollout_duration

            # Log performance metrics
            print(f"\nRollout Performance Metrics:")
            print(f"  Time to collect {self.args.n_steps} steps: {rollout_duration:.2f} seconds")
            print(f"  Steps per second: {steps_per_second:.2f}")
            print(f"  Experiences per second: {experiences_per_second:.2f}")
            print(f"  Number of environments: {self.args.n_rollout_threads}")

        return infos, rollout_data

    def insert(self, data):
        """
        Insert a new transition into the buffer.

        Args:
           data (tuple): Transition data containing:
            - obs: Agent observations (n_rollout_threads, n_agents, obs_dim)
            - share_obs: Shared observations (n_rollout_threads, n_agents, share_obs_dim)
            - rewards: Agent rewards (n_rollout_threads, n_agents, 1)
            - dones: Done flags (n_rollout_threads, n_agents)
            - infos: Environment info dicts [n_rollout_threads]
            - available_actions: Available actions mask (n_rollout_threads, n_agents, action_dim) or None
            - values: Value estimates (n_rollout_threads, n_agents, 1)
            - actions: Taken actions (n_rollout_threads, n_agents, action_dim)
            - action_log_probs: Action log probs (n_rollout_threads, n_agents, action_dim)
            - actor_rnn_states: Actor RNN states (n_rollout_threads, n_agents, num_layers, hidden_size)
            - critic_rnn_states: Critic RNN states (n_rollout_threads, n_agents, num_layers, hidden_size)
        """
        # Unpack transition data
        (obs, share_obs, rewards, dones, infos, available_actions,
        values, actions, action_log_probs, actor_rnn_states, critic_rnn_states) = data

        # Handle episode terminations
        done_envs = np.all(dones, axis=1)  # Check which environments are done
        # n_done_envs = done_envs.sum()
        done_env_mask = done_envs == True

        # Reset RNN states for done environments
        if self.args.use_rnn:
            actor_rnn_states[done_env_mask] = 0.0
            critic_rnn_states[done_env_mask] = 0.0

        # Create masks
        shape = (self.args.n_rollout_threads, self.envs.n_agents, 1)
        masks = np.ones(shape, dtype=np.float32)
        active_masks = np.ones(shape, dtype=np.float32)

        # Update masks for done environments and agents
        masks[done_env_mask, :, :] = 0.0 # broadcast across agents and last dim
        active_masks[dones, :] = 0.0 # for individual agent deaths
        active_masks[done_env_mask, : , :] = 1.0 # for full environment termination

        # Create truncation masks from environment infos
        truncates = np.array([[info['truncated']] for info in infos]) # (n_rollout_threads, 1)
        truncates = np.repeat(truncates, self.envs.n_agents, axis=1).reshape(*shape) # (n_rollout_threads, n_agents, 1)

        # Store trajectory in buffer
        self.buffer.insert(
            obs=obs,  # (n_rollout_threads, n_agents, n_obs)
            global_state=share_obs,  # (n_rollout_threads, n_agents, n_state)
            actions=actions,  # (n_rollout_threads, n_agents, 1)
            action_log_probs=action_log_probs,  # (n_rollout_threads, n_agents, 1)
            values=values,  # (n_rollout_threads, n_agents, 1)
            rewards=rewards,  # (n_rollout_threads, n_agents, 1)
            masks=masks,  # (n_rollout_threads, n_agents, 1)
            active_masks=active_masks,  # (n_rollout_threads, n_agents, 1)
            truncates=truncates,  # (n_rollout_threads, n_agents, 1)
            available_actions=available_actions,  # (n_rollout_threads, n_agents, n_actions) or None
            actor_rnn_states=actor_rnn_states,  # (num_layers, n_rollout_threads, n_agents, hidden_size)
            critic_rnn_states=critic_rnn_states,  # (num_layers, n_rollout_threads, n_agents, hidden_size)
        )

    def compute_returns(self):
        """
        Compute returns and advantages for the collected trajectories.
        """
        next_value, _ = self.agent.get_values(
            flatten_first_dims(self.buffer.global_state[-1]),
            flatten_first_dims(self.buffer.critic_rnn_states[-1]) if self.args.use_rnn else None,
            flatten_first_dims(self.buffer.masks[-1]))

        self.buffer.compute_returns_and_advantages(
            unflatten_first_dim(next_value, (self.args.n_rollout_threads, self.envs.n_agents)),
            self.args.gamma,
            self.args.gae_lambda)

    def _check_episode_outcome(self, done_envs, current_step):
        """
        Check the outcome of the episode, and log the results.

        Args:
            done_envs (np.ndarray): Array of booleans indicating whether each environment is done
            current_step (int): Current step in the rollout
        """
        done_indices = np.where(done_envs)[0]

        # Extract episode lengths and rewards for done environments
        episode_lens = self.episode_length[done_indices]
        episode_rews = self.episode_rewards[done_indices]
        
        # Reset episode lengths and rewards for done environments
        self.episode_length[done_indices] = 0
        self.episode_rewards[done_indices] = 0
        
        # Log metrics
        self.logger.add_scalar('train/length', np.mean(episode_lens), current_step)
        self.logger.add_scalar('train/rewards', np.mean(episode_rews), current_step)

    def _log_rollout_outcome(self, last_infos, rollout_data, train_info, current_step):
        """
        Log the outcome of the rollout.

        Args:
            last_infos (list): List of dictionaries containing environment information for each thread
            rollout_data (dict): Dictionary containing episode-specific data
            train_info (dict): Dictionary containing training information
            current_step (int): Current step in the rollout
        """
        # Log episode-specific data
        # self.logger.add_scalar('train/length', np.mean(rollout_data['episode_lengths']), current_step)
        # self.logger.add_scalar('train/rewards', np.mean(rollout_data['episode_rewards']), current_step)

        battles_won = []
        battles_game = []
        incre_battles_won = []
        incre_battles_game = []

        for i, info in enumerate(last_infos):
            if "battles_won" in info.keys():
                battles_won.append(info["battles_won"])
                incre_battles_won.append(
                    info["battles_won"] - self.last_battles_won[i]
                )
            if "battles_game" in info.keys():
                battles_game.append(info["battles_game"])
                incre_battles_game.append(
                    info["battles_game"] - self.last_battles_game[i]
                )

        win_rate = (
            np.sum(incre_battles_won) / np.sum(incre_battles_game)
            if np.sum(incre_battles_game) > 0
            else 0.0
        )
        self.logger.add_scalar("train/win_rate", win_rate, current_step)

        self.last_battles_game = battles_game
        self.last_battles_won = battles_won
        self.logger.log_training(
                {
                    "train/critic_loss":  train_info['critic_loss'],
                    "train/actor_loss": train_info['actor_loss'],
                    "train/entropy_loss": train_info['entropy_loss'],
                    "train/approx_kl": train_info['approx_kl'],
                    "train/clip_ratio": train_info['clip_ratio'],
                    "train/actor_grad_norm": train_info['actor_grad_norm'],
                    "train/critic_grad_norm": train_info['critic_grad_norm'],
                }
            )

    @torch.no_grad()
    def evaluate(self, num_episodes=10):
        """
        Evaluate the current policy, using vec envs.

        Args:
            num_episodes (int): Number of episodes to evaluate

        Returns:
            tuple: (mean_rewards, win_rate)
        """

        # Evaluation stats
        all_episode_rewards = []
        all_episode_lengths = []
        all_win_rates = []

        obs, _, available_actions = self.eval_envs.reset()

        # Episode tracking
        episode_rewards = np.zeros((self.args.n_eval_rollout_threads), dtype=np.float32)
        episode_length = np.zeros((self.args.n_eval_rollout_threads), dtype=np.float32)

        # Initialize RNN states
        if self.args.use_rnn:
            eval_rnn_states = np.zeros(
                (
                    self.args.n_eval_rollout_threads,
                    self.eval_envs.n_agents,
                    self.args.rnn_layers,
                    self.args.hidden_size
                ),
                dtype=np.float32)
        else:
            eval_rnn_states = None

        # Initialize masks
        eval_masks = np.ones(
            (self.args.n_eval_rollout_threads, self.eval_envs.n_agents, 1),
            dtype=np.float32)

        while True:
            flatten_obs = flatten_first_dims(obs)  # (n_rollout_threads*n_agents, obs_shape)
            flatten_masks = flatten_first_dims(eval_masks)  # (n_rollout_threads*n_agents, 1)
            # Handle available actions if present
            flatten_available_actions = (
                flatten_first_dims(available_actions)
                if available_actions is not None
                else None
            )
            # Prepare RNN states if using RNN
            flatten_eval_rnn_states = flatten_first_dims(eval_rnn_states) if self.args.use_rnn else None

            # Get actions
            actions, _, eval_rnn_states = self.agent.get_actions(
                flatten_obs,
                flatten_eval_rnn_states,
                flatten_masks,
                flatten_available_actions,
                deterministic=True
            )
            # Reshape actions and values
            shape = (self.args.n_eval_rollout_threads, self.eval_envs.n_agents)
            actions = unflatten_first_dim(actions, shape)
            eval_rnn_states = unflatten_first_dim(eval_rnn_states, shape) if self.args.use_rnn else None

            # Execute actions in environment
            obs, share_obs, rewards, dones, infos, available_actions = self.eval_envs.step(actions)

            # Update episode stats
            episode_rewards += rewards[:, 0, 0]
            episode_length += 1

            # Handle episode termination
            done_envs = np.all(dones, axis=1)
            done_env_mask = done_envs == True

            # Reset RNN states and masks for done environments
            if self.args.use_rnn:
                eval_rnn_states[done_env_mask] = 0.0
            eval_masks = np.ones(
                (self.args.n_eval_rollout_threads, self.eval_envs.n_agents, 1),
                dtype=np.float32,
            )
            eval_masks[done_env_mask] = 0.0

            # Update episode stats
            for i in range(self.args.n_eval_rollout_threads):
                if done_envs[i]:
                    all_episode_rewards.append(episode_rewards[i])
                    all_episode_lengths.append(episode_length[i])
                    episode_rewards[i] = 0
                    episode_length[i] = 0
                    # Check if episode was won
                    all_win_rates.append(infos[i]["battle_won"])


            if len(all_episode_rewards) >= num_episodes:
                break

        # Calculate statistics
        mean_rewards = np.mean(all_episode_rewards)
        mean_length = np.mean(all_episode_lengths)
        win_rate = np.mean(all_win_rates)

        # Log evaluation stats
        self.logger.add_scalar('eval/rewards', mean_rewards, self.total_steps)
        self.logger.add_scalar('eval/win_rate', win_rate, self.total_steps)
        self.logger.add_scalar('eval/length', mean_length, self.total_steps)
        print(f"{self.total_steps}/{self.args.max_steps} Evaluation: Mean rewards: {mean_rewards:.2f},  Mean length: {mean_length:.2f}, Win rate: {win_rate:.2f}")

        # Update best win rate
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            save_path = os.path.join(self.logger.dir_name, f"best-torch.model")
            self.agent.save(save_path)
            print(f"Saved best model with win rate {win_rate:.2f} to {save_path}")

        return mean_rewards, win_rate