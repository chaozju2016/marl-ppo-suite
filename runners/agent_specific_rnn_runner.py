import os
import time
import numpy as np
import torch
from buffers.as_rnn_buffer import RecurrentRolloutStorage
from algos.mappo_rnn import RMAPPOAgent
from utils.logger import Logger
from utils.reward_normalization import EfficientStandardNormalizer, EMANormalizer
# import wandb


class AgentSpecificRecurrentRunner:
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
        from envs import create_env
        self.env = create_env(args, is_eval=False)

        # Store args for creating evaluation environment later
        self.args = args
        self.evaluate_env = None

        # Set environment parameters
        self.args.n_agents = self.env.n_agents
        self.args.obs_dim = self.env.get_obs_size()
        self.args.state_dim = self.env.get_state_size()
        self.args.action_dim = self.env.n_actions

        # Create agent
        self.agent = RMAPPOAgent(args, args.obs_dim, args.state_dim, args.action_dim, self.device)

        # Create buffer
        self.buffer = RecurrentRolloutStorage(args.n_steps,
            args.n_agents,
            args.obs_dim,
            args.action_dim,
            args.state_dim,
            args.hidden_size,
            args.rnn_layers,
            agent_specific_global_state=args.use_agent_specific_state
            )

        # Normalize rewards
        if args.use_reward_norm:
            if args.reward_norm_type == "ema":
                self.reward_norm = EMANormalizer()
            else:
                self.reward_norm = EfficientStandardNormalizer()

        # Initialize logger
        run_name = (
            f"as_alr{args.lr}_nsteps{args.n_steps}_"
            f"minibatch{args.num_mini_batch}_epochs{args.ppo_epoch}_"
            f"gamma{args.gamma}_gae{args.gae_lambda}_"
            f"clip{args.clip_param}__{int(time.time())}"
        )

        run_name = "".join(run_name)
        env_name = "sc2_" + args.map_name
        self.logger = Logger(run_name=run_name, env=env_name, algo="AS_RMAPPO")

        # Log hyperparameters
        self.logger.log_hyperparameters({
            "env_name": env_name,
            "map_name": args.map_name,
            "difficulty": args.difficulty,
            "use_agent_id": args.use_agent_id,
            "state_agent_id": args.state_agent_id,
            "use_agent_specific_state": args.use_agent_specific_state,
            "add_distance_state": args.add_distance_state,
            "add_xy_state": args.add_xy_state,
            "add_visible_state": args.add_visible_state,
            "add_center_xy": args.add_center_xy,
            "add_enemy_action_state": args.add_enemy_action_state,
            "add_move_state": args.add_move_state,
            "add_local_obs": args.add_local_obs,
            "use_mustalive": args.use_mustalive,
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
            "num_mini_batch": args.num_mini_batch,
            "ppo_epoch": args.ppo_epoch,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_param": args.clip_param,
            "use_clipped_value_loss": args.use_clipped_value_loss,
            "entropy_coef": args.entropy_coef,
            "use_gae": args.use_gae,
            "use_proper_time_limits": args.use_proper_time_limits,
            "use_max_grad_norm": args.use_max_grad_norm,
            "max_grad_norm": args.max_grad_norm,
            "use_eval": args.use_eval,
            "eval_interval": args.eval_interval,
            "eval_episodes": args.eval_episodes,
            "save_interval": args.save_interval,
            "save_dir": args.save_dir,
            "save_replay": args.save_replay,
            "replay_dir": args.replay_dir
        })


    def run(self):
        """
        Run the training process.
        """
        # Training stats
        self.total_steps = 0
        self.episodes = 0
        self.best_win_rate = 0
        self.episode_length = 0
        self.episode_rewards = 0
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
            steps = self.collect_rollouts()
            self.total_steps += steps

            # Compute returns and advantages
            self.compute_returns()

            # Train agent
            train_info = self.agent.train(self.buffer)

            # Log training info
            self.logger.log_training(
                {
                    "train/episode_length": self.episode_length,
                    "train/value_loss": train_info["value_loss"],
                    "train/action_loss": train_info["action_loss"],
                    "train/entropy": train_info["entropy"],
                    "train/approx_kl": train_info["approx_kl"],
                    "train/clip_fraction": train_info["clip_fraction"],
                    "train/explained_variance": train_info["explained_variance"],
                }
            )

            # Reset buffer for next collection
            self.buffer.after_update()

        # Close environments
        self.env.close()
        self.evaluate_env.close()


    def warmup(self):
        """
        Warmup the agent.
        """
        # Reset environment
        obs, states = self.env.reset()

        # Store initial observations and states
        self.buffer.obs[0] = np.array(obs)  # (n_agents, n_obs)
        self.buffer.global_state[0] = np.array(states)  # (n_agents, n_state) or (n_state)
        self.buffer.available_actions[0] = np.array(self.env.get_avail_actions())  # (n_agents, n_actions)

        # Initialize masks with death masking
        # Take a dummy step to get the done flags
        actions = [0] * self.args.n_agents  # No-op actions
        _, done, _ = self.env.step(actions)

        # Reset environment again after dummy step
        obs, states = self.env.reset()
        self.buffer.obs[0] = np.array(obs)  # (n_agents, n_obs)
        self.buffer.global_state[0] = np.array(states)  # (n_agents, n_state) or (n_state)
        self.buffer.available_actions[0] = np.array(self.env.get_avail_actions())  # (n_agents, n_actions)

        # Get masks from done flags or get_masks method
        if isinstance(done, list):
            # If done is a list (from DeathMaskingWrapper), create masks from it
            masks = np.array([[0.0] if d else [1.0] for d in done], dtype=np.float32)
        elif hasattr(self.env, 'get_masks'):
            # Use the environment's get_masks method if available
            masks = self.env.get_masks()
        else:
            # Default to all ones if get_masks is not available
            masks = np.ones((self.args.n_agents, 1), dtype=np.float32)
        self.buffer.masks[0] = masks


    def collect_rollouts(self):
        """
        Collect trajectories by interacting with the environment.

        Returns:
            int: Number of steps collected
        """
        # Reset episode stats
        self.episode_rewards = 0
        self.episode_length = 0
        episode_done = False
        is_truncated = False

        # Get initial observations and states
        obs = self.buffer.obs[0].copy()  # (n_agents, n_obs)

        # Get initial states (agent-specific or global)
        state = self.buffer.global_state[0].copy()  # (n_agents, n_state)

        # Initialize RNN states
        actor_rnn_states = np.zeros((self.args.rnn_layers, self.args.n_agents, self.args.hidden_size), dtype=np.float32)
        critic_rnn_states = np.zeros((self.args.rnn_layers, self.args.n_agents, self.args.hidden_size), dtype=np.float32)

        # Initialize masks
        masks = np.ones((self.args.n_agents, 1), dtype=np.float32)

        # Get available actions
        avail_actions = self.buffer.available_actions[0].copy()  # (n_agents, n_actions)

        # Rollout steps
        for _ in range(self.args.n_steps):
            # Get actions and log probabilities
            actions, action_log_probs, actor_rnn_states = self.agent.get_actions(
                obs,
                actor_rnn_states,
                masks,
                avail_actions,
                deterministic=False)

            # Get values from critic
            # For agent-specific states, we need to handle each agent's state separately
            values, critic_rnn_states = self.agent.get_values(state, obs, critic_rnn_states, masks)

            # Execute actions
            reward, done, infos = self.env.step(actions)

            # Update episode counters
            self.episode_length += 1
            self.episode_rewards += reward

            # Normalize rewards
            if self.args.use_reward_norm:
                reward = self.reward_norm.normalize(reward)

            # Check if episode is done (all agents done or environment done)
            if isinstance(done, list):
                # If done is a list (from DeathMaskingWrapper), check if all agents are done
                episode_done = all(done)
            else:
                # Otherwise, use the done flag directly
                episode_done = done

            # Check if episode is truncated (reached max episode length)
            is_truncated = False
            if "bad_transition" in infos[0] and infos[0]["bad_transition"]:
                is_truncated = True
                episode_done = False

            # Get next observations and states
            next_obs = self.env.get_obs()

            # Get next states (agent-specific or global)
            if self.args.use_agent_specific_state:
                next_state = [self.env.get_state_agent(agent_id) for agent_id in range(self.args.n_agents)]
            else:
                next_state = [self.env.get_state()] * self.args.n_agents

            # Get next available actions
            next_avail_actions = self.env.get_avail_actions()

            # Get masks for dead agents
            if isinstance(done, list):
                # If done is a list (from DeathMaskingWrapper), create masks from it
                masks = np.array([[0.0] if d else [1.0] for d in done], dtype=np.float32)
            elif hasattr(self.env, 'get_masks'):
                # Use the environment's get_masks method if available
                masks = self.env.get_masks()
            else:
                # Default to all ones if get_masks is not available
                masks = np.ones((self.args.n_agents, 1), dtype=np.float32)

            # Set all masks to 0 if episode is done
            if episode_done:
                masks = np.zeros((self.args.n_agents, 1), dtype=np.float32)

            # Store trajectory in buffer
            self.buffer.insert(
                obs=obs,  # (n_agents, n_obs)
                global_state=state,  # (n_agents, n_state)
                actions=actions,  # (n_agents, 1)
                action_log_probs=action_log_probs,  # (n_agents, 1)
                values=values,  # (n_agents, 1)
                # Distribute rewards to living agents only if using death masking
                rewards=np.array([reward * masks[i][0] for i in range(self.args.n_agents)]).reshape(-1, 1),  # (n_agents, 1)
                masks=masks,  # (n_agents, 1)
                truncates=np.array([is_truncated]).repeat(self.args.n_agents).reshape(-1, 1),  # (n_agents, 1)
                available_actions=avail_actions,  # (n_agents, n_actions)
                actor_rnn_states=actor_rnn_states,  # (num_layers, n_agents, hidden_size)
                critic_rnn_states=critic_rnn_states,  # (num_layers, n_agents, hidden_size)
            )

            # Update for next step
            obs = np.array(next_obs)
            state = np.array(next_state)
            avail_actions = np.array(next_avail_actions)

            # If episode is done, reset environment
            if episode_done:
                self.episodes += 1

                # Reset environment
                obs, states = self.env.reset()

                # Convert states to appropriate format
                if isinstance(states, list):
                    state = np.array(states)
                else:
                    state = np.array([states] * self.args.n_agents)

                avail_actions = np.array(self.env.get_avail_actions())

                # Reset episode stats
                self.episode_rewards = 0
                self.episode_length = 0

                # Reset RNN states
                actor_rnn_states = np.zeros((self.args.rnn_layers, self.args.n_agents, self.args.hidden_size), dtype=np.float32)
                critic_rnn_states = np.zeros((self.args.rnn_layers, self.args.n_agents, self.args.hidden_size), dtype=np.float32)

                # Reset masks
                masks = np.ones((self.args.n_agents, 1), dtype=np.float32)

        return self.args.n_steps


    def compute_returns(self):
        """
        Compute returns and advantages for the collected trajectories.
        """
        # Get last state and observation
        last_state = self.buffer.global_state[-1]
        last_obs = self.buffer.obs[-1]
        last_masks = self.buffer.masks[-1]
        last_critic_rnn_states = self.buffer.critic_rnn_states[-1]

        # Get value of last state
        next_value, _ = self.agent.get_values(
            last_state,
            last_obs,
            last_critic_rnn_states,
            last_masks
        )

        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(next_value, self.args.gamma, self.args.gae_lambda)


    def evaluate(self, num_episodes=10):
        """
        Evaluate the current policy without exploration.

        Args:
            num_episodes (int): Number of episodes to evaluate

        Returns:
            tuple: (mean_rewards, win_rate)
        """
        # Evaluation stats
        episode_rewards = []
        episode_lengths = []
        win_rates = []

        # Evaluate for num_episodes
        for _ in range(num_episodes):
            # Reset environment
            obs, states = self.evaluate_env.reset()

            # Convert states to appropriate format - not used directly in this function
            # but keeping for consistency with the collect_rollouts method
            if isinstance(states, list):
                _ = np.array(states)
            else:
                _ = np.array([states] * self.args.n_agents)

            # Initialize RNN states
            eval_rnn_states = np.zeros((self.args.rnn_layers, self.args.n_agents, self.args.hidden_size), dtype=np.float32)

            # Initialize masks
            eval_masks = np.ones((self.args.n_agents, 1), dtype=np.float32)

            # Episode stats
            episode_reward = 0
            episode_length = 0
            episode_done = False

            # Run episode
            while not episode_done:
                # Get observations
                obs = np.array(self.evaluate_env.get_obs())
                avail_actions = np.array(self.evaluate_env.get_avail_actions())  # (n_agents, n_actions)

                # Get actions
                actions, _, eval_rnn_states = self.agent.get_actions(
                    obs,
                    eval_rnn_states,
                    eval_masks,
                    avail_actions,
                    deterministic=True)

                # Execute actions
                reward, done, infos = self.evaluate_env.step(actions)

                # Update episode stats
                episode_reward += reward
                episode_length += 1

                # Check if episode is done (all agents done or environment done)
                if isinstance(done, list):
                    # If done is a list (from DeathMaskingWrapper), check if all agents are done
                    episode_done = all(done)
                else:
                    # Otherwise, use the done flag directly
                    episode_done = done

                # Get masks for dead agents
                if isinstance(done, list):
                    # If done is a list (from DeathMaskingWrapper), create masks from it
                    eval_masks = np.array([[0.0] if d else [1.0] for d in done], dtype=np.float32)
                elif hasattr(self.evaluate_env, 'get_masks'):
                    # Use the environment's get_masks method if available
                    eval_masks = self.evaluate_env.get_masks()
                else:
                    # Default to all ones if get_masks is not available
                    eval_masks = np.ones((self.args.n_agents, 1), dtype=np.float32)

                # Set all masks to 0 if episode is done
                if episode_done:
                    eval_masks = np.zeros((self.args.n_agents, 1), dtype=np.float32)

            # Store episode stats
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Check if episode was won
            win = False
            for info in infos:
                if "won" in info and info["won"]:
                    win = True
                    break
            win_rates.append(1 if win else 0)

        # Calculate mean stats
        mean_reward = np.mean(episode_rewards)
        mean_length = np.mean(episode_lengths)
        win_rate = np.mean(win_rates)

        # Log evaluation stats
        self.logger.log_evaluation({
            "eval/mean_reward": mean_reward,
            "eval/mean_length": mean_length,
            "eval/win_rate": win_rate,
        })

        # Update best win rate
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            self.save(best=True)

        return mean_reward, win_rate


    def save(self, best=False):
        """
        Save the model.

        Args:
            best (bool): Whether this is the best model so far
        """
        # Create save directory if it doesn't exist
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        # Save model
        if best:
            save_path = os.path.join(self.args.save_dir, f"as_rmappo_best_{self.args.map_name}.pt")
        else:
            save_path = os.path.join(self.args.save_dir, f"as_rmappo_{self.args.map_name}_{self.total_steps}.pt")

        # Save agent state
        torch.save({
            "actor_state_dict": self.agent.actor.state_dict(),
            "critic_state_dict": self.agent.critic.state_dict(),
            "args": self.args,
        }, save_path)

        print(f"Model saved to {save_path}")
