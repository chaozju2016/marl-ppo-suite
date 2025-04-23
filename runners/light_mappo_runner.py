import os
import time
import numpy as np
import torch

from buffers.light_rollout_storage import RolloutStorage
from envs import create_env
from algos.light_mappo import LightMAPPO
from utils.logger import Logger
from utils.reward_normalization import EfficientStandardNormalizer, EMANormalizer
# import wandb

class LightMAPPORunner:
    """
    Runner class to handle environment interactions and training for MAPPO.

    This class manages the environment, agent, buffer, and training process,
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

        # Create training and evaluation environments using the factory function
        self.env = create_env(args, is_eval=False)
        self.evaluate_env = create_env(args, is_eval=True)

        # Get environment info
        env_info = self.env.get_env_info()
        args.n_agents = env_info["n_agents"]
        args.action_dim = env_info["n_actions"]
        args.state_dim = env_info["state_shape"]
        args.obs_dim = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        print(f"n_agents: {args.n_agents}, "
              f"action_dim: {args.action_dim}, "
              f"state_dim: {args.state_dim}, "
              f"obs_dim: {args.obs_dim}, "
              f"episode_limit: {args.episode_limit}")

        # Initialize game stats
        self.game_stats = {
            'battles_won': 0,
            'battles_game': 0,
            'win_rate': 0,
            'timeouts': 0
        }

        # Create agent
        self.agent = LightMAPPO(args, args.obs_dim, args.state_dim, args.action_dim, self.device)

        # Create buffer
        self.buffer = RolloutStorage(args.n_steps,
            args.n_agents,
            args.obs_dim,
            args.action_dim,
            args.state_dim)

        # Normalize rewards
        if args.use_reward_norm:
            if args.reward_norm_type == "ema":
                self.reward_norm = EMANormalizer()
            else:
                self.reward_norm = EfficientStandardNormalizer()

        # Initialize logger
        run_name = (
            f"lr{args.lr}_nsteps{args.n_steps}_"
            f"minibatch{args.num_mini_batch}_epochs{args.ppo_epoch}_"
            f"gamma{args.gamma}_gae{args.gae_lambda}_"
            f"clip{args.clip_param}_dmask{args.use_death_masking}_aid{args.use_agent_id}_"
            f"{int(time.time())}"
        )

        run_name = "".join(run_name)
        env_name = "sc2_" + args.map_name
        self.logger = Logger(run_name=run_name, env=env_name, algo="Light_MAPPO")
        # Log hyperparameters
        self.logger.log_hyperparameters({
            "env_name": env_name,
            "map_name": args.map_name,
            "difficulty": args.difficulty,
            "use_agent_id": args.use_agent_id,
            "death_masking": args.use_death_masking,
            "lr": args.lr,
            "optimizer_eps": args.optimizer_eps,
            "use_linear_lr_decay": args.use_linear_lr_decay,
            "use_feature_normalization": args.use_feature_normalization,
            "use_value_norm": args.use_value_norm,
            "use_reward_norm": args.use_reward_norm,
            "actor_gain": args.actor_gain,
            "hidden_size": args.hidden_size,
            "fc_layers": args.fc_layers,
            "n_steps": args.n_steps,
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

            # Log training information
            self.logger.add_scalar('train/critic_loss', train_info['critic_loss'], self.total_steps)
            self.logger.add_scalar('train/actor_loss', train_info['actor_loss'], self.total_steps)
            self.logger.add_scalar('train/entropy_loss', train_info['entropy_loss'], self.total_steps)
            self.logger.add_scalar('train/approx_kl', train_info['approx_kl'], self.total_steps)
            self.logger.add_scalar('train/clip_ratio', train_info['clip_ratio'], self.total_steps)
            self.logger.add_scalar('train/actor_grad_norm', train_info['actor_grad_norm'], self.total_steps)
            self.logger.add_scalar('train/critic_grad_norm', train_info['critic_grad_norm'], self.total_steps)


            # Reset buffer for next collection
            self.buffer.after_update()

        # Final evaluation
        if self.args.use_eval:
            print(f"Final evaluation at {self.total_steps}/{self.args.max_steps}")
            self.evaluate(self.args.eval_episodes)

        # Close environments
        self.env.close()
        self.evaluate_env.close()


    def warmup(self):
        """
        Warmup the agent.
        """
        self.env.reset()

        obs, state = self.env.get_obs(), self.env.get_state()

        # Store initial observations
        self.buffer.obs[0] = np.array(obs) # (n_agents, n_obs)
        self.buffer.global_state[0] = np.array(state) # (n_state)
        self.buffer.available_actions[0] = np.array(self.env.get_avail_actions()) # (n_agents, n_actions)

    def collect_rollouts(self):
        """
        Collect trajectories by interacting with the environment.

        Returns:
            tuple: (mean_rewards, win_rate, total_steps)
        """
        # Get initial observations
        obs, state = self.buffer.obs[0], self.buffer.global_state[0]
        avail_actions = self.buffer.available_actions[0] # (n_agents, n_actions)
        active_masks = self.buffer.active_masks[0] # (n_agents, )

        # Rollout steps
        for step in range(self.args.n_steps):
            # Get actions and log probabilities
            actions, action_log_probs  = self.agent.get_actions(
                obs, avail_actions, False)
            # Get values from critic
            values = self.agent.get_values(state, obs, active_masks)

            # Execute actions
            reward, dones, infos = self.env.step(actions)
            done = np.all(dones)

            # Update episode counters
            self.episode_length += 1
            self.episode_rewards += reward

            # Normalize rewards
            if self.args.use_reward_norm:
                reward = self.reward_norm.normalize(reward)

            # Handle episode termination
            if done:
                # Get latest stats and update game stats in one go
                latest_stats = self.env.get_stats()
                episode_outcome = self._check_episode_outcome(latest_stats)
                is_truncated = episode_outcome == "truncated"
                active_masks = np.ones_like(dones)

                # Update tracking stats
                self.game_stats.update(latest_stats)
                self.episodes += 1

                self.logger.add_scalar('train/win_rate', latest_stats['win_rate'], self.total_steps + step)
                self.logger.add_scalar('train/length', self.episode_length, self.total_steps + step)
                self.logger.add_scalar('train/rewards', self.episode_rewards, self.total_steps + step)

                # Reset environment
                self.env.reset()

                # Reset episode tracking
                self.episode_rewards = 0
                self.episode_length = 0

            else:
                is_truncated = False
                active_masks = np.array(1-dones)

            obs, state = np.array(self.env.get_obs()), np.array(self.env.get_state())
            avail_actions = np.array(self.env.get_avail_actions()) # (n_agents, n_actions)

            # Store trajectory in buffer
            self.buffer.insert(
                obs=obs, #(n_agents, n_obs)
                global_state=state, #(n_state)
                actions=actions.squeeze(-1), #(n_agents, )
                action_log_probs=action_log_probs.squeeze(-1), #(n_agents, )
                values=values.squeeze(-1), #(n_agents, )
                rewards=np.array([reward]).repeat(self.args.n_agents), #(n_agents, )
                masks=np.array([1-done]).repeat(self.args.n_agents), #(n_agents, )
                active_masks=active_masks, #(n_agents, )
                truncates=np.array([is_truncated]).repeat(self.args.n_agents), #(n_agents, )
                available_actions=avail_actions, # (n_agents, n_actions)
            )

        return self.args.n_steps

    def compute_returns(self):
        """
        Compute returns and advantages for the collected trajectories.
        """
        next_value = self.agent.get_values(self.buffer.global_state[-1],
                                           self.buffer.obs[-1],
                                           self.buffer.active_masks[-1])

        self.buffer.compute_returns_and_advantages(
            next_value.squeeze(-1),
            self.args.gamma,
            self.args.gae_lambda)

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

        # Evaluation loop
        for _ in range(num_episodes):
            # Reset environment
            self.evaluate_env.reset()


            # Episode tracking
            episode_reward = 0
            episode_length = 0
            episode_done = False

            while not episode_done:
                obs = np.array(self.evaluate_env.get_obs())
                avail_actions = np.array(self.evaluate_env.get_avail_actions()) # (n_agents, n_actions)

                # Get actions and log probabilities
                actions, _ = self.agent.get_actions(
                    obs, avail_actions, True)

                # Execute actions
                reward, dones, infos = self.evaluate_env.step(actions)

                # Update episode rewards
                episode_reward += reward
                episode_length += 1
                episode_done =  np.all(dones)


            # Track win rate
            win = True if 'battle_won' in infos and infos['battle_won'] else False
            win_rates.append(1 if win else 0)

            # Track episode rewards
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        # Calculate statistics
        mean_rewards = np.mean(episode_rewards)
        mean_length = np.mean(episode_lengths)
        win_rate = np.mean(win_rates)

        # Add model saving
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            save_path = os.path.join(self.logger.dir_name, f"best-torch.model")
            self.agent.save(save_path)
            print(f"Saved best model with win rate {win_rate:.2f} to {save_path}")

        self.logger.add_scalar('eval/rewards', mean_rewards, self.total_steps)
        self.logger.add_scalar('eval/win_rate', win_rate, self.total_steps)
        self.logger.add_scalar('eval/length', mean_length, self.total_steps)
        # Print evaluation results
        print(f"{self.total_steps}/{self.args.max_steps} Evaluation: Mean rewards: {mean_rewards:.2f},  Mean length: {mean_length:.2f}, Win rate: {win_rate:.2f}")

        return mean_rewards, win_rate


    def _check_episode_outcome(self, latest_stats):
        """
        Check the outcome of the episode.
        """

        # First check for timeout
        if latest_stats['timeouts'] > self.game_stats['timeouts']:
            return "truncated"

        # Then check for battle outcome
        if latest_stats['battles_game'] > self.game_stats['battles_game']:
            return "victory" if latest_stats['battles_won'] > self.game_stats['battles_won'] \
                else "defeat"

        return "ongoing"
