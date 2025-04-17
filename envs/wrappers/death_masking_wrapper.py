from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from envs.wrappers.base_wrapper import BaseWrapper


class DeathMaskingWrapper(BaseWrapper):
    """
    A wrapper for StarCraft2Env that implements death masking as described in the MAPPO paper.

    This wrapper ensures that dead agents are properly masked during training and evaluation,
    which improves learning stability and performance. It sets observations and states of dead
    agents to zero vectors and provides masks for identifying dead agents.
    """

    def __init__(self, env, use_death_masking=True):
        """
        Initialize the wrapper with the base environment.

        Args:
            env: The base environment to wrap (StarCraft2Env or another wrapper)
            use_death_masking: Whether to use death masking (default: True)
        """
        super(DeathMaskingWrapper, self).__init__(env)
        self.use_death_masking = use_death_masking

    def step(self, actions):
        """
        Take a step in the environment and update the done flag for dead agents.

        Args:
            actions: Actions to take

        Returns:
            tuple: (reward, done, info) with done updated for dead agents
        """
        # Call the base environment's step method
        reward, done, info = self.env.step(actions)

        # Always return a list of done flags for each agent
        # If death masking is enabled, update the done flag for dead agents
        if self.use_death_masking:
            # Get the base environment to access death_tracker_ally
            base_env = self.get_base_env()
            if done:
                dones = [True] * self.n_agents
            else:
                # Create a list of done flags for each agent based on death status
                dones = [bool(base_env.death_tracker_ally[agent_id]) for agent_id in range(self.n_agents)]
        else:
            # If death masking is disabled, all agents have the same done flag
            dones = [done] * self.n_agents

        return reward, np.array(dones), info

    def get_active_masks(self):
        """
        Get active masks for all agents based on whether they are alive or dead.
        These masks are used in the PPO loss calculation to exclude dead agents.

        Returns:
            numpy.ndarray: Array of shape (n_agents, 1) with 0 for dead agents and 1 for alive agents
        """
        active_masks = np.ones((self.n_agents, 1), dtype=np.float32)

        if self.use_death_masking:
            base_env = self.get_base_env()
            # 0.0 for dead agents (True in death_tracker_ally), 1.0 for alive agents (False in death_tracker_ally)
            active_masks = np.array([[0.0 if is_dead else 1.0] for is_dead in base_env.death_tracker_ally], dtype=np.float32)

        return active_masks

    def get_agent_mask(self, agent_id):
        """
        Get mask for a specific agent based on whether it is alive or dead.

        Args:
            agent_id (int): The agent ID

        Returns:
            float: 0.0 if the agent is dead, 1.0 if alive
        """
        # Check if death masking is enabled and agent_id is valid
        if self.use_death_masking:
            # Get the base environment to access death_tracker_ally
            base_env = self.get_base_env()
            if 0 <= agent_id < len(base_env.death_tracker_ally):
                # Return 0.0 for dead agents, 1.0 for alive agents
                return 0.0 if base_env.death_tracker_ally[agent_id] else 1.0
        # Default to alive if death masking is disabled or agent_id is invalid
        return 1.0