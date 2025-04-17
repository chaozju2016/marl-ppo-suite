from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from envs.wrappers.base_wrapper import BaseWrapper


class AgentIDWrapper(BaseWrapper):
    """
    A wrapper for StarCraft2Env that adds agent_id to observations and/or global state.

    This wrapper allows adding agent_id without modifying the original StarCraft2Env class.
    It can add agent IDs to:
    1. Observations - one-hot encoding of the agent's own ID
    """

    def __init__(self, env, use_agent_id=True):
        """
        Initialize the wrapper with the base environment.

        Args:
            env: The base environment to wrap (StarCraft2Env or another wrapper)
            use_agent_id: Whether to add agent ID to observations (default: True)
        """
        super(AgentIDWrapper, self).__init__(env)
        self.use_agent_id = use_agent_id

    def get_obs(self):
        """Get observations for all agents with agent_id if enabled."""
        # Get observations for all agents
        obs = []
        for i in range(self.n_agents):
            obs.append(self.get_obs_agent(i))
        return obs

    def get_obs_agent(self, agent_id):
        """
        Returns observation for agent_id with agent_id feature added if enabled.
        """
        # Get the original observation from the base environment
        agent_obs = self.env.get_obs_agent(agent_id)

        # If agent_id is enabled, add it to the observation
        if self.use_agent_id:
            # Create a one-hot encoding of the agent_id
            agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
            agent_id_feats[agent_id] = 1.0

            # Concatenate the agent_id features to the observation
            agent_obs = np.concatenate((agent_obs, agent_id_feats))

        return agent_obs

    def get_obs_size(self):
        """
        Returns the size of the observation with agent_id if enabled.
        """
        # Get the original observation size from the base environment
        obs_size = self.env.get_obs_size()

        # If agent_id is enabled, add the size of the agent_id features
        if self.use_agent_id:
            obs_size += self.n_agents

        return obs_size

    def reset(self):
        """
        Reset the environment and return observations and state with agent_id if enabled.
        """
        # Call the base environment's reset method
        obs, state = self.env.reset()

        # Process observations if agent_id is enabled
        if self.use_agent_id:
            new_obs = []
            for i, o in enumerate(obs):
                # Create a one-hot encoding of the agent_id
                agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
                agent_id_feats[i] = 1.0
                # Concatenate the agent_id features to the observation
                new_obs.append(np.concatenate((o, agent_id_feats)))
            obs = new_obs

        return obs, state

    def get_env_info(self):
        """
        Returns environment information with updated observation and state sizes.
        """
        # Get the original environment information
        env_info = self.env.get_env_info()

        # Update observation size if agent_id is enabled
        if self.use_agent_id:
            env_info['obs_shape'] = self.get_obs_size()

        return env_info
