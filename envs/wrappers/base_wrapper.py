from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class BaseWrapper:
    """
    Base wrapper class for StarCraft2Env that forwards all methods to the wrapped environment.

    This class provides a foundation for all other wrappers, handling the common
    method forwarding and attribute access patterns.
    """

    def __init__(self, env):
        """
        Initialize the wrapper with the base environment.

        Args:
            env: The base environment to wrap (StarCraft2Env or another wrapper)
        """
        self.env = env

        # Forward common environment attributes
        self.n_agents = getattr(self.env, 'n_agents', 0)
        self.n_enemies = getattr(self.env, 'n_enemies', 0)
        self.episode_limit = getattr(self.env, 'episode_limit', 0)
        self.n_actions = getattr(self.env, 'n_actions', 0)

    def reset(self):
        """Reset the environment and return initial observations and states."""
        return self.env.reset()

    def step(self, actions):
        """Take a step in the environment."""
        return self.env.step(actions)

    def get_obs(self):
        """Get observations for all agents."""
        return self.env.get_obs()

    def get_obs_agent(self, agent_id):
        """Get observation for a specific agent."""
        return self.env.get_obs_agent(agent_id)

    def get_obs_size(self):
        """Get the size of the observation space."""
        return self.env.get_obs_size()

    def get_base_env(self):
        """
        Get the base environment by traversing the wrapper stack.

        Returns:
            The base environment (StarCraft2Env)
        """
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env

    def get_state(self):
        """Get the global state."""
        return self.env.get_state()

    def get_state_size(self):
        """Get the size of the state space."""
        return self.env.get_state_size()

    def get_avail_actions(self):
        """Get available actions for all agents."""
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        """Get available actions for a specific agent."""
        return self.env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        """Get the total number of actions."""
        return self.env.get_total_actions()

    def get_env_info(self):
        """
        Get environment information.

        This method should be overridden by subclasses that modify the observation or state shapes.
        The default implementation just forwards the call to the wrapped environment.
        """
        return self.env.get_env_info()

    def close(self):
        """Close the environment."""
        return self.env.close()

    def save_replay(self):
        """Save a replay if the environment supports it."""
        if hasattr(self.env, 'save_replay'):
            return self.env.save_replay()

    # Forward any other attributes to the base environment
    def __getattr__(self, name):
        return getattr(self.env, name)
