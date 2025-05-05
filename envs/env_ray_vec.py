"""
Ray-based vectorized environment implementation.
This provides better performance than multiprocessing-based implementations,
especially on Apple Silicon.
"""

import numpy as np
import ray
from envs.env_vectorization import VecEnv, CloudpickleWrapper

@ray.remote
class RayEnvWorker:
    """
    Ray actor for running a single environment.
    """
    def __init__(self, env_fn, env_idx):
        """
        Initialize the worker with an environment creation function.
        
        Args:
            env_fn: Function that creates the environment
            env_idx: Index of this environment
        """
        self.env = env_fn()
        self.env_idx = env_idx
        
    def get_spaces(self):
        """Get the observation and action spaces."""
        return (
            self.env.observation_space, 
            self.env.share_observation_space, 
            self.env.action_space
        )
    
    def get_num_agents(self):
        """Get the number of agents in the environment."""
        return self.env.n_agents
    
    def get_episode_limit(self):
        """Get the episode limit."""
        return self.env.episode_limit
    
    def reset(self):
        """Reset the environment."""
        return self.env.reset()
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (obs, state, reward, done, info, available_actions)
        """
        ob, s_ob, reward, done, info, available_actions = self.env.step(action)
        
        # Handle episode termination
        if isinstance(done, bool):  # done is a bool
            if done:  # if done, save the original obs, state, and available actions in info, and then reset
                info["final_obs"] = ob.copy()
                info["final_state"] = s_ob.copy()
                info["final_avail_actions"] = available_actions.copy()
                ob, s_ob, available_actions = self.env.reset()
        else:
            if np.all(done):  # if all agents are done, reset the environment
                info["final_obs"] = ob.copy()
                info["final_state"] = s_ob.copy()
                info["final_avail_actions"] = available_actions.copy()
                ob, s_ob, available_actions = self.env.reset()
                
        return ob, s_ob, reward, done, info, available_actions
    
    def render(self, mode="human"):
        """Render the environment."""
        return self.env.render(mode=mode)
    
    def close(self):
        """Close the environment."""
        self.env.close()


class RayVecEnv(VecEnv):
    """
    Vectorized environment implementation using Ray for parallel execution.
    
    This implementation provides better performance than multiprocessing-based
    implementations, especially on Apple Silicon.
    """
    
    def __init__(self, env_fns):
        """
        Initialize the RayVecEnv.
        
        Args:
            env_fns: List of functions that create environments
        """
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        
        # Create Ray actors (workers)
        self.workers = [
            RayEnvWorker.remote(env_fn, i) for i, env_fn in enumerate(env_fns)
        ]
        
        # Get environment information from the first worker
        observation_space, share_observation_space, action_space = ray.get(self.workers[0].get_spaces.remote())
        self.n_agents = ray.get(self.workers[0].get_num_agents.remote())
        self.episode_limit = ray.get(self.workers[0].get_episode_limit.remote())
        
        # Initialize the base VecEnv
        VecEnv.__init__(
            self, self.num_envs, observation_space, share_observation_space, action_space,
            self.n_agents, self.episode_limit
        )
        
        # Pre-allocate buffers for observations
        # obs_shape = (self.n_agents,) + tuple(observation_space.shape)
        # share_obs_shape = tuple(share_observation_space.shape)
        
        # if isinstance(action_space, Discrete):
        #     avail_actions_shape = (self.n_agents, action_space.n)
        # else:
        #     avail_actions_shape = (self.n_agents, 1)
            
        # self.obs_buf = np.zeros((self.num_envs,) + obs_shape, dtype=np.float32)
        # self.share_obs_buf = np.zeros((self.num_envs,) + share_obs_shape, dtype=np.float32)
        # self.reward_buf = np.zeros((self.num_envs, self.n_agents, 1), dtype=np.float32)
        # self.done_buf = np.zeros((self.num_envs, self.n_agents), dtype=np.bool_)
        # self.avail_actions_buf = np.zeros((self.num_envs,) + avail_actions_shape, dtype=np.float32)
        
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step with the given actions.
        
        Args:
            actions: Actions to take in each environment
        """
        self.step_futures = [
            worker.step.remote(action) for worker, action in zip(self.workers, actions)
        ]
        self.waiting = True
        
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        
        Returns:
            obs: Observations from all environments
            states: States from all environments (for CTDE)
            rewards: Rewards from all environments
            dones: Done flags from all environments
            infos: Info dictionaries from all environments
            available_actions: Available actions for all environments
        """
        if not self.waiting:
            raise RuntimeError("Called step_wait() without calling step_async() first")
            
        results = ray.get(self.step_futures)
        self.waiting = False
        
        # Unpack results
        obs, share_obs, rewards, dones, infos, available_actions = zip(*results)

        return (
            np.stack(obs),
            np.stack(share_obs),
            np.stack(rewards),
            np.stack(dones),
            infos,
            np.stack(available_actions),
        )
        
        # # Copy data to buffers
        # for i, (ob, s_ob, rew, done, avail) in enumerate(zip(obs, share_obs, rewards, dones, available_actions)):
        #     self.obs_buf[i] = ob
        #     self.share_obs_buf[i] = s_ob
        #     self.reward_buf[i] = rew
        #     self.done_buf[i] = done
        #     self.avail_actions_buf[i] = avail
        
        # return (
        #     self.obs_buf.copy(),
        #     self.share_obs_buf.copy(),
        #     self.reward_buf.copy(),
        #     self.done_buf.copy(),
        #     infos,
        #     self.avail_actions_buf.copy(),
        # )
        
    def reset(self):
        """
        Reset all the environments.
        
        Returns:
            obs: Observations from all environments
            states: States from all environments (for CTDE)
            available_actions: Available actions for all environments
        """
        # Reset all environments in parallel
        futures = [worker.reset.remote() for worker in self.workers]
        results = ray.get(futures)
        
        # Unpack results
        obs, share_obs, available_actions = zip(*results)

        # Stack results directly
        return (
            np.stack(obs),
            np.stack(share_obs),
            np.stack(available_actions)
        )
        
        # Copy data to buffers
        # for i, (ob, s_ob, avail) in enumerate(zip(obs, share_obs, available_actions)):
        #     self.obs_buf[i] = ob
        #     self.share_obs_buf[i] = s_ob
        #     self.avail_actions_buf[i] = avail
        
        # return (
        #     self.obs_buf.copy(),
        #     self.share_obs_buf.copy(),
        #     self.avail_actions_buf.copy()
        # )
        
    def close(self):
        """Close all environments and terminate Ray actors."""
        if self.closed:
            return
            
        # Close all environments
        ray.get([worker.close.remote() for worker in self.workers])
        
        # Terminate Ray actors
        for worker in self.workers:
            ray.kill(worker)
            
        self.closed = True
        
    def render(self, mode="human"):
        """
        Render the environments.
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
            
        Returns:
            If mode is 'rgb_array', returns the rendered image
            If mode is 'human', returns None
        """
        if mode == "rgb_array":
            images = ray.get([worker.render.remote(mode) for worker in self.workers])
            from envs.env_vectorization import tile_images
            return tile_images(images)
        elif mode == "human":
            ray.get(self.workers[0].render.remote(mode))
            return None
        else:
            raise ValueError(f"Unsupported render mode: {mode}")