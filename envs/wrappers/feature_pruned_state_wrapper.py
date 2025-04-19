"""
Feature-Pruned State Wrapper for StarCraft 2 environments.

This wrapper provides feature-pruned agent-specific global states as described in the MAPPO paper,
and is designed to be independent from other wrappers and compatible with vectorized environments.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from gymnasium.spaces import Box, Discrete
from envs.wrappers.base_wrapper import BaseWrapper


class FeaturePrunedStateWrapper(BaseWrapper):
    """
    A wrapper for StarCraft2Env that implements Feature-Pruned Agent-Specific Global State (FP)
    as described in the MAPPO paper.

    This wrapper is designed to be independent from other wrappers and compatible with
    vectorized environments.
    """

    def __init__(self, env,
                 use_agent_specific_state=True,
                 add_distance_state=False,
                 add_xy_state=False,
                 add_visible_state=False,
                 add_center_xy=True,
                 add_enemy_action_state=False,
                 use_mustalive=True,
                 add_agent_id=True):
        """
        Initialize the wrapper with the base environment.

        Args:
            env: The base environment to wrap (StarCraft2Env or another wrapper)
            use_agent_specific_state: Whether to use agent-specific global state (default: True)
            add_distance_state: Whether to add distance features to the state (default: False)
            add_xy_state: Whether to add relative x,y coordinates to the state (default: False)
            add_visible_state: Whether to add visibility information to the state (default: False)
            add_center_xy: Whether to add center-relative coordinates to the state (default: True)
            add_enemy_action_state: Whether to add enemy action availability to the state (default: False)
            use_mustalive: Whether to only return non-zero state for alive agents (default: True)
            add_agent_id: Whether to add agent ID to observations and states (default: True)
        """
        super(FeaturePrunedStateWrapper, self).__init__(env)
        self.use_agent_specific_state = use_agent_specific_state
        self.add_distance_state = add_distance_state
        self.add_xy_state = add_xy_state
        self.add_visible_state = add_visible_state
        self.add_center_xy = add_center_xy
        self.add_enemy_action_state = add_enemy_action_state
        self.use_mustalive = use_mustalive
        self.add_agent_id = add_agent_id
        self.add_local_obs = False  # Always set to False as we don't need it
        self.add_move_state = False  # Always set to False as we don't need it

        # Define observation and action spaces for vectorization
        self.observation_space = Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(self.get_obs_size(),),
            dtype=np.float32
        )

        self.share_observation_space = Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(self.get_state_size(),),
            dtype=np.float32
        )

        self.action_space = Discrete(self.n_actions)

    def reset(self):
        """
        Reset the environment and return observations, states, and available actions.

        Returns:
            obs: List of observations for each agent
            states: List of agent-specific states for each agent
            available_actions: List of available actions for each agent
        """
        # Call the base environment's reset method
        obs, state = self.env.reset()

        # Get available actions for each agent
        available_actions = self.get_avail_actions()

        # Get agent-specific states if enabled
        if self.use_agent_specific_state:
            agent_specific_states = [self.get_state_agent(agent_id) for agent_id in range(self.n_agents)]
            return obs, agent_specific_states, available_actions
        else:
            # Return the original state for all agents
            return obs, [state] * self.n_agents, available_actions

    def step(self, actions):
        """
        Take a step in the environment and return observations, states, rewards, dones, infos, and available actions.

        Args:
            actions: Actions to take

        Returns:
            obs: List of observations for each agent
            states: List of agent-specific states for each agent
            rewards: Rewards for each agent
            dones: Done flags for each agent
            infos: Info dictionaries
            available_actions: List of available actions for each agent
        """
        # Call the base environment's step method
        reward, terminated, info = self.env.step(actions)

        # Get observations
        obs = self.get_obs()

        # Get agent-specific states if enabled
        if self.use_agent_specific_state:
            states = [self.get_state_agent(agent_id) for agent_id in range(self.n_agents)]
        else:
            states = [self.get_state()] * self.n_agents

        # Get available actions
        available_actions = self.get_avail_actions()

        # Format rewards for each agent
        rewards = [[reward]] * self.n_agents

        # Format dones for each agent
        if terminated:
            # If the episode is terminated, all agents are done
            dones = [True] * self.n_agents
        else:
            # Create a list of done flags for each agent based on death status
            dones = [bool(self.env.death_tracker_ally[agent_id]) for agent_id in range(self.n_agents)]

        return obs, states, rewards, dones, info, available_actions

    def get_avail_actions(self):
        """
        Get available actions for all agents.

        Returns:
            List of available action masks for each agent
        """
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return np.array(avail_actions)

    def get_state_agent(self, agent_id):
        """
        Returns a feature-pruned agent-specific global state for the given agent_id.

        This state includes:
        - The agent's own features
        - Features of allies relative to this agent
        - Features of enemies relative to this agent
        - Optional additional features based on configuration
        """
        # If agent-specific state is disabled, return the regular state
        if not self.use_agent_specific_state:
            return self.get_state()

        # Get the unit for this agent
        unit = self.get_unit_by_id(agent_id)

        # Get dimensions for different feature types
        nf_al = 2 + self.env.shield_bits_ally + self.env.unit_type_bits  # health, cooldown, shield, unit_type
        nf_en = 1 + self.env.shield_bits_enemy + self.env.unit_type_bits  # health, shield, unit_type

        # Add center coordinates if enabled
        if self.add_center_xy:
            nf_al += 2  # center_x, center_y
            nf_en += 2  # center_x, center_y

        # Add distance features if enabled
        if self.add_distance_state:
            nf_al += 1  # distance
            nf_en += 1  # distance

        # Add relative position features if enabled
        if self.add_xy_state:
            nf_al += 2  # rel_x, rel_y
            nf_en += 2  # rel_x, rel_y

        # Add visibility features if enabled
        if self.add_visible_state:
            nf_al += 1  # visible
            nf_en += 1  # visible

        # Add last action features if enabled
        if self.state_last_action:
            nf_al += self.n_actions  # last_action

        # Add enemy action availability if enabled
        if self.add_enemy_action_state:
            nf_en += 1  # available_to_attack

        # Initialize feature arrays
        ally_state = np.zeros((self.n_agents - 1, nf_al), dtype=np.float32)
        enemy_state = np.zeros((self.n_enemies, nf_en), dtype=np.float32)
        own_feats = np.zeros(nf_al, dtype=np.float32)

        # Get map center coordinates
        center_x = self.env.map_x / 2
        center_y = self.env.map_y / 2

        # Skip if agent is dead and use_mustalive is True
        if (self.use_mustalive and unit.health <= 0):
            # Return zeros for all features
            state = np.concatenate((
                own_feats.flatten(),
                ally_state.flatten(),
                enemy_state.flatten()
            ))

            # Add agent ID if enabled
            if self.add_agent_id:
                agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
                agent_id_feats[agent_id] = 1.0
                state = np.append(state, agent_id_feats)

            return state

        # Get agent's position and sight range
        x = unit.pos.x
        y = unit.pos.y
        sight_range = self.unit_sight_range(agent_id)

        # Get available actions for this agent
        avail_actions = self.get_avail_agent_actions(agent_id)

        # Fill in own features
        own_idx = 0
        own_feats[own_idx] = unit.health / unit.health_max  # health
        own_idx += 1

        # Add cooldown/energy
        max_cd = self.unit_max_cooldown(unit)
        if hasattr(unit, 'energy') and unit.energy > 0:
            own_feats[own_idx] = unit.energy / max_cd  # energy
        else:
            own_feats[own_idx] = unit.weapon_cooldown / max_cd  # cooldown
        own_idx += 1

        # Add center-relative coordinates if enabled
        if self.add_center_xy:
            own_feats[own_idx] = (x - center_x) / self.env.max_distance_x  # center_x
            own_feats[own_idx + 1] = (y - center_y) / self.env.max_distance_y  # center_y
            own_idx += 2

        # Add shield if applicable
        if self.env.shield_bits_ally > 0:
            max_shield = self.unit_max_shield(unit)
            if max_shield > 0:
                own_feats[own_idx] = unit.shield / max_shield  # shield
            own_idx += 1

        # Add unit type if applicable
        if self.env.unit_type_bits > 0:
            type_id = self.get_unit_type_id(unit, True)
            own_feats[own_idx + type_id] = 1  # unit_type
            own_idx += self.env.unit_type_bits

        # Add last action if enabled
        if self.env.state_last_action:
            # Get last_action from the environment
            last_action = getattr(self.env, 'last_action', None)
            if last_action is not None:
                own_feats[own_idx:own_idx + self.n_actions] = last_action[agent_id]

        # Fill in ally features
        al_ids = [al_id for al_id in range(self.n_agents) if al_id != agent_id]
        for i, al_id in enumerate(al_ids):
            al_unit = self.get_unit_by_id(al_id)
            if al_unit.health > 0:
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                max_cd = self.unit_max_cooldown(al_unit)
                dist = self.distance(x, y, al_x, al_y)

                # Basic features
                idx = 0
                ally_state[i, idx] = al_unit.health / al_unit.health_max  # health
                idx += 1

                # Add cooldown/energy
                if hasattr(al_unit, 'energy') and al_unit.energy > 0:
                    ally_state[i, idx] = al_unit.energy / max_cd  # energy
                else:
                    ally_state[i, idx] = al_unit.weapon_cooldown / max_cd  # cooldown
                idx += 1

                # Add center-relative coordinates if enabled
                if self.add_center_xy:
                    ally_state[i, idx] = (al_x - center_x) / self.env.max_distance_x  # center_x
                    ally_state[i, idx + 1] = (al_y - center_y) / self.env.max_distance_y  # center_y
                    idx += 2

                # Add shield if applicable
                if self.env.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(al_unit)
                    if max_shield > 0:
                        ally_state[i, idx] = al_unit.shield / max_shield  # shield
                    idx += 1

                # Add unit type if applicable
                if self.env.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(al_unit, True)
                    ally_state[i, idx + type_id] = 1  # unit_type
                    idx += self.env.unit_type_bits

                # Add distance if enabled
                if self.add_distance_state:
                    ally_state[i, idx] = dist / sight_range  # distance
                    idx += 1

                # Add relative coordinates if enabled
                if self.add_xy_state:
                    ally_state[i, idx] = (al_x - x) / sight_range  # relative_x
                    ally_state[i, idx + 1] = (al_y - y) / sight_range  # relative_y
                    idx += 2

                # Add visibility if enabled
                if self.add_visible_state:
                    ally_state[i, idx] = 1 if dist < sight_range else 0  # visible
                    idx += 1

                # Add last action if enabled
                if self.env.state_last_action:
                    # Get last_action from the environment
                    last_action = getattr(self.env, 'last_action', None)
                    if last_action is not None:
                        ally_state[i, idx:idx + self.n_actions] = last_action[al_id]

        # Fill in enemy features
        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                # Basic features
                idx = 0
                enemy_state[e_id, idx] = e_unit.health / e_unit.health_max  # health
                idx += 1

                # Add center-relative coordinates if enabled
                if self.add_center_xy:
                    enemy_state[e_id, idx] = (e_x - center_x) / self.env.max_distance_x  # center_x
                    enemy_state[e_id, idx + 1] = (e_y - center_y) / self.env.max_distance_y  # center_y
                    idx += 2

                # Add shield if applicable
                if self.env.shield_bits_enemy > 0:
                    max_shield = self.unit_max_shield(e_unit)
                    if max_shield > 0:
                        enemy_state[e_id, idx] = e_unit.shield / max_shield  # shield
                    idx += 1

                # Add unit type if applicable
                if self.env.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(e_unit, False)
                    enemy_state[e_id, idx + type_id] = 1  # unit_type
                    idx += self.env.unit_type_bits

                # Add distance if enabled
                if self.add_distance_state:
                    enemy_state[e_id, idx] = dist / sight_range  # distance
                    idx += 1

                # Add relative coordinates if enabled
                if self.add_xy_state:
                    enemy_state[e_id, idx] = (e_x - x) / sight_range  # relative_x
                    enemy_state[e_id, idx + 1] = (e_y - y) / sight_range  # relative_y
                    idx += 2

                # Add visibility if enabled
                if self.add_visible_state:
                    enemy_state[e_id, idx] = 1 if dist < sight_range else 0  # visible
                    idx += 1

                # Add enemy action availability if enabled
                if self.add_enemy_action_state:
                    enemy_state[e_id, idx] = avail_actions[self.env.n_actions_no_attack + e_id]  # available_to_attack

        # Combine all features
        state = np.concatenate((
            own_feats.flatten(),
            ally_state.flatten(),
            enemy_state.flatten()
        ))

        # Add agent ID if enabled
        if self.add_agent_id:
            agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
            agent_id_feats[agent_id] = 1.0
            state = np.append(state, agent_id_feats)

        return state

    def get_obs(self):
        """
        Get observations for all agents.

        Returns:
            List of observations for each agent
        """
        obs = []
        for i in range(self.n_agents):
            # Get the original observation from the base environment
            agent_obs = self.get_obs_agent(i)

            # Add agent ID if enabled
            if self.add_agent_id:
                agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
                agent_id_feats[i] = 1.0
                agent_obs = np.concatenate((agent_obs, agent_id_feats))

            obs.append(agent_obs)
        return obs

    def get_obs_size(self):
        """
        Returns the size of the observation.

        Returns:
            Size of the observation
        """
        # Get the original observation size
        obs_size = self.env.get_obs_size()

        # Add agent ID size if enabled
        if self.add_agent_id:
            obs_size += self.n_agents

        return obs_size

    def get_state_size(self):
        """
        Returns the size of the agent-specific global state.

        Returns:
            Size of the agent-specific global state
        """
        if not self.use_agent_specific_state:
            return self.env.get_state_size()

        # Calculate the size of the agent-specific state
        nf_al = 2 + self.env.shield_bits_ally + self.env.unit_type_bits  # health, cooldown, shield, unit_type
        nf_en = 1 + self.env.shield_bits_enemy + self.env.unit_type_bits  # health, shield, unit_type

        # Add center coordinates if enabled
        if self.add_center_xy:
            nf_al += 2  # center_x, center_y
            nf_en += 2  # center_x, center_y

        # Add distance features if enabled
        if self.add_distance_state:
            nf_al += 1  # distance
            nf_en += 1  # distance

        # Add relative position features if enabled
        if self.add_xy_state:
            nf_al += 2  # rel_x, rel_y
            nf_en += 2  # rel_x, rel_y

        # Add visibility features if enabled
        if self.add_visible_state:
            nf_al += 1  # visible
            nf_en += 1  # visible

        # Add last action features if enabled
        if self.env.state_last_action:
            nf_al += self.n_actions  # last_action for allies

        # Add enemy action availability if enabled
        if self.add_enemy_action_state:
            nf_en += 1  # available_to_attack

        # Calculate total size
        own_feats = nf_al
        ally_feats = (self.n_agents - 1) * nf_al
        enemy_feats = self.n_enemies * nf_en

        size = own_feats + ally_feats + enemy_feats

        # Add agent ID if enabled
        if self.add_agent_id:
            size += self.n_agents

        return size

    # Forward unit-related methods to the environment
    def get_unit_by_id(self, unit_id):
        """
        Get a unit by its ID.

        Args:
            unit_id: ID of the unit

        Returns:
            The unit with the given ID
        """
        return self.env.get_unit_by_id(unit_id)

    def unit_max_cooldown(self, unit):
        """
        Get the maximum cooldown of a unit.

        Args:
            unit: The unit

        Returns:
            Maximum cooldown of the unit
        """
        return self.env.unit_max_cooldown(unit)

    def unit_max_shield(self, unit):
        """
        Get the maximum shield of a unit.

        Args:
            unit: The unit

        Returns:
            Maximum shield of the unit
        """
        return self.env.unit_max_shield(unit)

    def get_unit_type_id(self, unit, ally):
        """
        Get the type ID of a unit.

        Args:
            unit: The unit
            ally: Whether the unit is an ally

        Returns:
            Type ID of the unit
        """
        return self.env.get_unit_type_id(unit, ally)

    def unit_sight_range(self, agent_id):
        """
        Get the sight range of an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Sight range of the agent
        """
        return self.env.unit_sight_range(agent_id)

    def distance(self, x1, y1, x2, y2):
        """
        Calculate the distance between two points.

        Args:
            x1: X-coordinate of the first point
            y1: Y-coordinate of the first point
            x2: X-coordinate of the second point
            y2: Y-coordinate of the second point

        Returns:
            Distance between the two points
        """
        return self.env.distance(x1, y1, x2, y2)

    def get_env_info(self):
        """
        Returns environment information with updated observation and state sizes.

        Returns:
            Dictionary of environment information
        """
        # Get the original environment information
        env_info = self.env.get_env_info()

        # Update observation size if agent ID is enabled
        if self.add_agent_id:
            env_info['obs_shape'] = self.get_obs_size()

        # Update state size if agent-specific state is enabled
        if self.use_agent_specific_state:
            env_info['state_shape'] = self.get_state_size()

        return env_info
