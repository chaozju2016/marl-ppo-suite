from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from envs.wrappers.base_wrapper import BaseWrapper

#   parser.add_argument('--units', type=str, default='10v10') # for smac v2
#     parser.add_argument("--add_move_state", action='store_true', default=False)
#     parser.add_argument("--add_local_obs", action='store_true', default=False)
#     parser.add_argument("--add_distance_state", action='store_true', default=False)
#     parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
#     parser.add_argument("--add_agent_id", action='store_true', default=False)
#     parser.add_argument("--add_visible_state", action='store_true', default=False)
#     parser.add_argument("--add_xy_state", action='store_true', default=False)
#     parser.add_argument("--use_state_agent", action='store_false', default=True)
#     parser.add_argument("--use_mustalive", action='store_false', default=True)
#     parser.add_argument("--add_center_xy", action='store_false', default=True)
class AgentSpecificStateWrapper(BaseWrapper):
    """
    A wrapper for StarCraft2Env that implements Agent-Specific Global State (AS) and
    Featured-Pruned Agent-Specific Global State (FP) as described in the MAPPO paper.

    This wrapper provides agent-specific global states with optional feature pruning.
    """

    def __init__(self, env, use_agent_specific_state=True, add_distance_state=False,
                 add_xy_state=False, add_visible_state=False, add_center_xy=True,
                 add_enemy_action_state=False, add_move_state=False, add_local_obs=False,
                 use_mustalive=True):
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
            add_move_state: Whether to add movement features to the state (default: False)
            add_local_obs: Whether to add the agent's local observation to the state (default: False)
            use_mustalive: Whether to only return non-zero state for alive agents (default: True)
            use_agent_id: Whether to add agent ID to observations (default: False)
        """
        super(AgentSpecificStateWrapper, self).__init__(env)
        self.use_agent_specific_state = use_agent_specific_state
        self.add_distance_state = add_distance_state
        self.add_xy_state = add_xy_state
        self.add_visible_state = add_visible_state
        self.add_center_xy = add_center_xy
        self.add_enemy_action_state = add_enemy_action_state
        self.add_move_state = add_move_state
        self.add_local_obs = add_local_obs
        self.use_mustalive = use_mustalive

    def reset(self):
        """Reset the environment and return initial observations and states."""
        obs, state = self.env.reset()

        # Agent ID functionality is now handled by AgentIDWrapper

        if self.use_agent_specific_state:
            # Return agent-specific states for each agent
            agent_specific_states = [self.get_state_agent(agent_id) for agent_id in range(self.n_agents)]
            return obs, agent_specific_states
        else:
            # Return the original state for all agents
            return obs, state

    def step(self, actions):
        """Take a step in the environment and return agent-specific states if enabled."""
        reward, terminated, info = self.env.step(actions)

        # Return the result
        return reward, terminated, info

    def get_state_agent(self, agent_id):
        """
        Returns an agent-specific global state for the given agent_id.

        This state includes:
        - The agent's own features
        - Features of allies relative to this agent
        - Features of enemies relative to this agent
        - Movement features
        - Optional additional features based on configuration
        """
        # If agent-specific state is disabled, return the regular state
        if not self.use_agent_specific_state:
            return self.get_state()

        # Get the unit for this agent
        unit = self.get_unit_by_id(agent_id)

        # Get dimensions for different feature types
        nf_al = 2 + self.shield_bits_ally + self.unit_type_bits  # health, cooldown, shield, unit_type
        nf_en = 1 + self.shield_bits_enemy + self.unit_type_bits  # health, shield, unit_type

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
        center_x = self.map_x / 2
        center_y = self.map_y / 2

        # Skip if agent is dead and use_mustalive is True
        if (self.use_mustalive and unit.health <= 0):
            # Return zeros for all features
            state = np.concatenate((
                own_feats.flatten(),
                ally_state.flatten(),
                enemy_state.flatten()
            ))

            # Add move state if enabled
            if self.add_move_state:
                move_feats = np.zeros(self.n_actions_move, dtype=np.float32)
                state = np.append(state, move_feats)

            # Add local observation if enabled
            if self.add_local_obs:
                local_obs = self.get_obs_agent(agent_id)
                state = np.append(state, local_obs)

            # Add agent ID if enabled
            if self.env.use_agent_id:
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
            own_feats[own_idx] = (x - center_x) / self.max_distance_x  # center_x
            own_feats[own_idx + 1] = (y - center_y) / self.max_distance_y  # center_y
            own_idx += 2

        # Add shield if applicable
        if self.shield_bits_ally > 0:
            max_shield = self.unit_max_shield(unit)
            if max_shield > 0:
                own_feats[own_idx] = unit.shield / max_shield  # shield
            own_idx += 1

        # Add unit type if applicable
        if self.unit_type_bits > 0:
            type_id = self.get_unit_type_id(unit, True)
            own_feats[own_idx + type_id] = 1  # unit_type
            own_idx += self.unit_type_bits

        # Add last action if enabled
        if self.state_last_action:
            own_feats[own_idx:own_idx + self.n_actions] = self.last_action[agent_id]

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
                    ally_state[i, idx] = (al_x - center_x) / self.max_distance_x  # center_x
                    ally_state[i, idx + 1] = (al_y - center_y) / self.max_distance_y  # center_y
                    idx += 2

                # Add shield if applicable
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(al_unit)
                    if max_shield > 0:
                        ally_state[i, idx] = al_unit.shield / max_shield  # shield
                    idx += 1

                # Add unit type if applicable
                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(al_unit, True)
                    ally_state[i, idx + type_id] = 1  # unit_type
                    idx += self.unit_type_bits

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
                if self.state_last_action:
                    ally_state[i, idx:idx + self.n_actions] = self.last_action[al_id]

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
                    enemy_state[e_id, idx] = (e_x - center_x) / self.max_distance_x  # center_x
                    enemy_state[e_id, idx + 1] = (e_y - center_y) / self.max_distance_y  # center_y
                    idx += 2

                # Add shield if applicable
                if self.shield_bits_enemy > 0:
                    max_shield = self.unit_max_shield(e_unit)
                    if max_shield > 0:
                        enemy_state[e_id, idx] = e_unit.shield / max_shield  # shield
                    idx += 1

                # Add unit type if applicable
                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(e_unit, False)
                    enemy_state[e_id, idx + type_id] = 1  # unit_type
                    idx += self.unit_type_bits

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
                    enemy_state[e_id, idx] = avail_actions[self.n_actions_no_attack + e_id]  # available_to_attack

        # Combine all features
        state = np.concatenate((
            own_feats.flatten(),
            ally_state.flatten(),
            enemy_state.flatten()
        ))

        # Add move state if enabled
        if self.add_move_state:
            move_feats = np.zeros(self.n_actions_move, dtype=np.float32)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]  # Skip no-op and stop
            state = np.append(state, move_feats)

        # Add local observation if enabled
        if self.add_local_obs:
            local_obs = self.get_obs_agent(agent_id)
            state = np.append(state, local_obs)

        # Add agent ID if enabled
        if self.env.use_agent_id:
            agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
            agent_id_feats[agent_id] = 1.0
            state = np.append(state, agent_id_feats)

        return state

    def get_state_size(self):
        """Returns the size of the agent-specific global state."""
        if not self.use_agent_specific_state:
            return super(AgentSpecificStateWrapper, self).get_state_size()

        # Calculate the size of the agent-specific state
        nf_al = 2 + self.shield_bits_ally + self.unit_type_bits  # health, cooldown, shield, unit_type
        nf_en = 1 + self.shield_bits_enemy + self.unit_type_bits  # health, shield, unit_type

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

        # Calculate total size
        own_feats = nf_al
        ally_feats = (self.n_agents - 1) * nf_al
        enemy_feats = self.n_enemies * nf_en

        size = own_feats + ally_feats + enemy_feats

        # Add move state if enabled
        if self.add_move_state:
            size += self.n_actions_move

        # Add local observation if enabled
        if self.add_local_obs:
            size += self.get_obs_size()

        # Add agent ID if enabled
        if self.env.use_agent_id:
            size += self.n_agents

        return size

    def get_env_info(self):
        """
        Returns environment information with updated observation and state sizes.
        """
        # Get the original environment information
        env_info = self.env.get_env_info()

        # Update state size if agent-specific state is enabled
        if self.use_agent_specific_state:
            env_info['state_shape'] = self.get_state_size()

        return env_info


