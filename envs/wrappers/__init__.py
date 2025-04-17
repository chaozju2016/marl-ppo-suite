"""
StarCraft 2 environment wrappers for MAPPO.
"""
from envs.wrappers.base_wrapper import BaseWrapper
from envs.wrappers.agent_id_wrapper import AgentIDWrapper
from envs.wrappers.agent_specific_state_wrapper import AgentSpecificStateWrapper
from envs.wrappers.death_masking_wrapper import DeathMaskingWrapper

__all__ = ["BaseWrapper", "AgentIDWrapper", "AgentSpecificStateWrapper", "DeathMaskingWrapper"]
