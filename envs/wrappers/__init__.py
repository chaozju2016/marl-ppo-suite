"""
StarCraft 2 environment wrappers for MAPPO.
"""
from envs.wrappers.base_wrapper import BaseWrapper
from envs.wrappers.agent_id_wrapper import AgentIDWrapper
from envs.wrappers.death_masking_wrapper import DeathMaskingWrapper
from envs.wrappers.feature_pruned_state_wrapper import FeaturePrunedStateWrapper

__all__ = [
    "BaseWrapper",
    "AgentIDWrapper",
    "DeathMaskingWrapper",
    "FeaturePrunedStateWrapper",
]
