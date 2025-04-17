"""
Environment factory for creating StarCraft 2 environments with various wrappers.
"""
from smac.env import StarCraft2Env
from envs.wrappers import AgentIDWrapper, DeathMaskingWrapper, AgentSpecificStateWrapper


def create_env(args, is_eval=False):
    """
    Create a StarCraft 2 environment with the specified wrappers based on arguments.

    Args:
        args: Arguments containing environment configuration
        is_eval: Whether this is an evaluation environment (default: False).
               Currently not used, but kept for potential future differentiation
               between training and evaluation environments.

    Returns:
        env: The wrapped environment
    """
    # Create base StarCraft2Env with safe defaults
    env = StarCraft2Env(map_name=args.map_name, difficulty=args.difficulty, obs_last_action=args.obs_last_actions)

    # Apply wrappers in the correct order (innermost to outermost)

    # 1. Always apply death masking first (innermost wrapper)
    env = DeathMaskingWrapper(env, use_death_masking=args.use_death_masking)

    # 2. Apply agent ID wrapper
    env = AgentIDWrapper(env,
                        use_agent_id=args.use_agent_id)

    # 3. Apply agent-specific state wrapper (if enabled) as the outermost wrapper
    if args.use_agent_specific_state:
        # Prepare parameters with safe defaults
        env = AgentSpecificStateWrapper(
            env,
            use_agent_specific_state=True,
            add_distance_state=args.add_distance_state,
            add_xy_state=args.add_xy_state,
            add_visible_state=args.add_visible_state,
            add_center_xy=args.add_center_xy,
            add_enemy_action_state=args.add_enemy_action_state,
            add_move_state=args.add_move_state,
            add_local_obs=args.add_local_obs,
            use_mustalive=args.use_mustalive
            # No use_agent_id parameter - handled by AgentIDWrapper
        )

    return env
