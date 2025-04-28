"""
Environment factory for creating StarCraft 2 environments with various wrappers.
"""
from envs.env_vectorization import SubprocVecEnv, DummyVecEnv


def create_env(args, is_eval=False):
    """
    LightMappo version only
    Create a StarCraft 2 environment with the specified wrappers based on arguments.

    Args:
        args: Arguments containing environment configuration
        is_eval: Whether this is an evaluation environment (default: False).
               Currently not used, but kept for potential future differentiation
               between training and evaluation environments.
    Returns:
        env: The wrapped environment
    """
    from smac.env import StarCraft2Env
    from envs.wrappers import AgentIDWrapper, DeathMaskingWrapper
    # Create base StarCraft2Env with safe defaults
    env = StarCraft2Env(map_name=args.map_name, difficulty=args.difficulty, obs_last_action=args.obs_last_actions)


    # Apply wrappers in the correct order (innermost to outermost)

    # 1. Always apply death masking first (innermost wrapper)
    env = DeathMaskingWrapper(env, use_death_masking=args.use_death_masking)

    # 2. Apply agent ID wrapper
    env = AgentIDWrapper(env,
                        use_agent_id=args.use_agent_id)

    return env


def make_env(args, rank=None, is_eval=False):
    """
    Helper function to create an environment with a given seed.

    Args:
        args: Arguments object containing environment configuration
        rank: Environment rank (for seeding, currently unused)
        is_eval: Whether this is an evaluation environment

    Returns:
        A function that creates the environment when called
    """
    def _thunk():
        
        env_name = args.env_name

        if env_name == "smacv1":
            if hasattr(args, "use_fp_wrapper") and args.use_fp_wrapper:
                # Legacy Version using wrapper.
                from smac.env import StarCraft2Env
                from envs.wrappers import FeaturePrunedStateWrapper

                env = StarCraft2Env(map_name=args.map_name)

                env = FeaturePrunedStateWrapper(
                    env,
                    use_agent_specific_state=args.use_agent_specific_state,
                    add_distance_state=args.add_distance_state,
                    add_xy_state=args.add_xy_state,
                    add_visible_state=args.add_visible_state,
                    add_center_xy=args.add_center_xy,
                    add_enemy_action_state=args.add_enemy_action_state,
                    use_mustalive=args.use_mustalive,
                    use_agent_id=args.use_agent_id
                )
            else: 
                # New Version using SMACv1Env.
                from envs.smacv1.Starcraft2_Env import StarCraft2Env as SMACv1Env

                env = SMACv1Env(
                    map_name=args.map_name,
                    state_type= args.state_type
                )
        elif env_name == "smacv2":
            from envs.smacv2 import SMACv2Env
            
            env = SMACv2Env(args)
        else:
            raise ValueError(f"Unknown environment name: {env_name}")

        return env

    return _thunk

def make_vec_envs(args, num_processes, is_eval=False):
    """
    Create vectorized environments.

    Args:
        args: Arguments object containing environment configuration
        num_processes: Number of parallel environments
        is_eval: Whether these are evaluation environments

    Returns:
        Vectorized environments
    """
    # Create environment thunks
    envs = [make_env(args, i, is_eval=is_eval) for i in range(num_processes)]

    # If only one environment, use DummyVecEnv for simplicity
    if len(envs) == 1:
        return DummyVecEnv(envs)
    else:
        return SubprocVecEnv(envs)