"""
Environment factory for creating StarCraft 2 environments with various wrappers.
"""
from smac.env import StarCraft2Env
from envs.env_vectorization import SubprocVecEnv, DummyVecEnv
from envs.wrappers import AgentIDWrapper, DeathMaskingWrapper, FeaturePrunedStateWrapper


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

    return env


def make_env(args, seed=None, rank=None, is_eval=False):
    """
    Helper function to create an environment with a given seed.

    Args:
        args: Arguments object containing environment configuration
        seed: Random seed (currently unused as StarCraft2Env doesn't support seeding)
        rank: Environment rank (for seeding, currently unused)
        is_eval: Whether this is an evaluation environment

    Returns:
        A function that creates the environment when called
    """
    def _thunk():

        env = StarCraft2Env(map_name=args.map_name, difficulty=args.difficulty, obs_last_action=args.obs_last_actions)

        env = FeaturePrunedStateWrapper(
            env,
            use_agent_specific_state=args.use_agent_specific_state,
            add_distance_state=args.add_distance_state,
            add_xy_state=args.add_xy_state,
            add_visible_state=args.add_visible_state,
            add_center_xy=args.add_center_xy,
            add_enemy_action_state=args.add_enemy_action_state,
            use_mustalive=args.use_mustalive,
            add_agent_id=args.use_agent_id
        )

        return env

    return _thunk

def make_vec_envs(args, seed, num_processes, is_eval=False):
    """
    Create vectorized environments.

    Args:
        args: Arguments object containing environment configuration
        seed: Random seed
        num_processes: Number of parallel environments
        is_eval: Whether these are evaluation environments

    Returns:
        Vectorized environments
    """
    envs = [make_env(args, seed, i, is_eval=is_eval) for i in range(num_processes)]

    # If only one environment, use DummyVecEnv for simplicity
    if len(envs) == 1:
        return DummyVecEnv(envs)
    else:
        return SubprocVecEnv(envs)