import argparse
import atexit
from runners.mlp_runner import Runner
from runners.rnn_runner import RecurrentRunner
from runners.mappo_runner import MAPPORunner
from utils.sc2_utils import kill_sc2_processes

# Register cleanup function to kill SC2 processes on exit
def cleanup_sc2_processes():
    print("\nCleaning up any lingering SC2 processes...")
    killed = kill_sc2_processes()
    if killed > 0:
        print(f"Killed {killed} SC2 processes during cleanup")
    else:
        print("No SC2 processes needed to be cleaned up")

# Register the cleanup function to run when the script exits
atexit.register(cleanup_sc2_processes)

def parse_args():
    """
    Parse command line arguments for MAPPO training.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser("MAPPO for StarCraft")

    # Algorithm parameters
    parser.add_argument("--algo", type=str, default="as_mappo_rnn",
                        choices=["mappo", "mappo_rnn", "as_mappo_rnn"],
                        help="Which algorithm to use")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True,
                        help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True,
                        help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--max_steps", type=int, default=1000000,
                        help="Number of environment steps to train on")
    # NOTE: rollout_threads supported only in as_mappo_rnn
    parser.add_argument("--n_rollout_threads", type=int, default=8,
                        help="Number of parallel environments (default: 8)")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of threads for evaluation (default: 1)")


    # Environment parameters
    parser.add_argument("--map_name", type=str, default="3m",
                        help="Which SMAC map to run on")
    parser.add_argument("--difficulty", type=str, default="7",
                        help="Difficulty of the SMAC map")
    parser.add_argument("--obs_last_actions", action="store_true", default=False,
                        help="Whether to include last actions in observations (default: False)")

    # Agent ID parameters
    parser.add_argument("--use_agent_id", action="store_false", default=True,
        help="Whether to add agent_id to observation (default: True)")

    # Death masking parameters
    parser.add_argument("--use_death_masking", action="store_true", default=False,
        help="Whether to use death masking to exclude dead agents from training (default: True)")

    # Agent-specific state parameters
    parser.add_argument("--use_agent_specific_state", action="store_false", default=True,
        help="Whether to use agent-specific global state (default: False)")
    parser.add_argument("--add_distance_state", action="store_true", default=False,
        help="Whether to add distance features to the state (default: False)")
    parser.add_argument("--add_xy_state", action="store_true", default=False,
        help="Whether to add relative x,y coordinates to the state (default: False)")
    parser.add_argument("--add_visible_state", action="store_true", default=False,
        help="Whether to add visibility information to the state (default: False)")
    parser.add_argument("--add_center_xy", action="store_false", default=True,
        help="Whether to add center-relative coordinates to the state (default: True)")
    parser.add_argument("--add_enemy_action_state", action="store_true", default=False,
        help="Whether to add enemy action availability to the state (default: False)")
    parser.add_argument("--use_mustalive", action="store_false", default=True,
        help="Whether to only return non-zero state for alive agents (default: True)")

    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Initial learning rate")
    parser.add_argument("--optimizer_eps", type=float, default=1e-5,
                        help="Epsilon for optimizer")
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        help='Use a linear schedule on the learning rate')
    parser.add_argument("--min_lr", type=float, default=1e-5,
                        help="Minimum learning rate")

    # Network parameters
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers")
    parser.add_argument("--use_rnn", action="store_true", default=False,
                        help="Whether to use RNN networks (default: False)")
    parser.add_argument("--rnn_layers", type=int, default=1,
                        help="Number of RNN layers")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")
    parser.add_argument("--fc_layers", type=int, default=1,
                        help="Number of fc layers in actor/critic network")
    parser.add_argument("--actor_gain", type=float, default=0.01,
                        help="Gain of the actor final linear layer")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        help="Apply layernorm to the inputs (default: True)")
    parser.add_argument("--use_value_norm", action="store_true",
                        help="Use running mean and std to normalize returns (default: False)")
    parser.add_argument("--value_norm_type", type=str, default="welford", choices=["welford", "ema"],
                        help="Type of value normalizer to use: 'welford' (original) or 'ema' (exponential moving average)")
    parser.add_argument("--use_reward_norm", action="store_false",
                        help="Use running mean and std to normalize rewards (default: True)")
    parser.add_argument("--reward_norm_type", type=str, default="efficient", choices=["efficient", "ema"],
                        help="Type of reward normalizer to use: 'efficient' (standard) or 'ema' (exponential moving average)")


    # PPO parameters
    parser.add_argument("--n_steps", type=int, default=400,
                        help="Number of steps to run in each environment per policy rollout (default: 400)")
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help="Number of epochs for PPO")
    parser.add_argument("--use_clipped_value_loss", action="store_false",
                        help="Use clipped value loss (default: True)")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help="PPO clip parameter")
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help="Number of mini-batches for PPO")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--use_gae", action="store_false",
                        help="Use Generalized Advantage Estimation (default: True)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="Lambda for Generalized Advantage Estimation")
    parser.add_argument("--use_proper_time_limits", action="store_false",
                        help="Use proper time limits (default: True)")
    parser.add_argument("--use_max_grad_norm", action="store_false",
                        help="Use max gradient norm (default: True)")
    parser.add_argument("--max_grad_norm", type=float, default=10,
                        help="Max gradient norm")
    parser.add_argument("--use_huber_loss", action="store_true",
                        help="Use huber loss (default: False)")
    parser.add_argument("--huber_delta", type=float, default=10.0,
                        help="Delta for huber loss")

    # Evaluation parameters
    parser.add_argument("--use_eval", action="store_false",
                        help="Evaluate the model during training (default: True)")
    parser.add_argument("--eval_interval", type=int, default=5000,
                        help="Evaluate the model every eval_interval steps")
    parser.add_argument("--eval_episodes", type=int, default=32,
                        help="Number of episodes for evaluation")

    # Save parameters
    parser.add_argument("--save_interval", type=int, default=100000,
                        help="Save the model every save_interval steps")
    parser.add_argument("--save_dir", type=str, default="./model",
                        help="Directory to save the model")
    parser.add_argument("--save_replay", type=bool, default=False,
                        help="Whether to save the replay")
    parser.add_argument("--replay_dir", type=str, default="./replay",
                        help="Directory to save the replay")

    # Performance monitoring
    parser.add_argument("--show_performance_metrics", action="store_true", default=False,
                        help="Show detailed performance metrics during training")

    return parser.parse_args()


def main():
    args = parse_args()
    runner = None

    print("=============================")
    print(f"Training {args.algo.upper()} for StarCraft")
    print("=============================")
    print(f"Map: {args.map_name}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Seed: {args.seed}")
    print(f"Algorithm: {args.algo}")
    print(f"Reward Norm: {args.use_reward_norm} ({args.reward_norm_type})")
    print(f"Value Norm: {args.use_value_norm} ({args.value_norm_type})")

    # Print agent-specific state parameters if using as_mappo_rnn
    if args.algo == "as_mappo_rnn":
        print("\nFeature Pruned AS State Parameters:")
        print(f"  Use Agent-Specific State: {args.use_agent_specific_state}")
        print(f"  Add Distance State: {args.add_distance_state}")
        print(f"  Add XY State: {args.add_xy_state}")
        print(f"  Add Visible State: {args.add_visible_state}")
        print(f"  Add Center XY: {args.add_center_xy}")
        print(f"  Add Enemy Action State: {args.add_enemy_action_state}")
        print(f"  Use Must Alive: {args.use_mustalive}")
        print(f"  Add Agent ID: {args.use_agent_id}")

        print("\nTraining Parameters:")
        print(f" Rollout Threads: {args.n_rollout_threads}")
        print(f"  Steps: {args.n_steps}")
        print(f"  Epochs: {args.ppo_epoch}")
        print(f"  Mini-Batches: {args.num_mini_batch}")

    # print(f"Using {'CUDA' if args.cuda and torch.cuda.is_available() else 'CPU'}")
    # print(f"Using {'recurrent' if args.use_recurrent_policy else 'MLP'} policy")
    print("=============================")

    try:
        if args.algo == "mappo":
            runner = Runner(args)
        elif args.algo == "mappo_rnn":
            runner = RecurrentRunner(args)
        elif args.algo == "as_mappo_rnn":
            runner = MAPPORunner(args)
        else:
            raise ValueError(f"Invalid algorithm: {args.algo}")

        runner.run()
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up environments even if an error occurs
        if runner is not None:
            try:
                if hasattr(runner, 'envs') and runner.envs is not None:
                    print("Closing training environments...")
                    runner.envs.close()
            except Exception as e:
                print(f"Error closing training environments: {e}")

            try:
                if hasattr(runner, 'eval_envs') and runner.eval_envs is not None:
                    print("Closing evaluation environments...")
                    runner.eval_envs.close()
            except Exception as e:
                print(f"Error closing evaluation environments: {e}")

            try:
                if hasattr(runner, 'logger') and runner.logger is not None:
                    print("Closing logger...")
                    runner.logger.close()
            except Exception as e:
                print(f"Error closing logger: {e}")

        # Force kill any remaining SC2 processes
        print("Ensuring all SC2 processes are terminated...")
        kill_sc2_processes()

if __name__ == "__main__":
    main()
    # TODO: Check mapsc2_2s3z with epsilon 1e-5, does it make change or not (because just saw that win rate went up to 0.78 and down to 0.0)



