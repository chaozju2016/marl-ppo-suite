#!/bin/bash
# Script to run MARL PPO Suite training on cloud GPU instances

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MARL PPO Suite Cloud Training Script ===${NC}"
echo ""

# Default parameters (only the most common ones)
ENV_NAME="smacv1"
MAP_NAME="3m"
ALGO="mappo"
USE_RNN=false
MAX_STEPS=1000000
N_ROLLOUT_THREADS=8
N_STEPS=400 d
USE_WANDB=false
USE_EVAL=true
SEED=1
STATE_TYPE="EP"
USE_DEATH_MASKING=false
USE_AGENT_ID=false

# Parse command line arguments for common parameters
# This makes the most common parameters easier to use
while [[ $# -gt 0 ]]; do
    case $1 in
        --map_name) MAP_NAME="$2"; shift 2 ;;
        --algo) ALGO="$2"; shift 2 ;;
        --use_rnn) USE_RNN=true; shift ;;
        --max_steps) MAX_STEPS="$2"; shift 2 ;;
        --n_rollout_threads) N_ROLLOUT_THREADS="$2"; shift 2 ;;
        --n_steps) N_STEPS="$2"; shift 2 ;;
        --use_wandb) USE_WANDB=true; shift ;;
        --no_eval) USE_EVAL=false; shift ;;
        --seed) SEED="$2"; shift 2 ;;
        --env_name) ENV_NAME="$2"; shift 2 ;;
        --state_type) STATE_TYPE="$2"; shift 2 ;;
        --use_death_masking) USE_DEATH_MASKING=true; shift ;;
        --use_agent_id) USE_AGENT_ID=true; shift ;;
        --help|-h)
            echo -e "${GREEN}Usage:${NC}"
            echo "  $0 [options]"
            echo ""
            echo -e "${YELLOW}Common options:${NC}"
            echo "  --map_name NAME           SMAC map name (default: $MAP_NAME)"
            echo "  --algo ALGO               Algorithm (default: $ALGO)"
            echo "  --env_name NAME           Environment (default: $ENV_NAME)"
            echo "  --state_type TYPE         State type (default: $STATE_TYPE)"
            echo "  --use_rnn                 Use RNN networks"
            echo "  --max_steps N             Maximum training steps (default: $MAX_STEPS)"
            echo "  --n_rollout_threads N     Parallel environments (default: $N_ROLLOUT_THREADS)"
            echo "  --n_steps N               Steps per rollout (default: $N_STEPS)"
            echo "  --use_wandb               Enable Weights & Biases logging"
            echo "  --no_eval                 Disable evaluation during training"
            echo "  --seed N                  Random seed (default: $SEED)"
            echo "  --use_death_masking       Enable death masking (SMACv2)"
            echo "  --use_agent_id            Enable agent ID (SMACv2)"
            echo ""
            echo -e "${YELLOW}Additional options:${NC}"
            echo "  Any other arguments will be passed directly to train.py"
            echo "  For a full list of options, run: python train.py --help"
            exit 0
            ;;
        *) break ;; # Stop at the first unknown option
    esac
done

# Build the command with common parameters
CMD="python train.py \
    --algo $ALGO \
    --map_name $MAP_NAME \
    --max_steps $MAX_STEPS \
    --n_rollout_threads $N_ROLLOUT_THREADS \
    --n_steps $N_STEPS \
    --seed $SEED \
    --env_name $ENV_NAME \
    --state_type $STATE_TYPE"

# Add optional flags
[ "$USE_RNN" = true ] && CMD="$CMD --use_rnn"
[ "$USE_WANDB" = true ] && CMD="$CMD --use_wandb"
[ "$USE_EVAL" = false ] && CMD="$CMD --use_eval"
[ "$USE_DEATH_MASKING" = true ] && CMD="$CMD --use_death_masking"
[ "$USE_AGENT_ID" = true ] && CMD="$CMD --use_agent_id"


# Add any remaining arguments directly to the command
# This allows passing any other arguments that train.py supports
if [ $# -gt 0 ]; then
    CMD="$CMD $@"
fi

# Set PYTHONPATH if needed
export PYTHONPATH=$PYTHONPATH:/app

# Print the command
echo -e "${YELLOW}Running command:${NC}"
echo -e "$CMD"
echo ""

# Run the command
eval $CMD