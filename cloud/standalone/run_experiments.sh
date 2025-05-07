#!/bin/bash
# Script to run multiple MARL PPO Suite experiments sequentially
# Edit the experiment configurations below to customize your logs

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MARL PPO Suite Experiment Runner ===${NC}"
echo ""

# Set optimal thread settings for Linux
# These must be set before Python starts
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 
# export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 # unbuffered output

echo -e "${YELLOW}Thread settings:${NC}"
echo -e "  OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo -e "  MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
# echo -e "  NUMEXPR_NUM_THREADS: ${NUMEXPR_NUM_THREADS}"
echo -e "  PYTHONUNBUFFERED: ${PYTHONUNBUFFERED}"
echo ""

# Activate the environment using the mpo function
source ~/.bashrc
mpo

# Create logs directory if it doesn't exist
mkdir -p logs

# Load .env file if it exists
if [ -f .env ]; then
    echo -e "${GREEN}Loading environment variables from .env file...${NC}"
    set -a
    source .env
    set +a
fi

# Function to run an experiment
run_experiment() {
    echo -e "${YELLOW}Starting experiment: $1${NC}"
    python train.py $1 > logs/$(date +%Y%m%d_%H%M%S)_$2.log 2>&1

    # Check if experiment completed successfully
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Experiment completed successfully!${NC}"
    else
        echo -e "${RED}Experiment failed with exit code $?${NC}"
    fi

    # Wait a bit before starting the next experiment
    sleep 10
}

# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================
# Define experiments as argument string | name pairs
# =============================================================================

# Define experiments
EXPERIMENTS=(
     "--algo happo \
     --env_name smacv2 \
     --map_name zerg_5_vs_5 \
     --n_rollout_threads 16 \
     --max_steps 10000000 \
     --log_interval 16000 \
     --eval_interval 80000 \
     --state_type AS \
     --n_steps 200 \
     --ppo_epoch 5 \
     --clip_param 0.05 \
     --use_rnn \
     --capture_video \
     --capture_video_interval 200000 \
     --use_wandb|zerg_5_vs_5_happo",

    "--algo mappo \
     --env_name smacv2 \
     --map_name terran_5_vs_5 \
     --n_rollout_threads 16 \
     --max_steps 10000000 \
     --log_interval 16000 \
     --eval_interval 80000 \
     --state_type AS \
     --n_steps 200 \
     --ppo_epoch 5 \
     --clip_param 0.05 \
     --use_rnn \
     --capture_video \
     --capture_video_interval 200000 \
     --use_wandb|terran_5_vs_5_mappo",

    "--algo happo \
     --env_name smacv2 \
     --map_name terran_5_vs_5 \
     --n_rollout_threads 16 \
     --max_steps 10000000 \
     --log_interval 16000 \
     --eval_interval 80000 \
     --state_type AS \
     --n_steps 200 \
     --ppo_epoch 5 \
     --clip_param 0.05 \
     --use_rnn \
     --capture_video \
     --capture_video_interval 200000 \
     --use_wandb|terran_5_vs_5_happo"
)

# Run all experiments in sequence
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r args name <<< "$exp"
    run_experiment "$args" "$name"
done

# =============================================================================
# END OF EXPERIMENT CONFIGURATIONS
# =============================================================================

echo -e "${GREEN}All experiments completed!${NC}"
