#!/bin/bash
# RunPods.io entrypoint script for MARL PPO Suite
set -e

# helper function  (add near top of entrypoint.sh)
stop_pod_if_possible () {
    if command -v runpodctl >/dev/null 2>&1 && [[ -n "$RUNPOD_POD_ID" ]]; then
        echo "Stopping RunPod pod $RUNPOD_POD_ID ..."
        runpodctl stop pod "$RUNPOD_POD_ID"
    else
        echo "runpodctl not found (local run) – skipping pod stop."
    fi
}

# Set up SSH key from RunPod.io PUBLIC_KEY environment variable
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p /root/.ssh
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
    echo "SSH public key configured."
fi

/usr/sbin/sshd -D & # Start SSH server in the background

# trap both Ctrl‑C and platform SIGTERM
trap 'stop_pod_if_possible; exit 0' SIGINT SIGTERM
# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== RunPods.io MARL PPO Suite Training ===${NC}"

# Create runs directory if it doesn't exist
mkdir -p runs

# Redirect all output to log file
exec > >(tee -a runs/training.log) 2>&1

# Print system information
echo -e "${YELLOW}System Information:${NC}"
echo -e "Date: $(date)"
echo -e "Hostname: $(hostname)"
echo -e "Kernel: $(uname -r)"
echo -e "CPU: $(lscpu | grep 'Model name' | cut -f 2 -d ':' | awk '{$1=$1}1')"
echo -e "Memory: $(free -h | grep Mem | awk '{print $2}')"

# Print GPU information
echo -e "${YELLOW}GPU Information:${NC}"
nvidia-smi || echo -e "${RED}No GPU detected${NC}"

# Print environment variables
echo -e "${YELLOW}Environment Variables:${NC}"
echo -e "SC2PATH: $SC2PATH"
echo -e "PYTHONPATH: $PYTHONPATH"

# Check if StarCraft II is installed correctly
echo -e "${YELLOW}Checking StarCraft II installation:${NC}"
if [ -d "$SC2PATH" ]; then
    echo -e "${GREEN}StarCraft II found at $SC2PATH${NC}"
else
    echo -e "${RED}StarCraft II not found at $SC2PATH${NC}"
    exit 1
fi

# Check if CUDA is available
echo -e "${YELLOW}Checking CUDA availability:${NC}"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Run validation test
# echo -e "${YELLOW}Running environment validation test:${NC}"
# python -c "from smac.env import StarCraft2Env; env = StarCraft2Env(map_name='3m'); env.reset(); print('SMAC environment initialized successfully')"
# if [ $? -ne 0 ]; then
#     echo -e "${RED}SMAC environment validation failed${NC}"
#     exit 1
# fi

# Parse command line arguments
echo -e "${YELLOW}Arguments: $@${NC}"

# Run the training command
echo -e "${GREEN}Starting training...${NC}"
./cloud/train_simple.sh "$@"
EXIT_CODE=$?

# Check if training completed successfully
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
else
    echo -e "${RED}Training failed with exit code $EXIT_CODE${NC}"
fi

# Exit with training status (avoids keeping container running)
echo -e "${YELLOW}Training finished. Logs available at runs/training.log${NC}"
# echo -e "To exit: exit"
# # Sleep indefinitely to keep container running
# tail -f /dev/null
# normal completion
stop_pod_if_possible 

exit $EXIT_CODE