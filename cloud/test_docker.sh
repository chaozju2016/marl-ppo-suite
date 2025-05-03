#!/bin/bash
# Test the Docker setup for MARL PPO Suite

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Testing Docker Setup for MARL PPO Suite ===${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Build the Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t marl-ppo-suite:pytorch2.5 -f cloud/runpods/Dockerfile .

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build Docker image.${NC}"
    exit 1
fi

echo -e "${GREEN}Docker image built successfully.${NC}"

# Create directory for runs if it doesn't exist
mkdir -p runs

# Run a simple test with minimal steps
echo -e "${YELLOW}Running a quick test with minimal steps...${NC}"

# Check if running on macOS
if [[ "$(uname)" == "Darwin" ]]; then
    echo -e "${YELLOW}Detected macOS. Using Docker Desktop GPU passthrough...${NC}"
    # On macOS, Docker Desktop handles GPU passthrough differently
    docker run --rm \
        --name marl-ppo-suite-test \
        -v $(pwd)/runs:/app/runs \
        --env-file .env  \
        marl-ppo-suite:pytorch2.5 \
        --map_name 3m --algo mappo --max_steps 100 --n_rollout_threads 1 --n_steps 100 --use_rnn
else
    # On Linux, use the --gpus flag
    docker run --rm \
        --name marl-ppo-suite-test \
        --gpus all \
        -v $(pwd)/runs:/app/runs \
        --env-file .env  \
        marl-ppo-suite:pytorch2.5 \
        --map_name 3m --algo mappo --max_steps 100 --n_rollout_threads 1 --n_steps 100 --use_rnn
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}Test failed.${NC}"
    echo -e "${YELLOW}Check the logs for details.${NC}"
    exit 1
fi

echo -e "${GREEN}Test completed successfully!${NC}"
echo -e "${YELLOW}Check the logs directory for training logs.${NC}"
echo ""
echo -e "${GREEN}Your Docker setup is working correctly.${NC}"
echo -e "${YELLOW}You can now deploy to RunPods.io using the instructions in cloud/README.md${NC}"
