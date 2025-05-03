#!/bin/bash
# Make all scripts executable

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Making all scripts executable...${NC}"

chmod +x cloud/train.sh
chmod +x cloud/train_simple.sh
chmod +x cloud/test_docker.sh
chmod +x cloud/runpods/entrypoint.sh

echo -e "${GREEN}All scripts are now executable.${NC}"
