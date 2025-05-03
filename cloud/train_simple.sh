#!/bin/bash
# Script to run MARL PPO Suite training on cloud GPU instances

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MARL PPO Suite Cloud Training Script ===${NC}"
echo ""

# Set PYTHONPATH if needed
export PYTHONPATH=$PYTHONPATH:/app

# Run the command with all provided arguments
CMD="python train.py $@"
# CMD="python -u train.py $@"
#↑ add -u  (or use PYTHONUNBUFFERED=1) to see the logs

echo -e "${YELLOW}Running command:${NC}"
echo -e "$CMD"
echo ""
eval $CMD
