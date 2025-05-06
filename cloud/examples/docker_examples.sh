#!/bin/bash
# docker_examples.sh - Examples of using the build_and_push.sh script

# Make sure the directory exists
mkdir -p cloud/examples

# ── Colors for output ─────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  MARL PPO Suite Docker Build & Push Examples${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"

echo -e "${GREEN}Example 1: Build and test locally${NC}"
echo -e "${YELLOW}./cloud/build_and_push.sh --test${NC}"
echo ""

echo -e "${GREEN}Example 2: Build with a specific version tag${NC}"
echo -e "${YELLOW}./cloud/build_and_push.sh --tag v1.0.0${NC}"
echo ""

echo -e "${GREEN}Example 3: Build and push to Docker Hub${NC}"
echo -e "${YELLOW}./cloud/build_and_push.sh --tag v1.0.0 --push${NC}"
echo ""

echo -e "${GREEN}Example 4: Build with a custom name and push to a specific registry${NC}"
echo -e "${YELLOW}./cloud/build_and_push.sh --name custom-mappo --registry ghcr.io/username --tag latest --push${NC}"
echo ""

echo -e "${GREEN}Example 5: Build with a different Dockerfile${NC}"
echo -e "${YELLOW}./cloud/build_and_push.sh --tag latest -f cloud/runpods/Dockerfile.latest${NC}"
echo ""

echo -e "${GREEN}Example 6: Complete workflow - build, test, and push with versioning${NC}"
echo -e "${YELLOW}./cloud/build_and_push.sh --tag \$(date +%Y%m%d)-\$(git rev-parse --short HEAD) --test --push${NC}"
echo ""

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  For more options, run: ./cloud/build_and_push.sh --help${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"

# This is just an example file, not meant to be executed
exit 0
