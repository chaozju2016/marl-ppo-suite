#!/bin/bash
# build_and_push.sh - Build, test, and push Docker images for MARL PPO Suite
# Following CI/CD best practices with versioning support

set -e  # Exit on any error

# ── Configuration ──────────────────────────────────────────────────────────────
# Default values
IMAGE_NAME="marl-ppo-suite"
DEFAULT_TAG="latest"
DOCKERFILE_PATH="cloud/runpods/Dockerfile"
TEST_ENABLED=false
PUSH_ENABLED=false
REGISTRY=""  # Docker Hub by default

# ── Colors for output ─────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ── Functions ─────────────────────────────────────────────────────────────────
print_header() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
}

print_step() {
    echo -e "${GREEN}➤ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_error() {
    echo -e "${RED}✖ $1${NC}"
}

show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -t, --tag TAG              Set image tag (default: latest)"
    echo "  -n, --name NAME            Set image name (default: marl-ppo-suite)"
    echo "  -r, --registry REGISTRY    Set registry (default: Docker Hub username)"
    echo "  -f, --file DOCKERFILE      Path to Dockerfile (default: cloud/runpods/Dockerfile)"
    echo "  --test                     Run test after build"
    echo "  --push                     Push image after build"
    echo ""
    echo "Examples:"
    echo "  $0 --tag v1.0.0 --test --push"
    echo "  $0 --name custom-mappo --registry ghcr.io/username --tag latest"
    echo "  $0 --file cloud/runpods/Dockerfile.latest --tag latest-cuda11.8"
    echo ""
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_info "Docker is installed."
}

build_image() {
    local full_tag="$1"
    print_step "Building Docker image: $full_tag"

    # Build the Docker image
    docker build \
        -t "$full_tag" \
        -f "$DOCKERFILE_PATH" \
        .

    print_info "Image built successfully: $full_tag"
}

test_image() {
    local full_tag="$1"
    print_step "Testing Docker image: $full_tag"

    # Create directory for runs if it doesn't exist
    mkdir -p runs

    # Check if running on macOS
    if [[ "$(uname)" == "Darwin" ]]; then
        print_info "Detected macOS. Using Docker Desktop GPU passthrough..."
        # On macOS, Docker Desktop handles GPU passthrough differently
        docker run --rm \
            --name marl-ppo-suite-test \
            -v "$(pwd)/runs:/app/runs" \
            "$full_tag" \
            --map_name 3m --algo mappo --max_steps 1000 --n_rollout_threads 1 --n_steps 100
    else
        # On Linux, use the --gpus flag
        docker run --rm \
            --name marl-ppo-suite-test \
            --gpus all \
            -v "$(pwd)/runs:/app/runs" \
            "$full_tag" \
            --map_name 3m --algo mappo --max_steps 1000 --n_rollout_threads 1 --n_steps 100
    fi

    print_info "Test completed successfully!"
}

push_image() {
    local full_tag="$1"
    print_step "Pushing Docker image: $full_tag"

    # Check if user is logged in to Docker
    if ! docker info | grep -q "Username"; then
        print_info "Not logged in to Docker registry. Logging in..."
        docker login
    fi

    docker push "$full_tag"
    print_info "Image pushed successfully: $full_tag"
}

# ── Parse command line arguments ─────────────────────────────────────────────
TAG="$DEFAULT_TAG"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -f|--file)
            DOCKERFILE_PATH="$2"
            shift 2
            ;;
        --test)
            TEST_ENABLED=true
            shift
            ;;
        --push)
            PUSH_ENABLED=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# ── Main execution ─────────────────────────────────────────────────────────────
print_header "MARL PPO Suite Docker Build & Push"

# Check Docker installation
check_docker

# If no registry is provided and push is enabled, ask for Docker Hub username
if [[ -z "$REGISTRY" && "$PUSH_ENABLED" == true ]]; then
    print_info "No registry specified. Using Docker Hub."
    print_info "Enter your Docker Hub username:"
    read -r REGISTRY
fi

# Construct the full image tag
if [[ -n "$REGISTRY" ]]; then
    FULL_TAG="$REGISTRY/$IMAGE_NAME:$TAG"
else
    FULL_TAG="$IMAGE_NAME:$TAG"
fi

print_info "Building image with the following configuration:"
echo "  - Image name: $IMAGE_NAME"
echo "  - Tag: $TAG"
echo "  - Full tag: $FULL_TAG"
echo "  - Dockerfile: $DOCKERFILE_PATH"
echo "  - Test enabled: $TEST_ENABLED"
echo "  - Push enabled: $PUSH_ENABLED"

# Build the Docker image
build_image "$FULL_TAG"

# Run tests if enabled
if [[ "$TEST_ENABLED" == true ]]; then
    test_image "$FULL_TAG"
fi

# Push the image if enabled
if [[ "$PUSH_ENABLED" == true ]]; then
    push_image "$FULL_TAG"
fi

print_header "Build & Push Process Completed Successfully"
echo ""
echo "Image details:"
echo "  - Tag: $FULL_TAG"

if [[ "$PUSH_ENABLED" == true ]]; then
    echo ""
    echo "To use this image on RunPods.io:"
    echo "1. Select 'Docker' as the template type"
    echo "2. Enter '$FULL_TAG' as the Docker image"
    echo "3. Configure your pod resources and settings"
    echo "4. Deploy and enjoy!"
fi

exit 0
