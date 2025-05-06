#!/bin/bash
# app_setup.sh - Application setup script for MARL PPO Suite on standalone Linux machines
# This script handles repository setup, environment configuration, and experiment setup

# ── Colors and formatting ────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ── Helper functions ─────────────────────────────────────────────────────────
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

# ── Main script ─────────────────────────────────────────────────────────────
print_header "MARL PPO Suite Application Setup for Standalone Deployment"

# Check if running as root
if [ "$(id -u)" = "0" ]; then
    print_error "This script should not be run as root"
    exit 1
fi

# ── Part 1: Repository setup ─────────────────────────────────────────────────
print_step "Setting up MARL PPO Suite repository"

# Clone the repository
print_info "Cloning the repository..."
if [ ! -d "$HOME/marl-ppo-suite" ]; then
    git clone https://github.com/legalaspro/marl-ppo-suite.git
    cd marl-ppo-suite
else
    print_info "Repository already exists. Updating..."
    cd marl-ppo-suite
    git pull
fi

# Create Python virtual environment
print_info "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
print_info "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install git+https://github.com/oxwhirl/smac.git
pip install git+https://github.com/oxwhirl/smacv2.git

# ── Part 2: Environment configuration ─────────────────────────────────────────
print_step "Configuring environment"

# Create a sample .env file for wandb configuration
print_info "Creating sample .env file for wandb configuration..."
cat > .env.sample << 'EOF'
# Weights & Biases configuration
# Rename this file to .env and fill in your values
# WANDB_API_KEY=your_api_key_here
# WANDB_PROJECT=marl-ppo-suite
# WANDB_ENTITY=your_username_or_team
# SC2PATH=/path/to/StarCraftII/
# RUNS_DIR=/workspace/runs # for the cloud docker image to write in volume data
EOF

print_info "Created .env.sample file. Rename to .env and add your wandb API key to use wandb."

# Set up environment variables and shortcuts
print_info "Setting up environment variables and shortcuts..."

# Add SC2PATH if not already in .bashrc
if ! grep -q "export SC2PATH=" ~/.bashrc; then
    echo 'export SC2PATH=$HOME/StarCraftII' >> ~/.bashrc
fi

# Add optimal thread settings to .bashrc if not already there
# if ! grep -q "export OMP_NUM_THREADS=" ~/.bashrc; then
#     echo '# Optimal thread settings for MARL PPO Suite' >> ~/.bashrc
#     echo 'export OMP_NUM_THREADS=1' >> ~/.bashrc
#     echo 'export MKL_NUM_THREADS=1' >> ~/.bashrc
#     echo 'export PYTHONUNBUFFERED=1' >> ~/.bashrc
#     echo '' >> ~/.bashrc
# fi

# Add mpo function if not already in .bashrc
if ! grep -q "mpo()" ~/.bashrc; then
    cat << 'EOF' >> ~/.bashrc

# shortcut to enter the marl-ppo-suite venv
export MPPODIR="$HOME/marl-ppo-suite"
mpo() {
  cd "$MPPODIR" || return
  [[ -d venv ]] || python3 -m venv venv
  source venv/bin/activate
}
EOF
fi

# ── Part 3: Experiment setup (optional) ─────────────────────────────────────
print_step "Setting up experiment environment"

# Create logs directory
mkdir -p logs

# Copy experiment runner script if it doesn't exist
if [ ! -f "run_experiments.sh" ]; then
    print_info "Setting up experiment runner script..."
    cp cloud/standalone/run_experiments.sh .
    chmod +x run_experiments.sh
fi

# Copy tmux session script if it doesn't exist
if [ ! -f "tmux_session.sh" ]; then
    print_info "Setting up tmux session script..."
    cp cloud/standalone/tmux_session.sh .
    chmod +x tmux_session.sh
fi


# ── Final instructions ─────────────────────────────────────────────────────
print_header "Setup Complete!"

print_info "To activate the environment, either:"
echo "1. Run 'source $HOME/marl-ppo-suite/venv/bin/activate'"
echo "2. Or use the 'mpo' shortcut after restarting your shell or running 'source ~/.bashrc'"
echo ""

print_info "To run a single experiment:"
echo "mpo"
echo "python train.py --algo mappo --env_name smacv2 --map_name protoss_5_vs_5 --n_rollout_threads 16"
echo ""

print_info "To run multiple experiments in sequence:"
echo "mpo"
echo "./run_experiments.sh"
echo ""

print_info "To run experiments in a tmux session (recommended for long-running experiments):"
echo "./tmux_session.sh"
echo ""

print_info "For more information, see the Standalone deployment guide:"
echo "https://github.com/legalaspro/marl-ppo-suite/blob/main/cloud/standalone/README.md"
