#!/bin/bash
# server_setup.sh - Comprehensive server setup script for MARL PPO Suite on standalone Linux machines
# This script handles both initial server setup (if run as root) and system dependencies

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

# ── Check if script is run as root ─────────────────────────────────────────────
IS_ROOT=false
if [ "$(id -u)" = "0" ]; then
    IS_ROOT=true
fi

# ── Main script ─────────────────────────────────────────────────────────────
print_header "MARL PPO Suite Server Setup for Standalone Deployment"

# ── Part 1: Initial server setup (root only) ─────────────────────────────────
if [ "$IS_ROOT" = true ]; then
    print_step "Running initial server setup as root"

    # Update system
    print_info "Updating system packages..."
    apt-get update
    apt-get upgrade -y

    # Install essential packages
    print_info "Installing essential packages..."
    apt-get install -y \
        build-essential \
        curl \
        wget \
        git \
        vim \
        htop \
        tmux \
        fail2ban \
        unzip \
        software-properties-common \
        ca-certificates \
        apt-transport-https

    # Set up a new user
    print_step "Setting up a new user"
    echo -e "${GREEN}Please enter a username for the new user:${NC}"
    read -r USERNAME

    # Check if user already exists
    if id "$USERNAME" &>/dev/null; then
        print_info "User $USERNAME already exists. Skipping user creation."
    else
        # Create user with home directory, disabled password, and add to sudo group
        adduser --disabled-password --gecos "" $USERNAME
        usermod -aG sudo $USERNAME
        # give user password‑less sudo
        echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/90-"$USERNAME"
        chmod 440 /etc/sudoers.d/90-"$USERNAME"

        # Set up SSH for the new user
        print_info "Setting up SSH for $USERNAME..."
        mkdir -p /home/$USERNAME/.ssh
        cp ~/.ssh/authorized_keys /home/$USERNAME/.ssh/ 2>/dev/null || \
            print_info "No SSH keys found for root. Please add SSH keys manually."
        # Set proper permissions in one command
        chown -R $USERNAME:$USERNAME /home/$USERNAME/.ssh
        chmod 700 /home/$USERNAME/.ssh
        [ -f /home/$USERNAME/.ssh/authorized_keys ] && chmod 600 /home/$USERNAME/.ssh/authorized_keys
    fi

    # Set timezone
    print_info "Setting timezone to UTC..."
    timedatectl set-timezone UTC

    # Create data directories
    print_info "Creating data directories..."
    mkdir -p /data/backups
    chown -R $USERNAME:$USERNAME /data

    # Set up swap space (if not already set up)
    print_step "Setting up swap space"
    SWAP_SIZE=4G  # ← change to 8G if you really need it

    if ! grep -q '^/swapfile' /proc/swaps; then
      fallocate -l "$SWAP_SIZE" /swapfile
      chmod 600 /swapfile
      mkswap  /swapfile
      swapon  /swapfile
      grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
    fi

    cat >/etc/sysctl.d/99-swap.conf <<'EOF'
vm.swappiness=10
vm.vfs_cache_pressure=50
EOF
    sysctl --system > /dev/null

    # Secure SSH
    print_step "Hardening SSH"
    mkdir -p /etc/ssh/sshd_config.d

    cat >/etc/ssh/sshd_config.d/99-hardening.conf <<'EOF'
PermitRootLogin no
PasswordAuthentication no
EOF

    # Restart SSH service
    systemctl restart ssh 2>/dev/null || systemctl restart sshd

    print_step "Initial server setup complete!"
    print_info "Please log out and reconnect as $USERNAME@$(hostname -I | awk '{print $1}')"
    print_info "Then run this script again as the non-root user to install dependencies"
    exit 0
fi

# ── Part 2: Install dependencies (non-root) ─────────────────────────────────
print_step "Installing system dependencies"

# Check if running as root
if [ "$IS_ROOT" = true ]; then
    print_error "This part of the script should not be run as root"
    exit 1
fi

# Update system
print_info "Updating system packages..."
sudo apt-get update

# Install Python 3.11
print_step "Installing Python 3.11 and development tools"

# Use deadsnakes PPA to ensure Python 3.11 is available
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    python3-setuptools \
    python3-wheel

# Create symlinks for python3 -> python3.11 and python -> python3.11
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
sudo ln -sf /usr/bin/python3 /usr/bin/python

# Install system libraries required for StarCraft II
print_step "Installing system libraries for StarCraft II"
sudo apt-get install -y \
    libgl1 \
    libosmesa6-dev \
    libglfw3 \
    libglew-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libsm6 \
    libxext6 \
    libxrender-dev

# Upgrade pip
print_info "Upgrading pip..."
python -m pip install --user --upgrade pip

# Create directories for StarCraft II
print_info "Creating directories for StarCraft II..."
mkdir -p $HOME/StarCraftII/Maps/

# Install StarCraft II and SMACv2 maps
print_step "Installing StarCraft II and SMACv2 maps"

# Install StarCraft II if not already installed
if [ ! -f "$HOME/StarCraftII/Versions/Base75689/SC2_x64" ]; then
    cd $HOME
    print_info "Downloading StarCraft II..."
    wget --no-check-certificate http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
    print_info "Extracting StarCraft II..."
    unzip -P iagreetotheeula SC2.4.10.zip
    rm SC2.4.10.zip
else
    print_info "StarCraft II already installed. Skipping..."
fi

# Create SMAC_Maps directory if it doesn't exist
mkdir -p $HOME/StarCraftII/Maps/SMAC_Maps/

# Download SMACv2 maps
if [ ! -f "$HOME/StarCraftII/Maps/SMAC_Maps/protoss_5_vs_5.SC2Map" ]; then
    print_info "Downloading SMACv2 maps..."
    cd $HOME
    wget -q https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip
    print_info "Extracting SMACv2 maps..."
    unzip -q SMAC_Maps.zip -d $HOME/StarCraftII/Maps/SMAC_Maps/
    rm SMAC_Maps.zip
    print_info "SMACv2 maps installed successfully!"
else
    print_info "SMACv2 maps already installed. Skipping..."
fi

# Set up environment variables
print_info "Setting up environment variables..."
if ! grep -q "SC2PATH" ~/.bashrc; then
    echo 'export SC2PATH=$HOME/StarCraftII' >> ~/.bashrc
fi

# Install monitoring tools and tmux
print_info "Installing monitoring tools and tmux..."
sudo apt-get install -y htop iotop tmux

print_step "Dependencies installation complete!"
print_info "Next step: Run the app_setup.sh script to set up the repository and environment"
print_info "Command: ./app_setup.sh"
