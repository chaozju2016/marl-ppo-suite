# Standalone Deployment Guide for MARL PPO Suite

This guide provides step-by-step instructions for setting up and running MARL PPO Suite on any standalone machine without Docker. While it's written with Hetzner Cloud servers in mind, the instructions apply to any Linux server or local machine.

Since StarCraft II is primarily CPU-intensive, we recommend using CPU-optimized instances like Hetzner's CPX51 or CX52 for the best performance-to-cost ratio.

## Recommended System Specifications

For optimal performance with StarCraft II multi-agent reinforcement learning, we recommend:

### High-Performance Setup

- 16+ CPU cores (AMD EPYC or Intel Xeon)
- 32+ GB RAM
- SSD storage (NVMe preferred)
- Example: Hetzner CPX51 (16 dedicated vCPUs, 32GB RAM, ~€44.90/month)

### Budget Setup

- 8+ CPU cores
- 16-32 GB RAM
- SSD storage
- Example: Hetzner CX52 (16 shared vCPUs, 32GB RAM, ~€29.90/month)

The key factor is having multiple CPU cores for parallel environment processing. Since StarCraft II is primarily CPU-bound, investing in more cores will yield better performance than a GPU for most SMAC scenarios.

## Initial Setup

### 1. Prepare Your Server or Machine

These instructions work for any Linux server or local machine. If you're using Hetzner Cloud:

1. Sign up at [Hetzner Cloud Console](https://console.hetzner.cloud/)
2. Create a new project
3. Set up a Cloud Firewall (recommended):
   - Create a rule to allow SSH (port 22) from your IP only
   - Block all other incoming traffic
   - Allow all outgoing traffic
4. Add a new server:
   - Choose Ubuntu 24.04 or Ubuntu 22.04 as the operating system
   - Select a CPU-optimized instance (CPX51 recommended)
   - Add your SSH key
   - Apply the firewall
   - Consider enabling backups
   - Create the server

For other cloud providers or local machines, ensure you have:

- Ubuntu 22.04 or 24.04 (other Linux distributions may work but are not tested)
- SSH access with sudo privileges
- At least 8 CPU cores and 16GB RAM for reasonable performance

### 2. Simplified Setup Process

We've simplified the setup process into just two scripts:

1. **server_setup.sh**: Handles initial server setup and system dependencies
2. **app_setup.sh**: Sets up the repository, environment, and experiment configuration

#### Step 1: Server Setup

```bash
# Download the server setup script
wget -O server_setup.sh https://raw.githubusercontent.com/legalaspro/marl-ppo-suite/main/cloud/standalone/server_setup.sh

# Make it executable
chmod +x server_setup.sh

# Run as root for initial server setup
./server_setup.sh

# After the script completes, log out and reconnect as the new user
# Then run the script again as the non-root user to install dependencies
./server_setup.sh
```

This script will:

- When run as root:

  - Update the system
  - Create a non-root user with sudo privileges
  - Install essential packages
  - Configure SSH for security
  - Set up swap space for better performance

- When run as non-root:
  - Install Python 3.11 and development tools
  - Install required system libraries
  - Download and install StarCraft II and SMACv2 maps

#### Step 2: Application Setup

After completing the server setup, run the application setup script:

```bash
# Download the application setup script
wget -O app_setup.sh https://raw.githubusercontent.com/legalaspro/marl-ppo-suite/main/cloud/standalone/app_setup.sh

# Make it executable
chmod +x app_setup.sh

# Run the script
./app_setup.sh
```

This script will:

- Clone the repository
- Create a Python virtual environment
- Install all required dependencies
- Set up environment variables (including optimal thread settings)
- Add a convenient `mpo` shortcut to your .bashrc
- Configure experiment settings

After running the script, you can use the `mpo` shortcut to activate the environment:

```bash
# Reload your shell configuration
source ~/.bashrc

# Activate the environment and change to the project directory
mpo
```

## Running Experiments

### 1. Single Experiment

To run a single experiment:

```bash
# Activate the environment using the shortcut
mpo

# Run the experiment
python train.py --algo mappo --env_name smacv2 --map_name protoss_5_vs_5 --n_rollout_threads 16
```

### 2. Setting Up Weights & Biases (wandb)

If you want to use wandb for experiment tracking (recommended):

#### Option 1: Direct Export (Recommended)

Add these environment variables to your `.bashrc` file:

```bash
# Add to .bashrc
echo 'export WANDB_API_KEY=your_api_key_here' >> ~/.bashrc
echo 'export WANDB_PROJECT=marl-ppo-suite' >> ~/.bashrc
echo 'export WANDB_ENTITY=your_username_or_team' >> ~/.bashrc
source ~/.bashrc
```

Or export them directly in your terminal session before running experiments:

```bash
export WANDB_API_KEY=your_api_key_here
export WANDB_PROJECT=marl-ppo-suite
export WANDB_ENTITY=your_username_or_team
```

Then add `--use_wandb` to your experiment arguments to enable wandb logging.

### 3. Running Experiments

The app_setup.sh script automatically sets up everything you need to run experiments, including:

- The run_experiments.sh script for running multiple experiments sequentially
- The tmux_session.sh script for running experiments in a tmux session
- Optimal thread settings (OMP_NUM_THREADS=1, MKL_NUM_THREADS=1) in your .bashrc

#### Running Multiple Sequential Experiments

The run_experiments.sh script is already set up in your marl-ppo-suite directory:

```bash
# Activate the environment
mpo

# Edit the experiment configurations if needed
nano run_experiments.sh

# Run the experiments
./run_experiments.sh
```

The script uses a simple format to define experiments:

```bash
EXPERIMENTS=(
    "--algo mappo \
     --env_name smacv2 \
     --map_name 3m \
     --n_rollout_threads 8 \
     --max_steps 1000000 \
     --state_type AS \
     --n_steps 400 \
     --ppo_epoch 5 \
     --clip_param 0.05 \
     --use_rnn \
     --use_wandb|3m_mappo"

    "--algo happo \
     --env_name smacv2 \
     --map_name 5m_vs_6m \
     --n_rollout_threads 8 \
     --max_steps 2000000 \
     --state_type AS \
     --n_steps 400 \
     --ppo_epoch 5 \
     --clip_param 0.05 \
     --use_rnn \
     --use_wandb|5m_vs_6m_happo"
)
```

Each line contains:

- The full command-line arguments for the experiment
- A pipe character (`|`)
- A short name for the experiment (used in log filenames)

#### Running in Background with tmux

**Why use tmux?** tmux allows your experiments to keep running even after you disconnect from SSH. This is essential for long-running experiments that might take days to complete.

The tmux_session.sh script is already set up in your marl-ppo-suite directory:

```bash
# Start the tmux session
./tmux_session.sh
```

This script creates a tmux session with multiple windows:

- **experiments**: Main window for running your experiments
- **monitoring**: System monitoring with htop
- **gpu**: GPU monitoring (if available)
- **logs**: Displays the last 50 lines of the most recent log file (updates every 2 seconds)
- **disk**: Disk usage monitoring

**Basic tmux commands:**

- Switch windows: `Ctrl+B`, then window number (0-4)
- Detach from session: `Ctrl+B`, then `D`
- Reconnect later: `tmux attach -t marl_experiments`

## Monitoring and Management

### 1. System Monitoring

```bash
# Install monitoring tools
sudo apt-get install -y htop iotop

# Monitor CPU and memory usage
htop

# Monitor disk I/O
iotop
```

### 2. Data Management

```bash
# Compress experiment results
tar -czvf experiments_results.tar.gz ~/marl-ppo-suite/runs/

# Download results to local machine (run on your local machine)
scp yourusername@your_server_ip:~/experiments_results.tar.gz .
```

## Maintenance

### 1. Updating the Code

```bash
cd ~/marl-ppo-suite
git pull
python -m pip install --user -r requirements.txt
```

### 2. Manual Backups

```bash
# Create a backup of experiment results
mkdir -p ~/backups
tar -czvf ~/backups/marl-backup-$(date +%Y%m%d).tar.gz ~/marl-ppo-suite/runs/
```

### 3. Server Maintenance

```bash
# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Check disk usage
df -h

# Simple cleanup of old experiment data (older than 30 days)
find ~/marl-ppo-suite/runs/ -type d -mtime +30 -delete
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**

   - Reduce `n_rollout_threads` in your experiment configuration
   - Check for memory leaks with `top` or `htop`
   - Add swap space if necessary

2. **Slow Performance**

   - Optimize `n_rollout_threads` based on CPU count
   - Check for CPU throttling with `lscpu`
   - Monitor disk I/O with `iotop`

3. **StarCraft II Issues**
   - Verify SC2PATH is set correctly
   - Check if StarCraft II maps are installed properly
   - Reinstall StarCraft II if necessary

### Getting Help

If you encounter issues not covered in this guide, please:

1. Check the [GitHub repository issues](https://github.com/legalaspro/marl-ppo-suite/issues)
2. Create a new issue with detailed information about your problem
