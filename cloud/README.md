# Cloud Deployment for MARL PPO Suite

This guide provides comprehensive instructions for deploying MARL PPO Suite on various cloud platforms.

## Directory Structure

- `train.sh`: Main script for running MARL PPO Suite on cloud instances
- `train_simple.sh`: Simplified script for running MARL PPO Suite on cloud instances
- `build_and_push.sh`: CI/CD-friendly script for building and pushing Docker images
- `docker-compose.yml`: Docker Compose configuration for local testing
- `examples/`: Example scripts and configurations
  - `docker_examples.sh`: Examples of using the build_and_push.sh script
- `runpods/`: RunPods.io specific deployment files
  - `Dockerfile`: Docker configuration for RunPods.io
  - `entrypoint.sh`: Container entrypoint script
- `standalone/`: Standalone deployment files for any Linux machine
  - `server_setup.sh`: Initial server setup and system dependencies
  - `app_setup.sh`: Repository and environment setup
  - `run_experiments.sh`: Script for running multiple experiments
  - `tmux_session.sh`: Script for managing experiments in tmux

## Quick Start

### Building and Pushing Docker Images

```bash
# Build and test
./cloud/build_and_push.sh --test

# Build with version tag and push to Docker Hub
./cloud/build_and_push.sh --tag v1.0.0 --test --push
```

### Using Different Dockerfiles

To use a different Dockerfile:

```bash
# Example: Using a custom Dockerfile
./cloud/build_and_push.sh --tag latest --file cloud/runpods/Dockerfile.custom --test
```

### macOS Compatibility

The scripts are designed to work on both Linux and macOS:

- On macOS, Docker Desktop handles GPU passthrough differently, so the `--gpus all` flag is automatically omitted
- If you're using Docker Desktop with a tool like OrbStack or Colima for better performance, the scripts should work without modification
- The docker-compose.yml file has the GPU configuration commented out by default for macOS compatibility

## Available Cloud Platforms

Currently supported cloud platforms:

- **RunPods.io**: Cost-effective GPU cloud platform

  - Uses the official PyTorch image for optimal compatibility
  - Includes PyTorch 2.5.1 with CUDA 12.4 support
  - Containerized deployment with Docker

- **Standalone Deployment**: For any Linux server or local machine
  - Direct code deployment (non-Docker)
  - Manual environment setup
  - Good for long-running experiments
  - Works on Hetzner, AWS, GCP, or any Linux machine

## Deployment Approaches

### 1. Containerized Deployment (RunPods.io)

- Uses Docker containers
- Pre-configured environment
- Easy to deploy and scale
- Good for experimentation and short-term runs

### 2. Direct Code Deployment (Standalone)

- Clone repository directly on the server or local machine
- Set up environment manually
- More control over the environment
- Better for long-running experiments
- Lower overhead
- Works on any Linux machine (Hetzner, AWS, GCP, local, etc.)

## RunPods.io Deployment Guide

### Prerequisites

- Docker installed on your local machine
- A RunPods.io account
- A Docker Hub account (for storing your Docker image)

### Docker Image Details

The Docker image is based on the official PyTorch image (`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`) and includes:

- PyTorch 2.5.1 with CUDA 12.4 support
- Python 3.10
- CUDA 12.4 with cuDNN 9
- StarCraft II and SMAC maps
- All required dependencies for MARL PPO Suite training

### Detailed Deployment Steps

#### 1. Test Locally

Before deploying to RunPods.io, test your Docker setup locally:

```bash
# Build and test in one step
./cloud/build_and_push.sh --test

# Check the logs
docker logs -f marl-ppo-suite-test
```

#### 2. Build and Push to Docker Hub

The `build_and_push.sh` script provides a CI/CD-friendly way to build, test, and push Docker images:

```bash
# Build and test locally
./cloud/build_and_push.sh --test

# Build with a specific version tag and push to Docker Hub
./cloud/build_and_push.sh --tag v1.0.0 --test --push

# For more options
./cloud/build_and_push.sh --help
```

The script supports:

- Custom image names and tags
- Automatic testing
- Multiple registries (Docker Hub, GitHub Container Registry, etc.)
- Proper error handling and logging

For more examples, see `cloud/examples/docker_examples.sh`.

#### 3. Set Up RunPods.io

1. Create a RunPods.io account at https://www.runpods.io/
2. Add a payment method
3. Create a volume for persistent storage:
   - Go to https://www.runpods.io/console/user/volumes
   - Click "Add Volume"
   - Name it "marl-ppo-suite-data"
   - Choose a size (at least 20GB)
   - Select a location close to you

#### 4. Deploy on RunPods.io

##### Option 1: Web Interface

1. Go to https://www.runpods.io/console/user/pods
2. Click "Deploy"
3. Select a GPU type (A4000 or better recommended)
4. Select a container type: "Docker Hub"
5. Container image: `yourusername/marl-ppo-suite:latest`
6. Volume: Select your "marl-ppo-suite-data" volume
7. Start command: `--map_name 3m --algo mappo --n_rollout_threads 8`
8. Deploy!

##### Option 2: RunPods CLI (Recommended)

1. Install the RunPods CLI:

   ```bash
   pip install runpodctl
   ```

2. Log in:

   ```bash
   runpodctl login
   ```

3. Deploy:
   ```bash
   runpodctl create pod \
     --name marl-ppo-suite \
     --image yourusername/marl-ppo-suite:latest \
     --gpu A4000 \
     --volume marl-ppo-suite-data:/app/data \
     --command "--map_name 3m --algo mappo --n_rollout_threads 8"
   ```

### Monitoring Training

1. Go to https://www.runpods.io/console/user/pods
2. Click on your pod to see details
3. Use the web terminal or SSH to connect
4. Check logs:
   ```bash
   cd /app
   tail -f runs/training.log
   ```

### Retrieving Results

Your trained models and logs will be saved to the persistent volume in the `/app/runs` directory. You can:

1. Download them via SSH/SFTP
2. Use the RunPods.io web interface to download files
3. Mount the volume to another pod for further processing

### Advanced Configuration

#### Using Weights & Biases

To use Weights & Biases for experiment tracking:

1. Add your W&B API key to the deployment:

   ```bash
   runpodctl create pod \
     --name marl-ppo-suite \
     --image yourusername/marl-ppo-suite:latest \
     --gpu A4000 \
     --volume marl-ppo-suite-data:/app/data \
     --env WANDB_API_KEY=your_api_key \
     --command "--map_name 3m --algo mappo --use_wandb"
   ```

2. Log in to W&B in your container:
   ```bash
   wandb login your_api_key
   ```

#### Running Multiple Experiments

To run multiple experiments in parallel:

1. Deploy multiple pods with different configurations
2. Use different map names or algorithms for each pod
3. Use W&B to compare results

### Troubleshooting

#### Training Performance Optimization

Since StarCraft II is primarily CPU-intensive, optimize your setup with:

1. Set environment variables for optimal thread performance:

   ```bash
   export OMP_NUM_THREADS=1
   export MKL_NUM_THREADS=1
   ```

2. Adjust `n_rollout_threads` based on your CPU core count:

   - For 8-core systems: 8-12 threads
   - For 16-core systems: 16-24 threads

3. Monitor CPU utilization with `htop` to ensure efficient resource usage

#### Container Issues

Check the logs for errors:

```bash
runpodctl logs pod_id
```

Common issues:

- StarCraft II installation issues: Check SC2PATH environment variable
- Missing dependencies: Update the Dockerfile

#### Out of Disk Space

If you run out of disk space:

1. Increase the volume size in RunPods.io
2. Clean up old logs and checkpoints
3. Implement more efficient checkpointing (save only best models)

## Common Parameters

All cloud deployment scripts support the following parameters:

- `--map_name`: SMAC map name (default: "3m")
- `--algo`: Algorithm to use (default: "mappo")
- `--use_rnn`: Use RNN networks
- `--max_steps`: Maximum number of steps (default: 1000000)
- `--n_rollout_threads`: Number of parallel environments (default: 8)
- `--n_steps`: Number of steps per rollout (default: 400)
- `--use_wandb`: Use Weights & Biases for logging
- `--no_eval`: Disable evaluation during training
- `--seed`: Random seed (default: 1)

## Example Commands

### Basic Training

```bash
--map_name 3m --algo mappo
```

### RNN Training

```bash
--map_name 3m --algo mappo --use_rnn
```

### Training with Weights & Biases

```bash
--map_name 3m --algo mappo --use_wandb
```

### Training on Larger Maps

```bash
--map_name 5m_vs_6m --algo mappo --n_steps 800
```

### Standalone Deployment

For standalone deployment on any Linux machine, we provide:

1. Setup scripts for installing dependencies directly on the server or local machine
2. Environment configuration scripts
3. Scripts for running and managing long-running experiments
4. Monitoring and resource management tools

See the [Standalone Deployment Guide](standalone/README.md) for detailed instructions.

This guide works for Hetzner Cloud, AWS, GCP, or any Linux machine with sufficient CPU resources.

## Resources

- [RunPods.io Documentation](https://docs.runpods.io/)
- [Docker Documentation](https://docs.docker.com/)
- [SMAC GitHub Repository](https://github.com/oxwhirl/smac)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
