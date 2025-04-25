# SMAC v2 Environment Integration

This directory contains wrappers and utilities for integrating the StarCraft Multi-Agent Challenge v2 (SMAC v2) environment with our MAPPO implementation.

## Overview

SMAC v2 is an improved version of the original SMAC environment, offering more flexibility in team composition and unit capabilities. This integration allows you to use SMAC v2 with our MAPPO implementation, including support for vectorized environments.

## Installation

To use SMAC v2, you need to install it first:

```bash
pip install git+https://github.com/oxwhirl/smacv2.git

# Make sure to also install gymnasium
pip install gymnasium
```

## Directory Structure

- `smacv2_env.py`: Main wrapper for SMAC v2 environment
- `vec_env_compat_wrapper.py`: Compatibility wrapper for vectorized environments
- `__init__.py`: Module initialization

## Configuration

SMAC v2 maps are configured using YAML files located in `configs/envs_cfgs/smacv2_map_config/`. Each file defines the team composition and capabilities for a specific map.

Example configuration for the `3m` map:

```yaml
map_name: 3m
capability_config:
  n_units: 3
  n_enemies: 3
  team_gen:
    dist_type: "fixed_team"
    unit_types: ["marine", "marine", "marine"]
  enemy_gen:
    dist_type: "fixed_team"
    unit_types: ["marine", "marine", "marine"]
```

## Usage

### Basic Usage

```python
from envs import create_smacv2_env

# Create a SMAC v2 environment
args = {"map_name": "3m"}
env = create_smacv2_env(args)

# Reset the environment
obs, state, avail_actions = env.reset()

# Take a step in the environment
actions = [0] * env.n_agents  # No-op actions
obs, state, rewards, dones, infos, avail_actions = env.step(actions)
```

### Vectorized Environments

```python
from envs.env_vectorization import DummyVecEnv
from envs.smacv2_factory import create_smacv2_env

# Create a function that returns a SMAC v2 environment
def make_env():
    args = {"map_name": "3m"}
    return create_smacv2_env(args)

# Create 4 environments
env_fns = [make_env for _ in range(4)]
vec_env = DummyVecEnv(env_fns)

# Reset all environments
obs, state, avail_actions = vec_env.reset()

# Take a step in all environments
actions = [[0] * vec_env.n_agents for _ in range(4)]
obs, state, rewards, dones, infos, avail_actions = vec_env.step(actions)
```

## Example

See `examples/smacv2_vec_env_example.py` for a complete example of how to use vectorized SMAC v2 environments.

## Features

- Support for fixed and randomly generated teams
- Compatible with vectorized environments
- Proper handling of observations, states, rewards, and available actions
- Integration with our MAPPO implementation
