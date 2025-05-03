# Use stable image with CUDA 12.4
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
# 2.4.1-cuda12.4-cudnn9-runtime
# 2.4.0-cuda12.4-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git \
    vim \
    htop \
    tmux \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglfw3 \
    libglew-dev \
    && rm -rf /var/lib/apt/lists/*

# Install StarCraft II
WORKDIR /root
RUN mkdir -p /root/StarCraftII/Maps/SMAC_Maps && \
    wget --no-check-certificate http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip \
    && unzip -P iagreetotheeula SC2.4.10.zip \
    && rm SC2.4.10.zip

# Download and install SMAC maps
RUN wget https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip \
    && unzip SMAC_Maps.zip -d /root/StarCraftII/Maps/SMAC_Maps \
    && rm SMAC_Maps.zip

# Set SC2PATH environment variable
ENV SC2PATH=/root/StarCraftII

# Switch to app directory for Python installations
WORKDIR /app    

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/oxwhirl/smac.git
RUN pip install --no-cache-dir git+https://github.com/oxwhirl/smacv2.git

# Copy project files
COPY . .

# Make scripts executable
RUN chmod +x cloud/train_simple.sh
RUN chmod +x cloud/runpods/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["./cloud/runpods/entrypoint.sh"]

# Default command (can be overridden)
CMD ["--map_name", "3m", "--algo", "mappo", "--n_rollout_threads", "8"]
