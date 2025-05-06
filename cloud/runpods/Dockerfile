# Use multi-stage build to reduce image size

# Stage 1: Download and extract StarCraft II
FROM ubuntu:22.04 AS sc2

RUN apt-get update && apt-get install -y wget unzip && rm -rf /var/lib/apt/lists/*

# Download StarCraft II
WORKDIR /root
RUN mkdir -p /root/StarCraftII/Maps/SMAC_Maps && \
    wget --no-check-certificate http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip \
    && unzip -P iagreetotheeula SC2.4.10.zip \
    && rm SC2.4.10.zip

# Download SMAC maps
RUN wget https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip \
    && unzip SMAC_Maps.zip -d /root/StarCraftII/Maps/SMAC_Maps \
    && rm SMAC_Maps.zip

# Stage 2: Final image with PyTorch and GPU support (CUDA 12.4)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# ── OS deps ───────────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget unzip git vim htop tmux \
    libgl1-mesa-glx libosmesa6-dev libglfw3 libglew-dev \
    openssh-server ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── OpenSSH minimal hardening ────────────────────────────────────────────────
# 1. create run dir, 2. disable passwords, 3. silence pam_loginuid warning
RUN mkdir /var/run/sshd && \
    sed -ri 's/^#?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    echo "session optional pam_loginuid.so" >> /etc/pam.d/sshd

# ── Copy StarCraft II download from builder stage ────────────────────────────
COPY --from=sc2 /root/StarCraftII /root/StarCraftII
ENV SC2PATH=/root/StarCraftII

# ── Python deps & code ────────────────────────────────────────────────────────
# Switch to app directory for Python installations
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir git+https://github.com/oxwhirl/smac.git \
 && pip install --no-cache-dir git+https://github.com/oxwhirl/smacv2.git

 # Copy project files
COPY . .

# ── Permissions & entrypoint ─────────────────────────────────────────────────
RUN chmod +x cloud/train_simple.sh cloud/runpods/entrypoint.sh
EXPOSE 22

# Set entrypoint
ENTRYPOINT ["./cloud/runpods/entrypoint.sh"]

# Default command (can be overridden)
CMD ["--map_name", "3m", "--algo", "mappo", "--n_rollout_threads", "8"]
