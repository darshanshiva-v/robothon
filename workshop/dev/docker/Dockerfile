# 1. Base Image: NVIDIA CUDA 12.1 on Ubuntu 22.04 (Jammy)
# Required for PyTorch CUDA support and ROS 2 Humble
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during apt-get installations
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 2. Core System & LeRobot AV Dependencies
RUN apt-get update && apt-get install -y \
    curl gnupg lsb-release wget build-essential git sudo \
    mesa-utils libgl1-mesa-glx \
    cmake python3-dev pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev \
    libsvtav1-dev nasm yasm \
    && rm -rf /var/lib/apt/lists/*

# 3. Install FFmpeg with SVT-AV1 support (pre-built static binary)
# Bypasses compilation while ensuring torchcodec compatibility
RUN wget https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz && \
    tar -xf ffmpeg-master-latest-linux64-gpl.tar.xz && \
    cp ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ && \
    cp ffmpeg-master-latest-linux64-gpl/bin/ffprobe /usr/local/bin/ && \
    rm -rf ffmpeg-master-latest-linux64-gpl*

# 4. Install ROS 2 Humble & MoveIt2
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && apt-get install -y \
    ros-humble-desktop ros-humble-ros-core ros-dev-tools python3-colcon-common-extensions \
    ros-humble-moveit ros-humble-pymoveit2 ros-humble-xacro ros-humble-joint-state-publisher-gui \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Gazebo Harmonic (Jetty)
RUN curl -sSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null && \
    apt-get update && apt-get install -y \
    gz-harmonic ros-humble-ros-gz \
    && rm -rf /var/lib/apt/lists/*

# 6. Install 'uv' for Ultra-Fast Python Management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    cp /root/.local/bin/uv /usr/local/bin/ && \
    cp /root/.local/bin/uvx /usr/local/bin/

# 7. Setup Python 3.12 Virtual Environment using uv
# This isolates the AI stack from the Ubuntu/ROS system Python 3.10
ENV VIRTUAL_ENV=/opt/lerobot_venv
ENV UV_CACHE_DIR=/opt/uv_cache
RUN uv python install 3.12 && \
    uv venv $VIRTUAL_ENV --python 3.12
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 8. Install PyTorch & Additional Simulation Tools inside the uv venv
RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    uv pip install mujoco opencv-python mediapipe transformers accelerate datasets einops

# 9. Clone and Install LeRobot with Hardware Support
RUN git clone https://github.com/huggingface/lerobot.git /opt/lerobot && \
    cd /opt/lerobot && \
    uv pip install -e ".[feetech]"

# 10. Create a Non-Root Hacker User
RUN useradd -m -s /bin/bash hacker && \
    echo "hacker ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    usermod -aG video hacker && \
    mkdir -p /opt/uv_cache && \
    chown -R hacker:hacker /opt/lerobot /opt/lerobot_venv /opt/uv_cache
USER hacker
WORKDIR /home/hacker/workspace

# 11. Configure bashrc to source ROS 2 and activate the uv environment automatically
RUN echo "source /opt/ros/humble/setup.bash" >> /home/hacker/.bashrc && \
    echo "source /opt/lerobot_venv/bin/activate" >> /home/hacker/.bashrc

CMD ["/bin/bash"]