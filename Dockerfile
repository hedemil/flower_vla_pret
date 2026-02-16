FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools \
    mesa-vulkan-drivers \
    libegl1 \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd0 \
    libglvnd-dev \
    libxext6 \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    xvfb \
    x11-utils \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.10 python3.10-venv python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install pip for python3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Configure Vulkan ICD for NVIDIA
RUN mkdir -p /etc/vulkan/icd.d /usr/share/glvnd/egl_vendor.d \
    && echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.2.0"}}' \
    > /etc/vulkan/icd.d/nvidia_icd.json \
    && echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

ENV VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
ENV __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Create non-root user
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser \
    && useradd -m -u ${UID} -g ${GID} appuser

WORKDIR /app

# Copy project files
COPY --chown=appuser:appuser . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_simpler.txt

# Entrypoint for headless rendering
COPY --chown=appuser:appuser docker_entrypoint.sh /app/docker_entrypoint.sh
RUN chmod +x /app/docker_entrypoint.sh

RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

ENTRYPOINT ["/app/docker_entrypoint.sh"]
CMD ["python", "-m", "flower_vla.eval.simpler.bridge_eval_only"]
