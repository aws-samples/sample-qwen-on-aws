# Dockerfile for Qwen-Image-2512 with SGLang Diffusion on SageMaker
# Patches SGLang JIT kernel compilation to support SM80 (A100) GPUs

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    wget \
    curl \
    libnuma1 \
    libnuma-dev \
    cmake \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12 and set as default
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /opt/ml/code

# Install PyTorch first (cu124 for CUDA 12.4)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# Remove system blinker package that conflicts with Flask
RUN apt-get remove -y python3-blinker || true

# Install base dependencies first
RUN pip install --no-cache-dir "transformers>=4.51.3" accelerate pillow requests sentencepiece && \
    pip install --no-cache-dir aiohttp fastapi uvicorn httpx orjson pydantic && \
    pip install --no-cache-dir numpy packaging psutil && \
    pip install --no-cache-dir triton && \
    pip install --no-cache-dir flask gunicorn && \
    rm -rf /root/.cache/pip

# Install additional dependencies that sglang[all] misses for multimodal/diffusion
# These are required at runtime but not properly specified in sglang's dependencies
RUN pip install --no-cache-dir \
    remote-pdb \
    imageio \
    imageio-ffmpeg \
    opencv-python-headless \
    av \
    decord \
    einops \
    timm \
    scipy \
    ftfy \
    regex \
    safetensors \
    && rm -rf /root/.cache/pip

# Install SGLang from GitHub main branch (has Qwen-Image-2512 fix)
# Python package is in python/ subdirectory
RUN pip install --no-cache-dir "git+https://github.com/sgl-project/sglang.git#subdirectory=python&egg=sglang[all]" && \
    pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git && \
    rm -rf /root/.cache/pip

# Install sgl-kernel from pre-built wheel
RUN pip install --no-cache-dir sgl-kernel && \
    rm -rf /root/.cache/pip

# CRITICAL FIX: Patch SGLang's JIT kernel compilation to include SM80 (A100) architecture
# The default JIT compilation doesn't specify GPU architecture, causing "no kernel image" errors
# This adds -gencode flags for SM80 to ensure JIT kernels work on A100 GPUs
RUN UTILS_FILE=$(python -c "import sglang.jit_kernel.utils as u; print(u.__file__)") && \
    echo "Patching SGLang JIT kernel utils at: $UTILS_FILE" && \
    sed -i 's/DEFAULT_CUDA_CFLAGS = \["-std=c++20", "-O3", "--expt-relaxed-constexpr"\]/DEFAULT_CUDA_CFLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr", "-gencode=arch=compute_80,code=sm_80", "-gencode=arch=compute_90,code=sm_90"]/' "$UTILS_FILE" && \
    echo "Verifying patch applied:" && \
    grep "DEFAULT_CUDA_CFLAGS" "$UTILS_FILE"

# Copy inference server code
COPY serve.py /opt/ml/code/serve.py

# SageMaker expects the container to listen on port 8080
ENV PORT=8080
EXPOSE 8080

# Set environment variables for model
# Uses --backend diffusers to load Qwen-Image-2512
ENV MODEL_ID="Qwen/Qwen-Image-2512"
ENV HF_HOME="/opt/ml/model"

# Enable CUDA memory optimizations - critical for large models
# Note: PYTORCH_CUDA_ALLOC_CONF is deprecated, using PYTORCH_ALLOC_CONF
ENV PYTORCH_ALLOC_CONF="backend:cudaMallocAsync,max_split_size_mb:512,expandable_segments:True"
# Reduce CPU memory pressure during model loading
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8

# NCCL settings for SageMaker multi-GPU with timeout protection
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=1
ENV NCCL_P2P_DISABLE=0
ENV NCCL_ASYNC_ERROR_HANDLING=1
ENV NCCL_TIMEOUT=7200

# Create entrypoint script for dynamic network interface detection
# This auto-detects the network interface name in SageMaker (may not always be eth0)
RUN echo '#!/bin/bash\n\
set -e\n\
# Detect the default network interface dynamically\n\
DETECTED_IFACE=$(ip -o -4 route show to default | awk "{print \$5}" | head -n1)\n\
if [ -n "$DETECTED_IFACE" ]; then\n\
  export NCCL_SOCKET_IFNAME="$DETECTED_IFACE"\n\
  echo "Detected network interface for NCCL: $NCCL_SOCKET_IFNAME"\n\
else\n\
  echo "Warning: Could not detect network interface, NCCL will use defaults"\n\
fi\n\
# Execute the main command\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Health check for SageMaker (30 min startup for large model loading on multi-GPU)
HEALTHCHECK --interval=30s --timeout=30s --start-period=1800s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

# Start the inference server with wrapper script for network interface detection
ENTRYPOINT ["/entrypoint.sh", "python", "serve.py"]
