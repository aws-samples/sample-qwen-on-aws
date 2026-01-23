# Dockerfile for Qwen-Image-2512 with vLLM-Omni on SageMaker
# Uses CUDA 11.8 for NCCL compatibility with SageMaker drivers

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and add deadsnakes PPA for Python 3.12
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
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12 and set as default
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# Set working directory
WORKDIR /opt/ml/code

# Install Python dependencies with CUDA 11.8 for NCCL compatibility
# CUDA 11.8 NCCL is compatible with CUDA 12.x drivers (forward compatible)
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118 && \
    python -m pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git && \
    python -m pip install --no-cache-dir "transformers>=4.51.3" accelerate pillow requests sentencepiece && \
    python -m pip install --no-cache-dir --ignore-installed flask gunicorn && \
    python -m pip install --no-cache-dir vllm==0.12.0 && \
    python -m pip install --no-cache-dir vllm-omni && \
    rm -rf /root/.cache/pip

# Copy inference server code
COPY serve.py /opt/ml/code/serve.py

# SageMaker expects the container to listen on port 8080
ENV PORT=8080
EXPOSE 8080

# Set environment variables for model
ENV MODEL_ID="Qwen/Qwen-Image-2512"
ENV HF_HOME="/opt/ml/model"
# Enable CUDA memory optimizations
ENV PYTORCH_ALLOC_CONF="expandable_segments:True"

# NCCL settings for SageMaker multi-GPU
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=1
ENV NCCL_SOCKET_IFNAME=eth0

# Health check for SageMaker (increased start-period for model loading)
HEALTHCHECK --interval=30s --timeout=30s --start-period=900s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

# Start the inference server
ENTRYPOINT ["python", "serve.py"]
