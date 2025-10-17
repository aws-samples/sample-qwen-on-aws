# vLLM Container for Qwen3-VL Vision-Language Models on SageMaker
# Supports ALL Qwen3-VL model variants:
#   - Qwen3-VL-4B-Instruct (4B parameters, lightweight)
#   - Qwen3-VL-8B-Instruct (9B parameters, balanced)
#   - Qwen3-VL-30B-A3B-Instruct (31B parameters, MoE, high performance)
#   - Qwen3-VL-235B-A22B-Instruct (236B parameters, MoE, maximum capability)

FROM vllm/vllm-openai:v0.10.2

# Copy the serve script that handles environment variable to CLI argument conversion
COPY ./serve /usr/bin/serve

# Make the serve script executable
RUN chmod 755 /usr/bin/serve

# Create python symlink to python3 (fix for python not found)
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install additional dependencies for vision processing
# Note: Qwen3-VL requires latest transformers from GitHub for qwen3_vl_moe architecture support
RUN pip install --no-cache-dir \
    Pillow>=10.0.0 \
    opencv-python-headless>=4.8.0 \
    qwen-vl-utils \
    git+https://github.com/huggingface/transformers.git

# Upgrade vLLM to ensure compatibility with Qwen3-VL models
RUN pip install --no-cache-dir --upgrade vllm

# Verify the serve script was copied correctly
RUN ls -la /usr/bin/serve && head -5 /usr/bin/serve

# Set the entrypoint to our serve script
ENTRYPOINT ["/usr/bin/serve"]
