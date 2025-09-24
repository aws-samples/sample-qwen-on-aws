# Simplified vLLM Container for Qwen3-Next Deployment on SageMaker
# Uses native vLLM OpenAI-compatible server with environment variable configuration

FROM vllm/vllm-openai:v0.10.2

# Copy the serve script that handles environment variable to CLI argument conversion
COPY ./serve /usr/bin/serve

# Make the serve script executable
RUN chmod 755 /usr/bin/serve

# Create python symlink to python3 (fix for python not found)
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Verify the serve script was copied correctly
RUN ls -la /usr/bin/serve && head -5 /usr/bin/serve

# Set the entrypoint to our serve script
ENTRYPOINT ["/usr/bin/serve"]