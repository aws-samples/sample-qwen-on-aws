# Qwen-Image-2512 on SageMaker with vLLM-Omni

Deploy Qwen-Image-2512 text-to-image model on Amazon SageMaker using vLLM-Omni for inference.

## Features

- **Simple Setup**: Uses `vllm serve --omni` command
- **OpenAI-Compatible API**: Standard chat completions endpoint
- **Memory Optimizations**: VAE slicing and tiling for efficient VRAM usage
- **Stable Operation**: Single-GPU mode for reliability on SageMaker

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   SageMaker Endpoint                     │
├─────────────────────────────────────────────────────────┤
│  Port 8080: Flask Proxy                                 │
│  ├── /ping          → Health check                      │
│  └── /invocations   → Image generation                  │
├─────────────────────────────────────────────────────────┤
│  Port 8000: vLLM-Omni Server                            │
│  └── /v1/chat/completions → Chat API with image output  │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Build and Deploy

Open and run `deploy_qwen_image_vllm.ipynb` in SageMaker. The notebook includes:
- Docker build steps
- ECR push
- SageMaker deployment
- Test inference

### 2. Manual Docker Build (Optional)

```bash
# Build the image
docker build -t qwen-image-vllm-omni:latest .

# Push to ECR (after authentication)
docker tag qwen-image-vllm-omni:latest <account>.dkr.ecr.<region>.amazonaws.com/qwen-image-vllm-omni:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/qwen-image-vllm-omni:latest
```

## Configuration

### Runtime Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | Qwen/Qwen-Image-2512 | HuggingFace model ID |
| `HF_TOKEN` | (none) | HuggingFace token (if needed) |
| `PORT` | 8080 | SageMaker proxy port |

## API Usage

### Request

```json
{
    "prompt": "A beautiful sunset over mountains",
    "negative_prompt": "blurry, low quality",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 4.0,
    "seed": 42
}
```

### Response

```json
{
    "image": "<base64_encoded_png>",
    "seed": 42
}
```

## Technical Details

### CUDA Version

Uses CUDA 11.8 for NCCL compatibility with SageMaker drivers. CUDA 11.8 NCCL is forward-compatible with CUDA 12.x drivers.

### vLLM-Omni Command

The server starts with:
```bash
vllm serve <model> --omni --port 8000 --host 0.0.0.0 --vae-use-slicing --vae-use-tiling
```

### Parameter Passing

vLLM-Omni uses `extra_body` for generation parameters:
```python
{
    "model": "Qwen/Qwen-Image-2512",
    "messages": [{"role": "user", "content": "prompt"}],
    "extra_body": {
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,
        "seed": 42
    }
}
```

### Single-GPU Mode

Multi-GPU tensor parallelism is disabled due to NCCL socket interface issues on SageMaker. The server runs in single-GPU mode for stability.

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Container with CUDA 11.8, Python 3.12, vLLM-Omni |
| `serve.py` | Flask proxy + vLLM server startup |
| `deploy_qwen_image_vllm.ipynb` | Deployment notebook |

## Limitations

- **Single-GPU Only**: Multi-GPU mode has NCCL issues on SageMaker
- **Slower Than SGLang**: No CFG parallelism optimization
- **Chat API**: Uses chat completions endpoint, not native image generation

## Troubleshooting

### NCCL Socket Interface Error

If you see "NCCL no socket interface found", this is expected on SageMaker. The server runs in single-GPU mode.

### Model Loading Timeout

The first startup takes 10-15 minutes to download the 20B model. Ensure `container_startup_health_check_timeout` is set to at least 1200 seconds.

### Memory Issues

If you encounter OOM errors, ensure VAE slicing/tiling is enabled (default in serve.py).

## References

- [vLLM-Omni Documentation](https://docs.vllm.ai/projects/vllm-omni)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [Qwen-Image GitHub](https://github.com/QwenLM/Qwen-Image)
- [Qwen-Image Model Card](https://huggingface.co/Qwen/Qwen-Image-2512)
