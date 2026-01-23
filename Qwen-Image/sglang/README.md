# Qwen-Image-2512 on SageMaker with SGLang

Deploy Qwen-Image-2512 text-to-image model on Amazon SageMaker using SGLang for optimized inference.

## Features

- **Multi-GPU Support**: Tensor parallelism across multiple GPUs with `--num-gpus`
- **CFG Parallelism**: 1.2x-5.9x speedup with `--enable-cfg-parallel`
- **Native Model Support**: Qwen-Image-2512 in SGLang's model registry
- **Automatic Fallback**: Falls back to single-GPU if multi-GPU initialization fails
- **OpenAI-Compatible API**: Standard `/v1/images/generations` endpoint

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   SageMaker Endpoint                     │
├─────────────────────────────────────────────────────────┤
│  Port 8080: Flask Proxy                                 │
│  ├── /ping          → Health check                      │
│  └── /invocations   → Image generation                  │
├─────────────────────────────────────────────────────────┤
│  Port 8000: SGLang Server                               │
│  └── /v1/images/generations → Native diffusion API      │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Build and Push Docker Image

```bash
# Build, push to ECR, and cleanup in one command
./build-docker.sh
```

The build script automatically:
- Creates ECR repository if needed
- Authenticates with ECR
- Builds with memory limits to prevent crashes
- Pushes to ECR
- Cleans up local images and cache

### 2. Deploy to SageMaker

Open and run `deploy_qwen_image_sglang.ipynb` in SageMaker or locally.

## Configuration

### Build Script Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | us-west-2 | AWS region for ECR |
| `AWS_ACCOUNT_ID` | (auto-detect) | AWS account ID |
| `ECR_REPO_NAME` | qwen-image-sglang | ECR repository name |
| `IMAGE_TAG` | latest | Docker image tag |
| `MEMORY_LIMIT` | 8g | Docker build memory limit |
| `CLEANUP_ENABLED` | true | Clean up after build |

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

### Why GitHub Main Branch?

SGLang is installed from GitHub main instead of PyPI because:
- PyPI version (v0.5.7) has a registry bug for Qwen-Image-2512 (#16762)
- GitHub main includes the fix with Qwen-Image-2512 in native registry
- Enables full SGLang optimizations without `--backend diffusers` workaround

### JIT Kernel Patch

The Dockerfile patches SGLang's JIT kernel compilation to support A100 GPUs (SM80):

```python
# Before: Only compiles for host GPU architecture
DEFAULT_CUDA_CFLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]

# After: Explicitly includes SM80 and SM90
DEFAULT_CUDA_CFLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr",
                       "-gencode=arch=compute_80,code=sm_80",
                       "-gencode=arch=compute_90,code=sm_90"]
```

This prevents "no kernel image is available for execution on the device" errors.

### Parameter Mapping

SGLang uses `true_cfg_scale` instead of `guidance_scale` for Qwen-Image:

| API Parameter | Internal SGLang Parameter |
|---------------|---------------------------|
| `guidance_scale` | `true_cfg_scale` |
| `num_inference_steps` | `num_inference_steps` |
| `seed` | `seed` |

Note: Parameters must be passed directly in the request body, not in `extra_body`.

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Container with CUDA 12.4, Python 3.12, SGLang |
| `serve.py` | Flask proxy + SGLang server startup |
| `build-docker.sh` | Build, push, and cleanup script |
| `deploy_qwen_image_sglang.ipynb` | Deployment notebook |
| `.dockerignore` | Files to exclude from Docker context |

## Troubleshooting

### "No kernel image available" Error

This occurs when JIT kernels are compiled without SM80 support. The Dockerfile includes a patch to fix this. If you see this error, ensure you're using the latest Dockerfile.

### Multi-GPU NCCL Failures

If multi-GPU mode fails with NCCL errors, the server automatically falls back to single-GPU mode. Check CloudWatch logs for details.

### Slow First Request

The first request triggers JIT kernel compilation, which can take 30-60 seconds. Subsequent requests are much faster.

## References

- [SGLang Diffusion Blog](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)
- [SGLang Documentation](https://docs.sglang.io)
- [Qwen-Image GitHub](https://github.com/QwenLM/Qwen-Image)
- [Qwen-Image Model Card](https://huggingface.co/Qwen/Qwen-Image-2512)
