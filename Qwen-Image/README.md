# Deploy Qwen-Image-2512 on Amazon SageMaker

This folder provides step-by-step examples for deploying the **Qwen-Image-2512** text-to-image model on Amazon SageMaker using the **Bring Your Own Container (BYOC)** approach.

## Model Overview

| Property | Value |
|----------|-------|
| **Model** | Qwen-Image-2512 |
| **Type** | Text-to-Image (20B MMDiT) |
| **Resolution** | Up to 2512×2512 pixels |
| **Source** | [QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image) |

## Deployment Options

Two inference frameworks are provided:

| Framework | Directory | Status | Best For |
|-----------|-----------|--------|----------|
| **SGLang** | `sglang/` | Recommended | Multi-GPU parallelism, production |
| **vLLM-Omni** | `vllm-omni/` | Stable | Single-GPU, simpler setup |

## Framework Comparison

### SGLang (Recommended)

**Advantages:**
- Native Qwen-Image-2512 support in model registry
- Multi-GPU tensor parallelism (`--num-gpus`)
- CFG parallelism (`--enable-cfg-parallel`) - 1.2x to 5.9x speedup
- OpenAI-compatible `/v1/images/generations` API
- Automatic fallback to single-GPU if multi-GPU fails

**Limitations:**
- Requires installation from GitHub main branch (PyPI version has registry bug)
- Parameters must be passed directly in request body (not in `extra_body`)
- Uses `true_cfg_scale` instead of standard `guidance_scale`

**Key Issues Discovered:**
- SGLang ignores `extra_body` for image generation - parameters must be at request root level
- Qwen-Image uses `true_cfg_scale` (for CFG vs guidance distillation models), not `guidance_scale`
- Default seed is 1024 if not specified, causing identical outputs

### vLLM-Omni

**Advantages:**
- Simpler setup with `vllm serve --omni` command
- Standard OpenAI chat completions API (`/v1/chat/completions`)
- `extra_body` parameter passing works correctly
- More mature codebase

**Limitations:**
- Single-GPU mode only (multi-GPU NCCL issues on SageMaker)
- Slower inference compared to SGLang with CFG parallelism
- Uses chat completions endpoint (not native image generation)

**Key Issues Discovered:**
- Multi-GPU mode causes NCCL socket interface errors on SageMaker
- VAE slicing/tiling required for memory optimization

## Instance Configuration

| Instance | GPUs | VRAM | Use Case | Cost (us-west-2) |
|----------|------|------|----------|------------------|
| ml.g5.12xlarge | 4x A10G | 96GB | Cost-effective production | ~$5.67/hr |
| ml.p4de.24xlarge | 8x A100 | 640GB | Maximum performance | ~$40.97/hr |

## Quick Start

### Option 1: SGLang (Recommended)

```bash
cd sglang/

# Build and push Docker image to ECR
./build-docker.sh

# Then open and run the deployment notebook
# deploy_qwen_image_sglang.ipynb
```

### Option 2: vLLM-Omni

```bash
cd vllm-omni/

# Open and run the deployment notebook (includes build steps)
# deploy_qwen_image_vllm.ipynb
```

## Project Structure

```
Qwen-Image/
├── README.md                           # This file
├── sglang/
│   ├── Dockerfile                      # Container with SGLang
│   ├── serve.py                        # Flask proxy + SGLang server
│   ├── build-docker.sh                 # Build, push to ECR, cleanup
│   ├── deploy_qwen_image_sglang.ipynb  # Deployment notebook
│   └── README.md                       # SGLang-specific documentation
└── vllm-omni/
    ├── Dockerfile                      # Container with vLLM-Omni
    ├── serve.py                        # Flask proxy + vLLM server
    ├── deploy_qwen_image_vllm.ipynb    # Deployment notebook
    └── README.md                       # vLLM-specific documentation
```

## API Reference

### Health Check Endpoint

```
GET /ping
Response: 200 OK (healthy) or 503 Service Unavailable
```

### Inference Endpoint

```
POST /invocations
Content-Type: application/json
```

**Request Body:**
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

**Response:**
```json
{
    "image": "<base64_encoded_png>",
    "seed": 42
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of desired image |
| `negative_prompt` | string | "" | What to avoid in generation |
| `width` | int | 1024 | Image width in pixels |
| `height` | int | 1024 | Image height in pixels |
| `num_inference_steps` | int | 50 | Denoising steps (Qwen-Image recommended: 50) |
| `guidance_scale` | float | 4.0 | Prompt adherence (Qwen-Image recommended: 4.0) |
| `seed` | int | random | Random seed for reproducibility |

**Note:** For Qwen-Image-2512, `guidance_scale` maps to `true_cfg_scale` internally. The recommended values from the model card are `num_inference_steps=50` and `guidance_scale=4.0`.

## Prerequisites

- AWS account with SageMaker access
- IAM role with permissions for:
  - Amazon ECR (push/pull images)
  - Amazon SageMaker (create models, endpoints)
- Docker installed locally
- SageMaker notebook instance or local environment with AWS CLI configured

## Cleanup

To avoid ongoing charges, delete resources when done:

```python
# In the notebook
predictor.delete_endpoint()
predictor.delete_model()
```

## References

- [Qwen-Image GitHub](https://github.com/QwenLM/Qwen-Image)
- [Qwen-Image HuggingFace](https://huggingface.co/Qwen/Qwen-Image-2512)
- [SGLang Diffusion Blog](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)
- [SGLang Documentation](https://docs.sglang.io)
- [vLLM-Omni Documentation](https://docs.vllm.ai/projects/vllm-omni)
- [SageMaker BYOC Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html)
