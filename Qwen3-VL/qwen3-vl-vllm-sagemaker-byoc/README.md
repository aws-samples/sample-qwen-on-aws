# Qwen3-VL Vision-Language Models on Amazon SageMaker

Deploy Qwen3-VL vision-language models on Amazon SageMaker using a custom vLLM container. This solution supports both the 8B and 30B variants of Qwen3-VL with full vision capabilities.

## 📋 Overview

This repository contains everything needed to deploy **ALL** Qwen3-VL model variants on SageMaker:

- **Qwen3-VL-4B-Instruct**: 4B parameters, entry-level, ideal for development/testing
- **Qwen3-VL-8B-Instruct**: 9B parameters, balanced performance and cost
- **Qwen3-VL-30B-A3B-Instruct**: 31B parameters (MoE), high performance
- **Qwen3-VL-235B-A22B-Instruct**: 236B parameters (MoE), maximum capability

## 🌟 Features

### Vision-Language Capabilities
- **Advanced Spatial Perception**: 2D/3D reasoning and object positioning
- **Multi-language OCR**: Support for 32 languages
- **Visual Agent Functionality**: GUI interaction and visual coding
- **Video Understanding**: Timestamp-level precision
- **Multiple Images**: Process up to 10 images per request
- **Context Length**: 256K tokens (expandable to 1M)

### Technical Features
- Built on vLLM v0.10.2 for high-performance inference
- OpenAI-compatible API endpoint
- Tool calling support 
- Configurable via environment variables
- Automatic model download from HuggingFace
- Latest transformers library from GitHub (required for qwen3_vl_moe architecture)

## 📁 Repository Structure

```
qwen3-vl/
├── Dockerfile                        # Container definition with vision support
├── build_and_push.sh                 # Script to build and push to ECR
├── serve                             # Entrypoint script for vLLM server
├── deploy_qwen3_vl_all_models.ipynb  # Complete deployment notebook
└── README.md                         # This file
```

## 🚀 Quick Start

### Prerequisites

1. This example is tested on SageMaker Notebooks with instance tyle ml.m5.8xlarge 
2. AWS CLI configured with appropriate permissions
3. Docker installed and running
4. SageMaker execution role with ECR and S3 permissions
5. Access to required SageMaker instance types

### Step 1: Build and Push Container

```bash
cd qwen3-vl
chmod +x build_and_push.sh
./build_and_push.sh
```

This will:
- Create ECR repository `qwen3-vl-byoc`
- Build Docker image with vision support
- Push to your ECR registry

### Step 2: Deploy Model

Open and run the Jupyter notebook:

```bash
jupyter notebook deploy_qwen3_vl_all_models.ipynb.ipynb
```

The notebook includes:
- Model selection (4B, 8B, 30B or 235B)
- Automated deployment
- Comprehensive testing scenarios
- Cleanup instructions


## 🔧 Customization

### Switching Between Models

In the deployment notebook, change the `MODEL_CHOICE` variable:

```python
MODEL_CHOICE = "8B"   # For Qwen3-VL-8B-Instruct
MODEL_CHOICE = "30B"  # For Qwen3-VL-30B-A3B-Instruct
```

The notebook automatically adjusts:
- Model ID
- Instance type
- Tensor parallel configuration
- Memory settings

### Modifying the Container

1. Edit `Dockerfile` to add dependencies
2. Edit `serve` script to change defaults
3. Rebuild and push:
   ```bash
   ./build_and_push.sh
   ```


## 🗑️ Cleanup

To delete the endpoint and stop billing:

```python
predictor.delete_endpoint(delete_endpoint_config=True)
```

Or use AWS Console:
1. Navigate to SageMaker → Endpoints
2. Select your endpoint
3. Choose "Delete"

## 📄 License

This solution follows the licensing terms of:
- Qwen3-VL models (Apache 2.0)
- vLLM (Apache 2.0)
- SageMaker (AWS Service Terms)
