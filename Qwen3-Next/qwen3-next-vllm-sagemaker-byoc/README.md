# Qwen3-Next-80B-A3B-Instruct vLLM BYOC Deployment on SageMaker Guide

This repository provides a complete solution for deploying **Qwen3-Next-80B-A3B-Instruct** on Amazon SageMaker using a custom vLLM container with the "Bring Your Own Container" (BYOC) approach.

## 🤖 Overview

Qwen3-Next-80B-A3B-Instruct is Alibaba's latest large language model featuring advanced capabilities including multi-token prediction, tool calling, and mathematical reasoning. This solution leverages vLLM's high-performance inference engine with tensor parallelism for optimal deployment on AWS infrastructure.

**Key Highlights:**
- 🚀 **High Performance**: vLLM engine with Multi-Token Prediction (MTP) for faster inference
- 🔧 **Production Ready**: Optimized for ml.g6e.12xlarge instances with 4x NVIDIA L40S GPUs
- 🎯 **Cost Optimized**: Sparse MoE architecture (3B activated from 80B total parameters)

## 📋 Supported Models

| Model | Parameters | Activated | Context Length | vLLM Support | Instance Type |
|-------|------------|-----------|----------------|--------------|---------------|
| Qwen3-Next-80B-A3B-Instruct | 80B | 3B | 100K tokens * | ✅ v0.10.2+ | ml.g6e.12xlarge |

*The Qwen 3 Next model supports 256K content length and is extensible to 1M tokens. The 100K content length limitation applies when using the g6e.12xlarge instance. If you need longer context length, use a larger GPU instance with more GPU memory.

## 🚀 Quick Start

### Prerequisites

1. **AWS Setup**:
   - AWS CLI configured with appropriate permissions
   - SageMaker execution role with ECR, S3, and SageMaker permissions
   - Access to ml.g6e.12xlarge instances (service quota check required)

2. **Container Preparation**:
   ```bash
    # Make the script executable
    chmod +x build_and_push.sh
    
    # Build and push to ECR (includes vLLM v0.10.0+)
    ./build_and_push.sh
   ```
   
3. **Run Deployment Notebook**:
   Open `deploy_qwen3_next.ipynb` in SageMaker Studio or Jupyter and execute all cells.

4. **Test Deployment**:
   The notebook includes multiple test scenarios:
   - Basic chat completion
   - Code generation
   - Scientific explanations
   - Strands Agents integration


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Important Notes

- **Costs**: ml.g6e.12xlarge instances cost ~$10/hour - remember to delete endpoints when not in use
- **Quotas**: Ensure sufficient service quotas for GPU instances in your region
- **Security**: Review IAM permissions and network configurations for production deployments
- **Compliance**: Ensure model usage complies with your organization's AI/ML policies
