"""
SageMaker inference server for Qwen-Image-2512 using SGLang Diffusion.

This uses SGLang's native diffusion model support which provides:
- Native Qwen-Image support
- Multi-GPU parallelism:
  * --num-gpus: Distributes the MMDiT model across GPUs
  * --enable-cfg-parallel: Runs positive/negative prompts in parallel (CFG parallelism)
- OpenAI-compatible /v1/images/generations endpoint
- 1.2x-5.9x speedup over vanilla Diffusers

Architecture:
- SGLang diffusion server runs on internal port (8000)
- Flask proxy handles SageMaker endpoints (/ping, /invocations) on port 8080
- Flask proxies requests to SGLang's /v1/images/generations endpoint

Endpoints:
- /ping: Health check (GET)
- /invocations: Image generation (POST)

Reference: https://lmsys.org/blog/2025-11-07-sglang-diffusion/
"""

import os
import sys
import io
import base64
import logging
import subprocess
import time
import signal
import json
import requests as http_requests
from flask import Flask, request, jsonify, Response
import torch

# Configure logging to stdout for CloudWatch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global SGLang server process
sglang_process = None
SGLANG_PORT = 8000
SGLANG_URL = f"http://localhost:{SGLANG_PORT}"


def start_sglang_server():
    """Start the SGLang diffusion server as a subprocess."""
    global sglang_process

    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-2512")

    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    logger.info(f"Starting SGLang server for model: {model_id}")
    logger.info(f"Available GPUs: {num_gpus}")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")

    # Build the sglang serve command for diffusion models
    # GitHub main has Qwen-Image-2512 in native registry - no need for --backend diffusers
    cmd = [
        "sglang", "serve",
        "--model-path", model_id,
        "--port", str(SGLANG_PORT),
        "--host", "0.0.0.0",
    ]

    # Multi-GPU configuration for diffusion models
    # Enable tensor parallelism and CFG parallelism for better performance
    if num_gpus > 1:
        cmd.extend(["--num-gpus", str(num_gpus)])
        cmd.append("--enable-cfg-parallel")
        logger.info(f"Multi-GPU mode: {num_gpus} GPUs with CFG parallelism enabled")
        logger.info("CFG parallelism runs positive/negative prompts in parallel (1.2x-5.9x speedup)")
    else:
        logger.info("Single-GPU mode (no parallelism)")

    logger.info(f"SGLang command: {' '.join(cmd)}")

    # Start the SGLang server process with retry logic for multi-GPU failures
    retry_single_gpu = False
    try:
        sglang_process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            preexec_fn=os.setsid  # Create new process group for cleanup
        )

        logger.info(f"SGLang server started with PID: {sglang_process.pid}")

        # Wait for server to be ready
        wait_for_server()

    except (RuntimeError, TimeoutError) as e:
        error_msg = str(e)
        logger.error(f"SGLang server failed to start: {error_msg}")

        # If multi-GPU was enabled and startup failed, retry with single-GPU
        if num_gpus > 1 and ("NCCL" in error_msg or "died" in error_msg or "not ready" in error_msg):
            logger.warning(f"Multi-GPU mode failed. Retrying with single-GPU mode as fallback...")
            retry_single_gpu = True

            # Clean up failed process if it exists
            if sglang_process and sglang_process.poll() is None:
                os.killpg(os.getpgid(sglang_process.pid), signal.SIGTERM)
                sglang_process.wait(timeout=10)
        else:
            raise

    # Retry with single-GPU if multi-GPU failed
    if retry_single_gpu:
        # Rebuild command without multi-GPU flags
        cmd = [
            "sglang", "serve",
            "--model-path", model_id,
            "--port", str(SGLANG_PORT),
            "--host", "0.0.0.0",
        ]
        logger.info(f"Fallback SGLang command (single-GPU): {' '.join(cmd)}")

        sglang_process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            preexec_fn=os.setsid
        )

        logger.info(f"SGLang server restarted in single-GPU mode with PID: {sglang_process.pid}")
        wait_for_server()

    return sglang_process


def wait_for_server(timeout=600):
    """Wait for SGLang server to be ready."""
    logger.info(f"Waiting for SGLang server to be ready (timeout: {timeout}s)...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = http_requests.get(f"{SGLANG_URL}/health", timeout=5)
            if response.status_code == 200:
                logger.info("SGLang server is ready!")
                return True
        except http_requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            logger.debug(f"Health check failed: {e}")

        # Check if process died
        if sglang_process and sglang_process.poll() is not None:
            logger.error(f"SGLang server process died with code: {sglang_process.returncode}")
            raise RuntimeError("SGLang server failed to start")

        time.sleep(5)

    raise TimeoutError(f"SGLang server not ready after {timeout} seconds")


def shutdown_sglang_server():
    """Gracefully shutdown the SGLang server."""
    global sglang_process
    if sglang_process:
        logger.info("Shutting down SGLang server...")
        os.killpg(os.getpgid(sglang_process.pid), signal.SIGTERM)
        sglang_process.wait(timeout=30)
        logger.info("SGLang server shut down")


@app.route("/ping", methods=["GET"])
def health_check():
    """SageMaker health check endpoint."""
    try:
        response = http_requests.get(f"{SGLANG_URL}/health", timeout=5)
        if response.status_code == 200:
            return Response(status=200)
    except:
        pass
    return Response(status=503)


@app.route("/invocations", methods=["POST"])
def invoke():
    """
    SageMaker inference endpoint for text-to-image generation.

    Request JSON:
    {
        "prompt": "A beautiful sunset over mountains",
        "negative_prompt": "blurry, low quality",  # optional
        "width": 1024,                              # optional, default 1024
        "height": 1024,                             # optional, default 1024
        "num_inference_steps": 50,                  # optional, default 50 (Qwen-Image recommended)
        "guidance_scale": 4.0,                      # optional, default 4.0 (maps to true_cfg_scale)
        "seed": 42                                  # optional, random if not set
    }

    Note: For Qwen-Image-2512, guidance_scale maps to true_cfg_scale internally.
    Recommended values from model card: num_inference_steps=50, guidance_scale=4.0

    Response JSON:
    {
        "image": "<base64_encoded_png>",
        "seed": 42
    }
    """
    try:
        data = request.get_json()

        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400

        prompt = data["prompt"]
        negative_prompt = data.get("negative_prompt", "")
        width = data.get("width", 1024)
        height = data.get("height", 1024)
        # Qwen-Image recommended: num_inference_steps=50, true_cfg_scale=4.0
        num_inference_steps = data.get("num_inference_steps", 50)
        guidance_scale = data.get("guidance_scale", 4.0)  # Maps to true_cfg_scale
        seed = data.get("seed", None)

        # Set random seed if not provided
        if seed is None:
            import random
            seed = random.randint(0, 2**32 - 1)

        logger.info(f"=== New Generation Request ===")
        logger.info(f"Prompt: {prompt[:100]}...")
        logger.info(f"Parameters: {width}x{height}, steps={num_inference_steps}, cfg={guidance_scale}")
        logger.info(f"Seed: {seed}")

        # Build request for SGLang's OpenAI-compatible images API
        # IMPORTANT: Parameters must be passed directly in the request body,
        # NOT in extra_body - SGLang ignores extra_body for image generation
        sglang_request = {
            "model": os.environ.get("MODEL_ID", "Qwen/Qwen-Image-2512"),
            "prompt": prompt,
            "n": 1,
            "size": f"{width}x{height}",
            "response_format": "b64_json",
            # Generation parameters - passed directly, not in extra_body
            "num_inference_steps": num_inference_steps,
            # IMPORTANT: Qwen-Image uses true_cfg_scale, not guidance_scale
            # true_cfg_scale is for "CFG vs guidance distillation" models like QwenImage
            "true_cfg_scale": guidance_scale,
            "seed": seed,
        }

        # Add optional parameters directly to request
        if negative_prompt:
            sglang_request["negative_prompt"] = negative_prompt

        logger.info(f"SGLang request: {json.dumps(sglang_request, indent=2)}")

        # Call SGLang server's images endpoint
        response = http_requests.post(
            f"{SGLANG_URL}/v1/images/generations",
            json=sglang_request,
            timeout=300  # 5 minute timeout for generation
        )

        if response.status_code != 200:
            logger.error(f"SGLang error: {response.text}")
            return jsonify({"error": f"SGLang server error: {response.text}"}), 500

        result = response.json()
        logger.info(f"SGLang response keys: {result.keys()}")

        # Extract image from response
        # Response format: {"data": [{"b64_json": "..."}]}
        image_b64 = None
        if "data" in result and len(result["data"]) > 0:
            image_data = result["data"][0]
            if "b64_json" in image_data:
                image_b64 = image_data["b64_json"]
            elif "url" in image_data:
                # If URL is a data URL, extract base64
                url = image_data["url"]
                if url.startswith("data:image"):
                    image_b64 = url.split(",", 1)[1]

        if not image_b64:
            logger.error(f"Could not extract image from response: {result}")
            return jsonify({"error": "Failed to extract image from response"}), 500

        logger.info(f"Image generated successfully! Seed used: {seed}")

        return jsonify({
            "image": image_b64,
            "seed": seed,
        })

    except http_requests.exceptions.Timeout:
        logger.error("SGLang request timed out")
        return jsonify({"error": "Generation timed out"}), 504
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Register cleanup handler
    import atexit
    atexit.register(shutdown_sglang_server)

    # Handle SIGTERM gracefully
    def sigterm_handler(signum, frame):
        logger.info("Received SIGTERM, shutting down...")
        shutdown_sglang_server()
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    # Start SGLang server
    start_sglang_server()

    # Start Flask proxy server on SageMaker's expected port
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting SageMaker proxy server on port {port}...")

    app.run(host="0.0.0.0", port=port, threaded=True)
