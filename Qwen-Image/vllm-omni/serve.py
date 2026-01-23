"""
SageMaker inference server for Qwen-Image-2512 using vLLM-Omni.

This uses vLLM's built-in OpenAI-compatible server (vllm serve --omni)
which properly handles multiple requests without shutting down workers.

Architecture:
- vLLM server runs on internal port (8000)
- Flask proxy handles SageMaker endpoints (/ping, /invocations) on port 8080
- Flask proxies requests to vLLM's /v1/chat/completions endpoint
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

# Configure logging to stdout for CloudWatch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global vLLM server process
vllm_process = None
VLLM_PORT = 8000
VLLM_URL = f"http://localhost:{VLLM_PORT}"


def start_vllm_server():
    """Start the vLLM-Omni server as a subprocess."""
    global vllm_process

    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-2512")

    logger.info(f"Starting vLLM-Omni server for model: {model_id}")

    # Build the vllm serve command - SINGLE GPU MODE for stability
    cmd = [
        "vllm", "serve", model_id,
        "--omni",
        "--port", str(VLLM_PORT),
        "--host", "0.0.0.0",
        # Memory optimizations
        "--vae-use-slicing",
        "--vae-use-tiling",
    ]

    logger.info("Using single-GPU config for stability")
    logger.info(f"vLLM command: {' '.join(cmd)}")

    # Start the vLLM server process
    vllm_process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        preexec_fn=os.setsid  # Create new process group for cleanup
    )

    logger.info(f"vLLM server started with PID: {vllm_process.pid}")

    # Wait for server to be ready
    wait_for_server()

    return vllm_process


def wait_for_server(timeout=600):
    """Wait for vLLM server to be ready."""
    logger.info(f"Waiting for vLLM server to be ready (timeout: {timeout}s)...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = http_requests.get(f"{VLLM_URL}/health", timeout=5)
            if response.status_code == 200:
                logger.info("vLLM server is ready!")
                return True
        except http_requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            logger.debug(f"Health check failed: {e}")

        # Check if process died
        if vllm_process and vllm_process.poll() is not None:
            logger.error(f"vLLM server process died with code: {vllm_process.returncode}")
            raise RuntimeError("vLLM server failed to start")

        time.sleep(5)

    raise TimeoutError(f"vLLM server not ready after {timeout} seconds")


def shutdown_vllm_server():
    """Gracefully shutdown the vLLM server."""
    global vllm_process
    if vllm_process:
        logger.info("Shutting down vLLM server...")
        os.killpg(os.getpgid(vllm_process.pid), signal.SIGTERM)
        vllm_process.wait(timeout=30)
        logger.info("vLLM server shut down")


@app.route("/ping", methods=["GET"])
def health_check():
    """SageMaker health check endpoint."""
    try:
        response = http_requests.get(f"{VLLM_URL}/health", timeout=5)
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
        "width": 1024,                              # optional
        "height": 1024,                             # optional
        "num_inference_steps": 30,                  # optional
        "guidance_scale": 7.5,                      # optional
        "seed": 42                                  # optional
    }

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
        num_inference_steps = data.get("num_inference_steps", 30)
        guidance_scale = data.get("guidance_scale", 7.5)
        seed = data.get("seed", None)

        # Set random seed if not provided
        if seed is None:
            import random
            seed = random.randint(0, 2**32 - 1)

        logger.info(f"=== New Generation Request ===")
        logger.info(f"Prompt: {prompt[:100]}...")
        logger.info(f"Parameters: {width}x{height}, steps={num_inference_steps}, cfg={guidance_scale}")
        logger.info(f"Seed: {seed}")  # Log seed prominently

        # Build request for vLLM's OpenAI-compatible API
        extra_body = {
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": guidance_scale,
            "seed": seed,
        }

        # Add negative prompt if provided
        if negative_prompt:
            extra_body["negative_prompt"] = negative_prompt

        vllm_request = {
            "model": os.environ.get("MODEL_ID", "Qwen/Qwen-Image-2512"),
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "extra_body": extra_body
        }

        # Log the full request for debugging
        logger.info(f"vLLM request extra_body: {json.dumps(extra_body)}")

        # Call vLLM server
        response = http_requests.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=vllm_request,
            timeout=300  # 5 minute timeout for generation
        )

        if response.status_code != 200:
            logger.error(f"vLLM error: {response.text}")
            return jsonify({"error": f"vLLM server error: {response.text}"}), 500

        result = response.json()

        # Log the response structure for debugging
        logger.info(f"vLLM response keys: {result.keys()}")
        if "usage" in result:
            logger.info(f"Usage info: {result['usage']}")

        # Extract image from response
        # Response format: choices[0].message.content contains image data
        content = result["choices"][0]["message"]["content"]

        # Find image in content (could be list or string)
        image_b64 = None
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:image"):
                        # Extract base64 data from data URL
                        image_b64 = image_url.split(",", 1)[1]
                        break
        elif isinstance(content, str):
            if content.startswith("data:image"):
                image_b64 = content.split(",", 1)[1]
            else:
                # Might be direct base64
                image_b64 = content

        if not image_b64:
            logger.error(f"Could not extract image from response: {result}")
            return jsonify({"error": "Failed to extract image from response"}), 500

        logger.info(f"Image generated successfully! Seed used: {seed}")

        return jsonify({
            "image": image_b64,
            "seed": seed,
        })

    except http_requests.exceptions.Timeout:
        logger.error("vLLM request timed out")
        return jsonify({"error": "Generation timed out"}), 504
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Register cleanup handler
    import atexit
    atexit.register(shutdown_vllm_server)

    # Handle SIGTERM gracefully
    def sigterm_handler(signum, frame):
        logger.info("Received SIGTERM, shutting down...")
        shutdown_vllm_server()
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    # Start vLLM server
    start_vllm_server()

    # Start Flask proxy server on SageMaker's expected port
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting SageMaker proxy server on port {port}...")

    app.run(host="0.0.0.0", port=port, threaded=True)
