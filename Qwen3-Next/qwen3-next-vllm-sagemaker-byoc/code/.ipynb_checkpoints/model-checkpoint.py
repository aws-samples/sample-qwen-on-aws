#!/usr/bin/env python3

import os
import json
import logging
import asyncio
import uvicorn
import time
import sys
from typing import Dict, List, Optional, Union, AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import torch
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "qwen3-next"
    max_tokens: Optional[int] = Field(default=16384, le=32768)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.8, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=20, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: Optional[dict] = None

class SageMakerInferenceServer:
    def __init__(self):
        self.app = FastAPI(title="Qwen3-Next SageMaker Inference Server")
        self.engine = None
        self.model_config = self._load_model_config()
        self.engine_loading = False  # Track loading state
        self.engine_failed = False   # Track failed state
        self._setup_routes()

        # Start model loading in background thread
        self.model_loading_thread = threading.Thread(
            target=self._initialize_engine_sync,
            daemon=True
        )
        self.model_loading_thread.start()

    def _initialize_engine_sync(self):
        """Initialize vLLM engine in synchronous context"""
        self.engine_loading = True
        loop = None
        try:
            logger.info("🚀 Starting model loading during container startup...")

            # Debug GPU detection BEFORE model loading
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    logger.info(f"🎮 GPU {i}: {props.name}, {memory_gb:.1f}GB, CC: {props.major}.{props.minor}")
            else:
                logger.error("❌ CUDA not available!")
                self.engine_failed = True
                return

            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run the async initialization with timeout protection
            logger.info("⏱️ Starting model initialization with 20-minute timeout...")
            try:
                loop.run_until_complete(
                    asyncio.wait_for(
                        self._initialize_engine(skip_state_check=True),
                        timeout=1200  # 20 minutes timeout
                    )
                )
                logger.info("✅ Model loaded successfully during startup!")
            except asyncio.TimeoutError:
                logger.error("❌ Model loading timed out after 20 minutes")
                self.engine_failed = True
                return

        except Exception as e:
            logger.error(f"❌ Failed to load model during startup: {e}")
            logger.exception("Full error traceback:")
            self.engine_failed = True
        finally:
            self.engine_loading = False
            if loop:
                loop.close()

    def _load_model_config(self) -> Dict:
        """Load model configuration from serving.properties"""
        config = {}
        serving_properties_path = "/opt/ml/code/serving.properties"

        if os.path.exists(serving_properties_path):
            with open(serving_properties_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            if key.startswith('option.'):
                                config_key = key.replace('option.', '')
                                # Handle special value types
                                if value.lower() in ['true', 'false']:
                                    config[config_key] = value.lower() == 'true'
                                elif value.isdigit():
                                    config[config_key] = int(value)
                                elif value.replace('.', '', 1).isdigit():
                                    config[config_key] = float(value)
                                elif value.startswith('{') and value.endswith('}'):
                                    try:
                                        config[config_key] = json.loads(value)
                                    except json.JSONDecodeError:
                                        config[config_key] = value
                                else:
                                    config[config_key] = value

        # Set defaults
        defaults = {
            'model_id': 'Qwen/Qwen3-Next-80B-A3B-Instruct',
            'tensor_parallel_degree': 4,
            'max_model_len': 32768,
            'gpu_memory_utilization': 0.8,
            'dtype': 'auto',
            'trust_remote_code': True,
            'served_model_name': 'qwen3-next'
        }

        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value

        return config

    async def _initialize_engine(self, skip_state_check=False):
        """Initialize the vLLM engine"""
        if self.engine is not None:
            logger.info("🔄 Engine already initialized, skipping...")
            return

        # Skip state checking when called from background thread to avoid deadlock
        if not skip_state_check:
            if self.engine_loading:
                logger.info("🔄 Engine is loading, waiting...")
                # Wait for loading to complete
                while self.engine_loading and not self.engine_failed:
                    await asyncio.sleep(1)
                if self.engine_failed:
                    raise RuntimeError("Engine initialization failed previously")
                return

            if self.engine_failed:
                raise RuntimeError("Engine initialization failed previously")

        logger.info("🚀 Initializing vLLM engine...")
        logger.info(f"📦 Model ID: {self.model_config.get('model_id')}")

        try:
            # Convert serving.properties config to vLLM engine args
            engine_args = AsyncEngineArgs(
                model=self.model_config.get('model_id'),
                tensor_parallel_size=self.model_config.get('tensor_parallel_degree', 4),
                max_model_len=self.model_config.get('max_model_len', 32768),
                gpu_memory_utilization=self.model_config.get('gpu_memory_utilization', 0.8),
                dtype=self.model_config.get('dtype', 'auto'),
                trust_remote_code=self.model_config.get('trust_remote_code', True),
                tokenizer_mode=self.model_config.get('tokenizer_mode', 'auto'),
                enforce_eager=self.model_config.get('enforce_eager', False),
                max_num_seqs=self.model_config.get('max_num_seqs', 64),
                swap_space=self.model_config.get('swap_space', 4),
                enable_prefix_caching=self.model_config.get('enable_prefix_caching', False),
                disable_log_stats=self.model_config.get('disable_log_stats', False),
                block_size=self.model_config.get('block_size', 16),
                seed=self.model_config.get('seed', 0),
                load_format=self.model_config.get('load_format', 'auto'),
                revision=self.model_config.get('revision', 'main')
            )

            # Handle speculative config if present
            if 'speculative_config' in self.model_config:
                logger.info(f"Using speculative config: {self.model_config['speculative_config']}")
                engine_args.speculative_config = self.model_config['speculative_config']

            logger.info("Creating AsyncLLMEngine...")
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info("✅ vLLM engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {str(e)}")
            logger.exception("Full error traceback:")
            logger.error(f"Model config: {self.model_config}")
            self.engine_failed = True

            # Try without speculative config as fallback
            if 'speculative_config' in self.model_config:
                logger.info("Retrying without speculative config...")
                try:
                    engine_args_fallback = AsyncEngineArgs(
                        model=self.model_config.get('model_id'),
                        tensor_parallel_size=self.model_config.get('tensor_parallel_degree', 4),
                        max_model_len=self.model_config.get('max_model_len', 32768),
                        gpu_memory_utilization=self.model_config.get('gpu_memory_utilization', 0.8),
                        dtype=self.model_config.get('dtype', 'auto'),
                        trust_remote_code=self.model_config.get('trust_remote_code', True),
                        tokenizer_mode=self.model_config.get('tokenizer_mode', 'auto'),
                        enforce_eager=self.model_config.get('enforce_eager', False),
                        max_num_seqs=self.model_config.get('max_num_seqs', 64),
                        swap_space=self.model_config.get('swap_space', 4),
                        enable_prefix_caching=self.model_config.get('enable_prefix_caching', False),
                        disable_log_stats=self.model_config.get('disable_log_stats', False),
                        block_size=self.model_config.get('block_size', 16),
                        seed=self.model_config.get('seed', 0),
                        load_format=self.model_config.get('load_format', 'auto'),
                        revision=self.model_config.get('revision', 'main')
                    )
                    self.engine = AsyncLLMEngine.from_engine_args(engine_args_fallback)
                    self.engine_failed = False  # Reset failed state on success
                    logger.info("✅ vLLM engine initialized successfully (without speculative config)")
                except Exception as e2:
                    logger.error(f"Fallback initialization also failed: {str(e2)}")
                    logger.exception("Fallback error traceback:")
                    self.engine_failed = True
                    raise e2
            else:
                raise e

    def _setup_routes(self):
        @self.app.get("/ping")
        async def ping():
            """Health check endpoint required by SageMaker"""
            # logger.info("📡 Ping endpoint called")

            if self.engine_failed:
                logger.error("🚨 Engine failed to initialize")
                return JSONResponse(
                    status_code=503,
                    content={"status": "failed", "message": "Engine initialization failed"}
                )

            if self.engine is None:
                if self.engine_loading:
                    logger.info("⏳ Engine is still loading")
                    return JSONResponse(
                        status_code=503,
                        content={"status": "loading", "message": "Model is still loading..."}
                    )
                else:
                    logger.warning("❓ Engine not initialized and not loading")
                    return JSONResponse(
                        status_code=503,
                        content={"status": "not_ready", "message": "Engine not initialized"}
                    )

            logger.info("✅ Ping successful - engine ready")
            return {"status": "healthy", "model_loaded": True}

        @self.app.get("/health")
        async def health():
            """Detailed health check"""
            return {
                "status": "healthy" if self.engine else "loading",
                "model_loaded": self.engine is not None,
                "model_loading": self.engine_loading,
                "model_failed": self.engine_failed,
                "model_id": self.model_config.get('model_id'),
                "timestamp": time.time()
            }

        @self.app.post("/invocations")
        async def invocations(request: Request):
            """Main inference endpoint for SageMaker"""
            logger.info("📨 Invocations endpoint called")

            if self.engine_failed:
                logger.error("🚨 Engine failed, rejecting request")
                raise HTTPException(status_code=503, detail="Engine initialization failed")

            if self.engine is None:
                logger.warning("⏳ Engine not ready, initializing...")
                try:
                    await self._initialize_engine()
                except Exception as e:
                    logger.error(f"❌ Failed to initialize engine for request: {e}")
                    raise HTTPException(status_code=503, detail="Failed to initialize engine")

            try:
                body = await request.json()
                logger.info(f"📨 Request body keys: {list(body.keys()) if isinstance(body, dict) else 'non-dict'}")

                # Handle both direct text input and chat completion format
                if isinstance(body, dict):
                    if "messages" in body:
                        # Chat completion format
                        chat_request = ChatCompletionRequest(**body)
                        return await self._handle_chat_completion(chat_request)
                    elif "inputs" in body:
                        # Direct text input
                        prompt = body["inputs"]
                        parameters = body.get("parameters", {})
                        return await self._handle_text_generation(prompt, parameters)
                    else:
                        # Assume direct prompt
                        prompt = body.get("prompt", str(body))
                        return await self._handle_text_generation(prompt, {})
                else:
                    # String input
                    return await self._handle_text_generation(str(body), {})

            except Exception as e:
                logger.error(f"❌ Error in invocations endpoint: {str(e)}")
                logger.exception("Full error traceback:")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """OpenAI-compatible chat completions endpoint"""
            logger.info("📨 Chat completions endpoint called")

            if self.engine_failed:
                logger.error("🚨 Engine failed, rejecting request")
                raise HTTPException(status_code=503, detail="Engine initialization failed")

            if self.engine is None:
                logger.warning("⏳ Engine not ready, initializing...")
                try:
                    await self._initialize_engine()
                except Exception as e:
                    logger.error(f"❌ Failed to initialize engine for request: {e}")
                    raise HTTPException(status_code=503, detail="Failed to initialize engine")

            return await self._handle_chat_completion(request)

    async def _handle_chat_completion(self, request: ChatCompletionRequest):
        """Handle chat completion requests"""
        try:
            # Convert messages to prompt
            prompt = self._messages_to_prompt(request.messages)
            logger.info(f"🔤 Generated prompt (first 200 chars): {prompt[:200]}...")

            # Create sampling parameters with proper stop tokens for Qwen3-Next
            stop_tokens = request.stop if request.stop else ["<|im_end|>", "<|endoftext|>"]
            if isinstance(stop_tokens, str):
                stop_tokens = [stop_tokens]

            # Ensure we have the Qwen3-Next stop tokens
            if "<|im_end|>" not in stop_tokens:
                stop_tokens.append("<|im_end|>")

            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
                stop=stop_tokens
            )
            logger.info(f"⚙️ Sampling params: temp={request.temperature}, top_p={request.top_p}, max_tokens={request.max_tokens}, stop={stop_tokens}")

            request_id = random_uuid()
            logger.info(f"🆔 Request ID: {request_id}")

            if request.stream:
                logger.info("📡 Starting streaming response...")
                return StreamingResponse(
                    self._stream_chat_completion(prompt, sampling_params, request_id, request.model),
                    media_type="text/plain"
                )
            else:
                # Non-streaming response
                logger.info("⏳ Starting non-streaming generation...")
                results = []

                # Add timeout to prevent hanging
                generation_start = time.time()
                try:
                    async for output in self.engine.generate(prompt, sampling_params, request_id):
                        # Check timeout manually since asyncio.wait_for doesn't work with async generators
                        if time.time() - generation_start > 300:  # 5 minutes timeout
                            logger.error("❌ Generation timed out after 5 minutes")
                            raise asyncio.TimeoutError("Generation timeout")
                        logger.info(f"📦 Received output chunk: {len(results)} outputs so far")
                        results.append(output)

                        # Log first few characters of generated text for debugging
                        if output.outputs and len(output.outputs) > 0:
                            text_preview = output.outputs[0].text[:50] if output.outputs[0].text else "No text"
                            logger.info(f"📝 Generated text preview: {text_preview}...")

                except asyncio.TimeoutError:
                    logger.error("❌ Generation timed out after 5 minutes")
                    raise HTTPException(status_code=504, detail="Generation timed out")

                if not results:
                    logger.error("❌ No results generated")
                    raise HTTPException(status_code=500, detail="No results generated")

                final_output = results[-1]
                logger.info(f"✅ Generation completed with {len(results)} outputs")

                if not final_output.outputs or len(final_output.outputs) == 0:
                    logger.error("❌ Final output has no text")
                    raise HTTPException(status_code=500, detail="No text generated")

                generated_text = final_output.outputs[0].text
                logger.info(f"📄 Final generated text length: {len(generated_text)} characters")

                response = ChatCompletionResponse(
                    id=request_id,
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text
                        },
                        "finish_reason": "stop"
                    }]
                )

                logger.info("✅ Chat completion response ready")
                return response.dict()

        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _handle_text_generation(self, prompt: str, parameters: Dict):
        """Handle direct text generation requests"""
        try:
            # Set proper stop tokens for text generation
            stop_tokens = parameters.get("stop", ["<|im_end|>", "<|endoftext|>"])
            if isinstance(stop_tokens, str):
                stop_tokens = [stop_tokens]

            sampling_params = SamplingParams(
                temperature=parameters.get("temperature", 0.7),
                top_p=parameters.get("top_p", 0.8),
                top_k=parameters.get("top_k", 20),
                max_tokens=parameters.get("max_tokens", 16384),
                stop=stop_tokens
            )

            logger.info(f"🔤 Text generation prompt (first 200 chars): {prompt[:200]}...")
            logger.info(f"⚙️ Text generation sampling params: {sampling_params}")

            request_id = random_uuid()
            results = []

            generation_start = time.time()
            async for output in self.engine.generate(prompt, sampling_params, request_id):
                # Check timeout
                if time.time() - generation_start > 300:  # 5 minutes timeout
                    logger.error("❌ Text generation timed out after 5 minutes")
                    raise asyncio.TimeoutError("Generation timeout")

                results.append(output)
                logger.info(f"📦 Text generation output chunk received: {len(results)} outputs so far")

            if not results:
                logger.error("❌ No text generation results")
                raise HTTPException(status_code=500, detail="No results generated")

            final_output = results[-1]
            generated_text = final_output.outputs[0].text
            logger.info(f"✅ Text generation completed, length: {len(generated_text)} characters")

            return {"generated_text": generated_text}

        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _stream_chat_completion(self, prompt: str, sampling_params: SamplingParams, request_id: str, model: str):
        """Stream chat completion responses"""
        try:
            async for output in self.engine.generate(prompt, sampling_params, request_id):
                if output.outputs:
                    delta_text = output.outputs[0].text

                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": delta_text},
                            "finish_reason": None
                        }]
                    }

                    yield f"data: {json.dumps(chunk)}\n\n"

            # Send final chunk
            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to Qwen3-Next prompt format"""
        prompt = ""
        for message in messages:
            if message.role == "system":
                prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
            elif message.role == "user":
                prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
            elif message.role == "assistant":
                prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"

        prompt += "<|im_start|>assistant\n"
        return prompt

    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the FastAPI server"""
        logger.info("🌟 Starting SageMaker Inference Server...")
        logger.info(f"📍 Server will bind to {host}:{port}")
        logger.info(f"🤖 Model: {self.model_config.get('model_id')}")
        logger.info(f"🎮 GPUs: {self.model_config.get('tensor_parallel_degree', 4)}")

        uvicorn.run(self.app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    server = SageMakerInferenceServer()
    server.run()