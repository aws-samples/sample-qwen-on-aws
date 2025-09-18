#!/usr/bin/env python3

# Disable torch compilation and GPU optimizations BEFORE importing torch/vllm
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['PYTORCH_JIT'] = '0'
os.environ['TORCH_DYNAMO_DISABLE'] = '1'
os.environ['TORCH_LOGS'] = 'dynamo'
os.environ['VLLM_USE_PRECOMPILED'] = '1'
os.environ['VLLM_DISABLE_COMPILATION'] = '1'

# Disable problematic GPU optimizations that cause worker hangs
os.environ['VLLM_DISABLE_CUSTOM_ALL_REDUCE'] = '1'
os.environ['VLLM_DISABLE_CUDAGRAPH'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

# Qwen3-Next requires v1 engine, but fix queue issues with proper configuration
os.environ['VLLM_USE_V1'] = '1'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_LOGGING_LEVEL'] = 'INFO'
os.environ['VLLM_ENGINE_DISABLE_LOG_STATS'] = '1'
os.environ['VLLM_DISTRIBUTED_EXECUTOR_BACKEND'] = 'mp'

import json
import logging
import asyncio
import uvicorn
import time
import sys
from typing import Dict, List, Optional, Union, AsyncGenerator, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
import torch
from vllm import LLM, SamplingParams
from vllm.utils import random_uuid
import threading
import re
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    @field_validator('content', mode='before')
    @classmethod
    def normalize_content(cls, v):
        """Convert list content format to string, handle missing content"""
        if v is None:
            return ""  # Default to empty string if content is missing

        if isinstance(v, list):
            # Handle format like [{'text': 'message', 'type': 'text'}]
            text_parts = []
            for item in v:
                if isinstance(item, dict):
                    if 'text' in item:
                        text_parts.append(item['text'])
                    elif 'content' in item:
                        text_parts.append(item['content'])
                    else:
                        # If no 'text' or 'content' key, convert whole dict to string
                        text_parts.append(str(item))
                else:
                    text_parts.append(str(item))
            return ' '.join(text_parts)
        return v

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "qwen3-next"
    max_tokens: Optional[int] = Field(default=16384, le=32768)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.8, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=20, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: Optional[dict] = None

# Tool calling utility functions for Qwen3-Next
def supports_function_calling() -> bool:
    """Check if the current model supports function calling"""
    return True  # Qwen3-Next supports tool calling

def parse_qwen_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Parse tool calls from Qwen3-Next generated text.
    Qwen3-Next uses format: <tool_call>{"name": "function_name", "arguments": {...}}</tool_call>
    """
    tool_calls = []

    # Pattern to match Qwen3-Next tool call format
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            tool_data = json.loads(match)
            tool_call = {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_data.get("name", ""),
                    "arguments": json.dumps(tool_data.get("arguments", {}))
                }
            }
            tool_calls.append(tool_call)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse tool call: {match}, error: {e}")
            continue

    return tool_calls

def clean_qwen_tool_calls_from_text(text: str) -> str:
    """Remove tool call markers from response text"""
    # Remove <tool_call>...</tool_call> blocks
    cleaned_text = re.sub(r'<tool_call>\s*\{.*?\}\s*</tool_call>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

def detect_pending_tool_responses(messages: List[Dict[str, Any]]) -> bool:
    """Check if there are pending tool responses in the conversation"""
    for message in reversed(messages):
        if message.get("role") == "assistant":
            # Check if the assistant message contains tool calls
            tool_calls = message.get("tool_calls")
            if tool_calls:
                return True
        elif message.get("role") == "tool":
            # If we see a tool response, the tool calls were handled
            return False
    return False

def format_tools_for_qwen(tools: List[Dict[str, Any]]) -> str:
    """Format tools into instructions for Qwen3-Next"""
    if not tools:
        return ""

    tool_descriptions = []
    for tool in tools:
        function = tool.get("function", {})
        name = function.get("name", "")
        description = function.get("description", "")
        parameters = function.get("parameters", {})

        tool_desc = f"- {name}: {description}"
        if parameters.get("properties"):
            props = []
            for prop_name, prop_info in parameters["properties"].items():
                prop_type = prop_info.get("type", "string")
                prop_desc = prop_info.get("description", "")
                props.append(f"{prop_name} ({prop_type}): {prop_desc}")
            if props:
                tool_desc += f"\n  Parameters: {', '.join(props)}"

        tool_descriptions.append(tool_desc)

    tool_instruction = f"""
You have access to the following tools:
{chr(10).join(tool_descriptions)}

To use a tool, respond with:
<tool_call>
{{"name": "function_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}
</tool_call>

Use tools when they can help answer the user's question. You can use multiple tools in sequence if needed.
"""
    return tool_instruction

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
            logger.info("Starting model loading during container startup...")

            # Check GPU availability
            if not torch.cuda.is_available():
                logger.error("CUDA not available!")
                self.engine_failed = True
                return

            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} GPUs available")

            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Initialize the model engine
            self._initialize_engine_direct()
            logger.info("Model loaded successfully during startup")

        except Exception as e:
            logger.error(f"Failed to load model during startup: {e}")
            logger.exception("Model loading error traceback:")
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

        # Set defaults optimized for g6e.12xlarge memory constraints
        defaults = {
            'model_id': 'Qwen/Qwen3-Next-80B-A3B-Instruct',
            'tensor_parallel_degree': 4,
            'max_model_len': 16384,
            'gpu_memory_utilization': 0.9,
            'dtype': 'auto',
            'trust_remote_code': True,
            'served_model_name': 'qwen3-next',
            'enforce_eager': False,
            'max_num_seqs': 32,
            'block_size': 16,
            'swap_space': 4,
            'max_num_batched_tokens': 16384,
            'enable_prefix_caching': False,
            'kv_cache_dtype': 'auto',
            'disable_chunked_prefill': True,
            'tool_call_parser': 'hermes',
            'enable_auto_tool_choice': True
        }

        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value

        return config

    def _initialize_engine_direct(self):
        """Initialize the vLLM engine using synchronous LLM class"""
        if self.engine is not None:
            logger.info("Engine already initialized, skipping...")
            return

        model_id = self.model_config.get('model_id')
        logger.info(f"Initializing vLLM engine for model: {model_id}")

        try:
            # Build LLM arguments optimized for g6e.12xlarge memory constraints
            llm_args = {
                'model': self.model_config.get('model_id'),
                'tensor_parallel_size': self.model_config.get('tensor_parallel_degree', 4),
                'max_model_len': self.model_config.get('max_model_len', 16384),
                'gpu_memory_utilization': self.model_config.get('gpu_memory_utilization', 0.9),
                'dtype': self.model_config.get('dtype', 'auto'),
                'trust_remote_code': self.model_config.get('trust_remote_code', True),
                'tokenizer_mode': self.model_config.get('tokenizer_mode', 'auto'),
                'enforce_eager': self.model_config.get('enforce_eager', False),
                'max_num_seqs': self.model_config.get('max_num_seqs', 32),
                'swap_space': self.model_config.get('swap_space', 4),
                'block_size': self.model_config.get('block_size', 16),
                'seed': self.model_config.get('seed', 0),
                'load_format': self.model_config.get('load_format', 'auto'),
                'revision': self.model_config.get('revision', 'main')
            }

            # Add optional parameters if they exist
            if hasattr(LLM, 'disable_chunked_prefill') and 'disable_chunked_prefill' in self.model_config:
                llm_args['disable_chunked_prefill'] = self.model_config.get('disable_chunked_prefill', True)

            if hasattr(LLM, 'enable_prefix_caching') and 'enable_prefix_caching' in self.model_config:
                llm_args['enable_prefix_caching'] = self.model_config.get('enable_prefix_caching', False)

            if hasattr(LLM, 'kv_cache_dtype') and 'kv_cache_dtype' in self.model_config:
                llm_args['kv_cache_dtype'] = self.model_config.get('kv_cache_dtype', 'auto')

            self.engine = LLM(**llm_args)
            logger.info("vLLM engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {str(e)}")
            logger.exception("Engine initialization error:")
            self.engine_failed = True
            raise e

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
            # Convert serving.properties config to vLLM engine args for Qwen3-Next
            engine_args = AsyncEngineArgs(
                model=self.model_config.get('model_id'),
                tensor_parallel_size=self.model_config.get('tensor_parallel_degree', 4),
                max_model_len=self.model_config.get('max_model_len', 16384),
                gpu_memory_utilization=self.model_config.get('gpu_memory_utilization', 0.95),
                dtype=self.model_config.get('dtype', 'auto'),
                trust_remote_code=self.model_config.get('trust_remote_code', True),
                tokenizer_mode=self.model_config.get('tokenizer_mode', 'auto'),
                enforce_eager=self.model_config.get('enforce_eager', True),
                max_num_seqs=self.model_config.get('max_num_seqs', 8),
                swap_space=self.model_config.get('swap_space', 2),
                block_size=self.model_config.get('block_size', 8),
                seed=self.model_config.get('seed', 0),
                load_format=self.model_config.get('load_format', 'auto'),
                revision=self.model_config.get('revision', 'main')
            )

            # Add optional parameters if they exist
            try:
                # Memory optimization parameters
                if hasattr(engine_args, 'max_num_batched_tokens'):
                    engine_args.max_num_batched_tokens = self.model_config.get('max_num_batched_tokens', 8192)

                if hasattr(engine_args, 'enable_prefix_caching'):
                    engine_args.enable_prefix_caching = self.model_config.get('enable_prefix_caching', False)

                # Try to set chunked prefill based on config
                if hasattr(engine_args, 'enable_chunked_prefill'):
                    engine_args.enable_chunked_prefill = not self.model_config.get('disable_chunked_prefill', True)
                elif hasattr(engine_args, 'disable_chunked_prefill'):
                    engine_args.disable_chunked_prefill = self.model_config.get('disable_chunked_prefill', True)

                # Try to set log stats
                if hasattr(engine_args, 'disable_log_stats'):
                    engine_args.disable_log_stats = self.model_config.get('disable_log_stats', False)
            except Exception as e:
                logger.warning(f"⚠️ Could not set optional parameters: {e}")

            # Handle speculative config if present
            if 'speculative_config' in self.model_config:
                logger.info(f"Using speculative config: {self.model_config['speculative_config']}")
                engine_args.speculative_config = self.model_config['speculative_config']

            logger.info("Creating AsyncLLMEngine...")
            # Create engine without usage_context to avoid enum issues
            from vllm.engine.async_llm_engine import AsyncLLMEngine
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
                        max_model_len=self.model_config.get('max_model_len', 16384),
                        gpu_memory_utilization=self.model_config.get('gpu_memory_utilization', 0.95),
                        dtype=self.model_config.get('dtype', 'auto'),
                        trust_remote_code=self.model_config.get('trust_remote_code', True),
                        tokenizer_mode=self.model_config.get('tokenizer_mode', 'auto'),
                        enforce_eager=self.model_config.get('enforce_eager', True),
                        max_num_seqs=self.model_config.get('max_num_seqs', 8),
                        swap_space=self.model_config.get('swap_space', 2),
                        block_size=self.model_config.get('block_size', 8),
                        seed=self.model_config.get('seed', 0),
                        load_format=self.model_config.get('load_format', 'auto'),
                        revision=self.model_config.get('revision', 'main')
                    )

                    # Add optional parameters if they exist for fallback
                    try:
                        # Memory optimization parameters
                        if hasattr(engine_args_fallback, 'max_num_batched_tokens'):
                            engine_args_fallback.max_num_batched_tokens = self.model_config.get('max_num_batched_tokens', 8192)
                        if hasattr(engine_args_fallback, 'enable_prefix_caching'):
                            engine_args_fallback.enable_prefix_caching = self.model_config.get('enable_prefix_caching', False)

                        if hasattr(engine_args_fallback, 'enable_chunked_prefill'):
                            engine_args_fallback.enable_chunked_prefill = not self.model_config.get('disable_chunked_prefill', True)
                        elif hasattr(engine_args_fallback, 'disable_chunked_prefill'):
                            engine_args_fallback.disable_chunked_prefill = self.model_config.get('disable_chunked_prefill', True)
                        if hasattr(engine_args_fallback, 'disable_log_stats'):
                            engine_args_fallback.disable_log_stats = self.model_config.get('disable_log_stats', False)
                    except Exception as e:
                        logger.warning(f"⚠️ Could not set optional parameters in fallback: {e}")
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
            if self.engine_failed:
                return JSONResponse(
                    status_code=503,
                    content={"status": "failed", "message": "Engine initialization failed"}
                )

            if self.engine is None:
                if self.engine_loading:
                    return JSONResponse(
                        status_code=503,
                        content={"status": "loading", "message": "Model is still loading..."}
                    )
                else:
                    return JSONResponse(
                        status_code=503,
                        content={"status": "not_ready", "message": "Engine not initialized"}
                    )

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
            if self.engine_failed:
                raise HTTPException(status_code=503, detail="Engine initialization failed")

            if self.engine is None:
                try:
                    await self._initialize_engine()
                except Exception as e:
                    logger.error(f"Failed to initialize engine for request: {e}")
                    raise HTTPException(status_code=503, detail="Failed to initialize engine")

            try:
                body = await request.json()

                # Handle both direct text input and chat completion format
                if isinstance(body, dict):
                    if "messages" in body:
                        # Debug: Log message structure for troubleshooting
                        for i, msg in enumerate(body.get("messages", [])):
                            logger.debug(f"Message {i}: role={msg.get('role')}, "
                                       f"has_content={'content' in msg}, "
                                       f"has_tool_calls={'tool_calls' in msg}, "
                                       f"content_type={type(msg.get('content'))}")

                        try:
                            chat_request = ChatCompletionRequest(**body)
                            return await self._handle_chat_completion(chat_request)
                        except Exception as validation_error:
                            logger.error(f"ChatCompletionRequest validation failed: {validation_error}")
                            logger.error(f"Request body: {json.dumps(body, indent=2)}")
                            raise
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
                logger.error(f"Error in invocations endpoint: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """OpenAI-compatible chat completions endpoint"""
            if self.engine_failed:
                raise HTTPException(status_code=503, detail="Engine initialization failed")

            if self.engine is None:
                try:
                    await self._initialize_engine()
                except Exception as e:
                    logger.error(f"Failed to initialize engine for request: {e}")
                    raise HTTPException(status_code=503, detail="Failed to initialize engine")

            return await self._handle_chat_completion(request)

    async def _handle_chat_completion(self, request: ChatCompletionRequest):
        """Handle chat completion requests"""
        try:
            # Convert messages to prompt (with tools if provided)
            prompt = self._messages_to_prompt(request.messages, request.tools)

            # Create sampling parameters with proper stop tokens for Qwen3-Next
            stop_tokens = request.stop if request.stop else ["<|im_end|>", "<|endoftext|>"]
            if isinstance(stop_tokens, str):
                stop_tokens = [stop_tokens]

            # Ensure we have the Qwen3-Next stop tokens
            if "<|im_end|>" not in stop_tokens:
                stop_tokens.append("<|im_end|>")

            # Create sampling parameters - used for both streaming and non-streaming
            actual_sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
                stop=stop_tokens
            )

            request_id = random_uuid()

            if request.stream:
                return StreamingResponse(
                    self._stream_chat_completion_sync(prompt, actual_sampling_params, request_id, request.model),
                    media_type="text/event-stream"
                )
            else:
                # Non-streaming response
                try:
                    # Run synchronous generation in thread pool to avoid blocking
                    def sync_generation():
                        return self.engine.generate(prompt, actual_sampling_params)

                    # Execute in thread pool with timeout
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(sync_generation)
                        try:
                            results = future.result(timeout=120.0)  # 2 minutes timeout
                        except concurrent.futures.TimeoutError:
                            raise asyncio.TimeoutError("Generation timeout")

                except asyncio.TimeoutError:
                    raise HTTPException(status_code=504, detail="Generation timed out")
                except Exception as gen_error:
                    logger.error(f"Error during generation: {gen_error}")
                    raise HTTPException(status_code=500, detail=f"Generation failed: {str(gen_error)}")

                if not results or not results[-1].outputs:
                    raise HTTPException(status_code=500, detail="No text generated")

                generated_text = results[-1].outputs[0].text

                # Parse tool calls from generated text if tools were provided
                tool_calls = None
                message_content = generated_text

                if request.tools and supports_function_calling():
                    tool_calls = parse_qwen_tool_calls(generated_text)
                    if tool_calls:
                        # Clean the tool calls from the content
                        message_content = clean_qwen_tool_calls_from_text(generated_text)

                # Build the response message
                response_message = {
                    "role": "assistant",
                    "content": message_content
                }

                # Add tool calls to the message if present
                if tool_calls:
                    response_message["tool_calls"] = tool_calls

                response = ChatCompletionResponse(
                    id=request_id,
                    created=int(time.time()),
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "message": response_message,
                        "finish_reason": "tool_calls" if tool_calls else "stop"
                    }]
                )

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

            request_id = random_uuid()
            results = []

            generation_start = time.time()
            async for output in self.engine.generate(prompt, sampling_params, request_id):
                # Check timeout
                if time.time() - generation_start > 300:  # 5 minutes timeout
                    raise asyncio.TimeoutError("Generation timeout")

                results.append(output)

            if not results:
                raise HTTPException(status_code=500, detail="No results generated")

            final_output = results[-1]
            generated_text = final_output.outputs[0].text

            return {"generated_text": generated_text}

        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _stream_chat_completion_sync(self, prompt: str, sampling_params: SamplingParams, request_id: str, model: str):
        """Stream chat completion responses using synchronous engine"""
        try:
            import concurrent.futures
            import queue
            import threading
            import json

            result_queue = queue.Queue()

            def sync_streaming_generation():
                """Run synchronous generation and put results in queue"""
                try:
                    # Use synchronous generate with streaming-like behavior
                    results = self.engine.generate(prompt, sampling_params)
                    for result in results:
                        if result.outputs and len(result.outputs) > 0:
                            generated_text = result.outputs[0].text
                            result_queue.put(("data", generated_text))
                    result_queue.put(("done", None))
                except Exception as e:
                    result_queue.put(("error", str(e)))

            # Start generation in background thread
            thread = threading.Thread(target=sync_streaming_generation)
            thread.daemon = True
            thread.start()

            # Stream results as they become available
            accumulated_text = ""
            while True:
                try:
                    msg_type, content = result_queue.get(timeout=1.0)

                    if msg_type == "data":
                        # Calculate delta (new text since last chunk)
                        delta_text = content[len(accumulated_text):]
                        accumulated_text = content

                        if delta_text:
                            chunk = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": delta_text},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"

                    elif msg_type == "done":
                        # Send final chunk
                        final_chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        break

                    elif msg_type == "error":
                        error_chunk = {
                            "error": {
                                "message": content,
                                "type": "internal_error"
                            }
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                        break

                except queue.Empty:
                    # Send keep-alive chunk
                    continue

        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

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

    def _messages_to_prompt(self, messages: List[ChatMessage], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Convert chat messages to Qwen3-Next prompt format with optional tool support"""
        prompt = ""

        # Add tool instructions at the beginning if tools are provided
        if tools and supports_function_calling():
            tool_instructions = format_tools_for_qwen(tools)
            prompt += f"<|im_start|>system\n{tool_instructions}<|im_end|>\n"

        for message in messages:
            if message.role == "system":
                prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
            elif message.role == "user":
                prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
            elif message.role == "assistant":
                # Handle assistant messages with tool calls
                content = message.content or ""

                # Add tool calls to the message content if present
                if message.tool_calls:
                    tool_call_texts = []
                    for tool_call in message.tool_calls:
                        function = tool_call.get('function', {})
                        tool_call_text = f"<tool_call>\n{{\"name\": \"{function.get('name', '')}\", \"arguments\": {function.get('arguments', '{}')}}}\n</tool_call>"
                        tool_call_texts.append(tool_call_text)

                    # Combine content and tool calls
                    if content and content.strip():
                        content = content + "\n\n" + "\n\n".join(tool_call_texts)
                    else:
                        content = "\n\n".join(tool_call_texts)

                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"

            elif message.role == "tool":
                # Handle tool responses
                tool_name = message.name or 'unknown_tool'
                tool_content = message.content or ""
                prompt += f"<|im_start|>tool\nTool '{tool_name}' result: {tool_content}<|im_end|>\n"

        prompt += "<|im_start|>assistant\n"
        return prompt

    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the FastAPI server"""
        model_id = self.model_config.get('model_id')
        logger.info(f"Starting SageMaker Inference Server for {model_id}")
        logger.info(f"Server binding to {host}:{port}")

        uvicorn.run(self.app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    server = SageMakerInferenceServer()
    server.run()