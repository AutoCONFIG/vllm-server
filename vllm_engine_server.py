#!/usr/bin/env python3
"""
vLLM Engine 服务器
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from threading import Lock
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

# 抑制可避免的警告
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cuda\\.(cudart|nvrtc) module is deprecated.*")

# ============================================================================
# 日志配置
# ============================================================================

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "api_requests.log")
SERVER_LOG_FILE = os.path.join(LOG_DIR, "server.log")
LOG_LOCK = Lock()

# 确保日志目录存在
os.makedirs(LOG_DIR, exist_ok=True)

def log_request(request_data: dict, response_data: Optional[dict] = None):
    """将API请求数据记录到日志文件"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request": request_data,
            "response": response_data,
        }
        
        with LOG_LOCK:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️  日志写入失败: {e}")


def setup_server_logging():
    """配置服务器日志输出到文件"""
    # 创建日志处理器
    file_handler = logging.FileHandler(SERVER_LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # 配置 uvicorn 的日志
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.addHandler(file_handler)
    uvicorn_logger.setLevel(logging.INFO)
    
    # 配置 uvicorn.access 的日志
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.addHandler(file_handler)
    access_logger.setLevel(logging.INFO)


# ============================================================================
# 默认配置
# ============================================================================

MODEL_PATH = None
HOST = None
PORT = None
LOG_REQUESTS = False  # 默认不记录请求

# ============================================================================
# Engine参数配置
# ============================================================================

def get_engine_args(
    model_path: Optional[str] = MODEL_PATH,
    gpu_util: float = 0.95,
    max_len: int = 8192,
    max_seqs: int = 64,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    distributed_executor_backend: str = "mp",
    limit_mm_per_prompt: Optional[dict] = None,
    disable_prefix_caching: bool = False,  # 禁用前缀缓存以避免Mamba警告
) -> AsyncEngineArgs:
    """Engine参数配置，支持多卡并行和多模态输入"""
    
    # 验证必需参数
    if model_path is None:
        raise ValueError("model_path 不能为 None")
    
    args = {
        "model": model_path,
        # 注意: trust_remote_code 参数仅用于transformers的Auto类，
        # 在AsyncEngineArgs中不生效，因此移除以避免警告
        "dtype": "bfloat16",
        
        # 显存配置
        "gpu_memory_utilization": gpu_util,
        "max_model_len": max_len,
        "max_num_seqs": max_seqs,
        
        # 性能优化
        "enable_chunked_prefill": True,
        "enable_prefix_caching": not disable_prefix_caching,  # 可选禁用前缀缓存
        
        # 调度配置
        "scheduling_policy": "fcfs",
        "enforce_eager": False,
        "seed": 42,
        
        # 多卡并行配置
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "distributed_executor_backend": distributed_executor_backend,
        
        # 禁用不支持的优化以避免警告
        "disable_custom_all_reduce": True,
    }
    
    # 多模态配置
    if limit_mm_per_prompt is not None:
        args["limit_mm_per_prompt"] = limit_mm_per_prompt
    
    return AsyncEngineArgs(**args)


# ============================================================================
# API模型
# ============================================================================

class ChatRequest(BaseModel):
    model: str = Field(default="qwen-3.5")
    messages: list[dict]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = Field(default=False)
    
    class Config:
        # 允许任意字典类型（用于多模态内容）
        extra = "allow"


# ============================================================================
# Engine管理
# ============================================================================

class EngineManager:
    _instance: Optional["EngineManager"] = None
    _engine: Optional[AsyncLLMEngine] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def initialize(self, args):
        if self._engine is None:
            print("🚀 正在初始化vLLM AsyncLLMEngine...")
            print(f"📁 模型路径: {args.model}")
            print(f"🎯 GPU利用率: {args.gpu_util}")
            print(f"📝 最大长度: {args.max_len}")
            
            # 多卡并行信息
            total_gpus = args.tensor_parallel_size * args.pipeline_parallel_size * args.data_parallel_size
            if total_gpus > 1:
                print(f"🖥️  多卡配置: TP={args.tensor_parallel_size}, PP={args.pipeline_parallel_size}, DP={args.data_parallel_size}")
                print(f"🖥️  总GPU数: {total_gpus}")
                print(f"🔧 分布式后端: {args.distributed_executor_backend}")
            
            # 多模态信息
            if hasattr(args, 'limit_mm_per_prompt') and args.limit_mm_per_prompt is not None:
                print(f"🖼️  多模态限制: {args.limit_mm_per_prompt}")
            
            # 前缀缓存配置信息
            if hasattr(args, 'disable_prefix_caching') and args.disable_prefix_caching:
                print(f"⚙️  前缀缓存已禁用（避免Mamba警告）")
            
            engine_args = get_engine_args(
                model_path=args.model,
                gpu_util=args.gpu_util,
                max_len=args.max_len,
                max_seqs=args.max_seqs,
                tensor_parallel_size=args.tensor_parallel_size,
                pipeline_parallel_size=args.pipeline_parallel_size,
                distributed_executor_backend=args.distributed_executor_backend,
                limit_mm_per_prompt=args.limit_mm_per_prompt if hasattr(args, 'limit_mm_per_prompt') else None,
                disable_prefix_caching=args.disable_prefix_caching if hasattr(args, 'disable_prefix_caching') else False,
            )
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            print("⏳ 等待模型加载完成...")
            # 等待EngineCore准备就绪
            await asyncio.sleep(2)
            print("✅ Engine初始化完成!")
    
    @property
    def engine(self) -> AsyncLLMEngine:
        if self._engine is None:
            raise RuntimeError("Engine未初始化")
        return self._engine
    
    async def shutdown(self):
        if self._engine is not None:
            print("🛑 正在关闭Engine...")
            await self._engine.shutdown()
            self._engine = None
            print("✅ Engine已关闭")


engine_manager = EngineManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    args = app.state.args
    await engine_manager.initialize(args)
    yield
    await engine_manager.shutdown()


app = FastAPI(
    title="vLLM Engine Server",
    description="vLLM Engine 服务器",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": MODEL_PATH,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen-3.5",
                "object": "model",
                "created": 1700000000,
                "owned_by": "vllm",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, http_request: Request):
    """处理聊天补全请求，支持多模态输入"""
    engine = engine_manager.engine
    
    # 记录请求
    request_data = {
        "model": request.model,
        "messages": request.messages,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "stream": request.stream,
        "client_ip": http_request.client.host if http_request.client else "unknown",
    }
    if LOG_REQUESTS:
        log_request(request_data)
    
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )
    
    # 转换消息，提取多模态数据
    prompt, multi_modal_data = messages_to_prompt(request.messages)
    
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(engine, prompt, multi_modal_data, sampling_params, request.model, request_data),
            media_type="text/event-stream",
        )
    else:
        return await non_stream_chat_completion(engine, prompt, multi_modal_data, sampling_params, request.model, request_data)


def messages_to_prompt(messages: list[dict]) -> tuple[str, dict | None]:
    """
    将消息列表转换为提示字符串，并提取多模态数据。
    
    返回: (prompt, multi_modal_data)
    - prompt: 转换后的提示字符串（包含图像/视频占位符）
    - multi_modal_data: 包含图像/视频/音频数据的字典，如果没有则为None
    """
    prompt_parts = []
    multi_modal_data = {"image": [], "video": []}  # 支持图片和视频
    image_count = 0  # 跟踪图像数量
    video_count = 0  # 跟踪视频数量
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # 处理多模态内容（数组格式）
        if isinstance(content, list):
            text_parts = []
            for item in content:
                item_type = item.get("type", "")
                if item_type == "text":
                    text_parts.append(item.get("text", ""))
                elif item_type == "image_url":
                    # 提取图像URL或base64数据
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url:
                        # 添加图像占位符到文本中（Qwen模型使用 <|vision_start|><|image_pad|><|vision_end|>）
                        text_parts.append("<|vision_start|><|image_pad|><|vision_end|>")
                        # 添加到多模态数据列表
                        multi_modal_data["image"].append(image_url)
                        image_count += 1
                elif item_type == "video_url":
                    # 提取视频URL或base64数据
                    video_url = item.get("video_url", {}).get("url", "")
                    if video_url:
                        # 添加视频占位符（Qwen模型使用 <|vision_start|><|video_pad|><|vision_end|>）
                        text_parts.append("<|vision_start|><|video_pad|><|vision_end|>")
                        # 添加到多模态数据列表
                        multi_modal_data["video"].append(video_url)
                        video_count += 1
                elif item_type == "video":
                    # 处理直接传递的视频数据（Qwen2.5-VL格式）
                    video_data = item.get("video", "")
                    if video_data:
                        # 添加视频占位符
                        text_parts.append("<|vision_start|><|video_pad|><|vision_end|>")
                        # 构建视频数据字典
                        video_item = {
                            "url": video_data if isinstance(video_data, str) else "",
                        }
                        # 添加其他参数如 total_pixels, min_pixels
                        for key in ["total_pixels", "min_pixels", "fps"]:
                            if key in item:
                                video_item[key] = item[key]
                        multi_modal_data["video"].append(video_item)
                        video_count += 1
            # 合并所有文本部分
            content = "\n".join(text_parts) if text_parts else ""
        
        if role == "system":
            prompt_parts.append(f"<|im_start|>system\n{content} <|end|>")
        elif role == "user":
            prompt_parts.append(f"<|im_start|>user\n{content} <|end|>")
        elif role == "assistant":
            prompt_parts.append(f"<|im_start|>assistant\n{content} <|end|>")
    
    prompt_parts.append("<|im_start|>assistant\n")
    
    # 如果没有多模态数据，返回None
    has_image = bool(multi_modal_data["image"])
    has_video = bool(multi_modal_data["video"])
    final_mm_data = None
    if has_image or has_video:
        final_mm_data = {}
        if has_image:
            final_mm_data["image"] = multi_modal_data["image"]
        if has_video:
            final_mm_data["video"] = multi_modal_data["video"]
    
    return "\n".join(prompt_parts), final_mm_data


async def non_stream_chat_completion(
    engine: AsyncLLMEngine,
    prompt: str,
    multi_modal_data: Optional[dict],
    sampling_params: SamplingParams,
    model: str,
    request_data: dict,
) -> dict:
    """非流式聊天补全，支持请求日志和多模态数据"""
    request_id = f"cmpl-{int(time.time() * 1000)}"
    
    final_output = None
    
    # 构建输入，支持多模态数据
    mm_data = {}
    
    if multi_modal_data:
        # 处理图像数据
        if multi_modal_data.get("image"):
            images = []
            for image_url in multi_modal_data["image"]:
                try:
                    from vllm.multimodal.utils import fetch_image
                    if image_url.startswith("data:image"):
                        import base64
                        from io import BytesIO
                        from PIL import Image
                        
                        base64_data = image_url.split(",")[1]
                        image_bytes = base64.b64decode(base64_data)
                        image = Image.open(BytesIO(image_bytes))
                        images.append(image)
                    else:
                        image = fetch_image(image_url)
                        images.append(image)
                except Exception as e:
                    print(f"⚠️  图像加载失败: {e}")
                    continue
            
            if images:
                mm_data["image"] = images[0] if len(images) == 1 else images
        
        # 处理视频数据
        if multi_modal_data.get("video"):
            videos = []
            for video_item in multi_modal_data["video"]:
                try:
                    video_url = ""
                    if isinstance(video_item, str):
                        video_url = video_item
                    elif isinstance(video_item, dict):
                        video_url = video_item.get("url", "")
                    
                    if not video_url:
                        continue
                    
                    # 加载视频
                    from vllm.multimodal.utils import fetch_video
                    
                    try:
                        # video_url 可能是:
                        # - base64编码: data:video/mp4;base64,...
                        # - HTTP URL: http://... 或 https://...
                        # 直接传递给 fetch_video，让它自动处理
                        # fetch_video 返回 (video_data, video_meta) 元组
                        video_data, video_meta = fetch_video(video_url)
                        # 需要以tuple格式存储: (video_data, metadata_dict)
                        videos.append((video_data, video_meta))
                    except Exception as e:
                        print(f"⚠️  视频加载失败: {e}")
                        continue
                            
                except Exception as e:
                    print(f"⚠️  视频处理失败: {e}")
                    continue
            
            if videos:
                mm_data["video"] = videos[0] if len(videos) == 1 else videos
    
    # 构建最终输入
    if mm_data:
        from vllm.inputs.data import TextPrompt
        multi_modal_input = TextPrompt(
            prompt=prompt,
            multi_modal_data=mm_data,
        )
    else:
        multi_modal_input = prompt
    
    # 生成响应
    if isinstance(multi_modal_input, dict):
        # 多模态输入（TextPrompt是一个dict）
        async for output in engine.generate(multi_modal_input, sampling_params, request_id):
            final_output = output
    else:
        # 纯文本输入
        async for output in engine.generate(prompt, sampling_params, request_id):
            final_output = output
    
    if final_output is None:
        error_response = {"error": "Generation failed"}
        if LOG_REQUESTS:
            log_request(request_data, error_response)
        return error_response
    
    choices = []
    for i, output in enumerate(final_output.outputs):
        choices.append({
            "index": i,
            "message": {
                "role": "assistant",
                "content": output.text,
            },
            "finish_reason": output.finish_reason,
        })
    
    usage = {
        "prompt_tokens": len(final_output.prompt_token_ids),
        "completion_tokens": sum(len(o.token_ids) for o in final_output.outputs),
        "total_tokens": len(final_output.prompt_token_ids) + sum(len(o.token_ids) for o in final_output.outputs),
    }
    
    response = {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
        "usage": usage,
    }
    
    # 记录响应
    if LOG_REQUESTS:
        log_request(request_data, response)
    
    return response


async def stream_chat_completion(
    engine: AsyncLLMEngine,
    prompt: str,
    multi_modal_data: Optional[dict],
    sampling_params: SamplingParams,
    model: str,
    request_data: dict,
) -> AsyncGenerator[str, None]:
    """流式聊天补全，支持请求日志和多模态数据"""
    request_id = f"cmpl-{int(time.time() * 1000)}"
    
    # 准备输入数据（支持多模态）
    if multi_modal_data and multi_modal_data.get("image"):
        # 从URL或base64加载图像
        images = []
        for image_url in multi_modal_data["image"]:
            try:
                if image_url.startswith("data:image"):
                    # base64编码的图像
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    
                    base64_data = image_url.split(",")[1]
                    image_bytes = base64.b64decode(base64_data)
                    image = Image.open(BytesIO(image_bytes))
                    images.append(image)
                else:
                    # URL图像 - 尝试使用vllm的工具
                    try:
                        from vllm.multimodal.utils import fetch_image
                        image = fetch_image(image_url)
                        images.append(image)
                    except:
                        # 如果vllm工具不可用，使用基本请求
                        import requests
                        from PIL import Image
                        from io import BytesIO
                        
                        response = requests.get(image_url, timeout=30)
                        image = Image.open(BytesIO(response.content))
                        images.append(image)
            except Exception as e:
                print(f"⚠️  图像加载失败: {e}")
                continue
        
        if images:
            # 使用TextPrompt类型
            from vllm.inputs.data import TextPrompt
            if len(images) == 1:
                inputs = TextPrompt(
                    prompt=prompt,
                    multi_modal_data={"image": images[0]},
                )
            else:
                inputs = TextPrompt(
                    prompt=prompt,
                    multi_modal_data={"image": images},
                )
        else:
            inputs = prompt
    else:
        inputs = prompt
    
    previous_text = ""
    full_response_text = ""
    
    # 执行生成
    # 检查是否是dict类型（TextPrompt本质上是一个TypedDict）
    is_multimodal = isinstance(inputs, dict) and "prompt" in inputs
    if is_multimodal:
        # 多模态输入
        async for output in engine.generate(inputs, sampling_params, request_id):
            for i, completion_output in enumerate(output.outputs):
                delta_text = completion_output.text[len(previous_text):]
                if delta_text:
                    previous_text = completion_output.text
                    full_response_text = completion_output.text
                    
                    response = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": i,
                                "delta": {"content": delta_text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(response)}\n\n"
    else:
        async for output in engine.generate(prompt, sampling_params, request_id):
            for i, completion_output in enumerate(output.outputs):
                delta_text = completion_output.text[len(previous_text):]
                if delta_text:
                    previous_text = completion_output.text
                    full_response_text = completion_output.text
                    
                    response = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": i,
                                "delta": {"content": delta_text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(response)}\n\n"
    
    # 发送结束消息
    final_response = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_response)}\n\n"
    yield "data: [DONE]\n\n"
    
    # 记录完整响应
    if LOG_REQUESTS and full_response_text:
        response_data = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_response_text,
                },
                "finish_reason": "stop",
            }],
        }
        log_request(request_data, response_data)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="vLLM Engine Server - 支持多卡并行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
多卡并行示例:
  # 使用4张卡进行张量并行
  python vllm_engine_server.py --tensor-parallel-size 4
  
  # 使用张量并行+流水线并行 (例如4个GPU/节点，2个节点)
  python vllm_engine_server.py --tensor-parallel-size 4 --pipeline-parallel-size 2
  
  # 使用数据并行 (需要Ray后端)
  python vllm_engine_server.py --data-parallel-size 4 --distributed-executor-backend ray
  
多模态配置示例:
  # 限制最多3张图片，不允许视频
  python vllm_engine_server.py --limit-mm-per-prompt '{"image": 3, "video": 0}'
  
  # 限制最多1个视频和2张图片
  python vllm_engine_server.py --limit-mm-per-prompt '{"video": 1, "image": 2}'
  
日志配置:
  # 启用请求日志（所有API请求将被记录到logs/api_requests.log）
  python vllm_engine_server.py --log-requests
  
实验性功能:
  # 禁用前缀缓存以避免Mamba警告
  python vllm_engine_server.py --disable-prefix-caching
        """
    )
    
    # 模型配置
    parser.add_argument("--model", "-m", type=str, default=MODEL_PATH, help="模型路径")
    
    # 显存配置
    parser.add_argument("--gpu-util", "-g", type=float, default=0.95, help="GPU显存利用率")
    parser.add_argument("--max-len", "-l", type=int, default=8192, help="最大上下文长度")
    parser.add_argument("--max-seqs", type=int, default=64, help="最大并发序列数")
    
    # 多卡并行配置
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1,
                        help="张量并行大小 (每个节点的GPU数)")
    parser.add_argument("--pipeline-parallel-size", "-pp", type=int, default=1,
                        help="流水线并行大小 (节点数)")
    parser.add_argument("--data-parallel-size", "-dp", type=int, default=1,
                        help="数据并行大小 (需要使用distributed_executor_backend=ray)")
    parser.add_argument("--distributed-executor-backend", type=str, default="mp",
                        choices=["mp", "ray"],
                        help="分布式执行后端: mp(multiprocessing)或ray")
    
    # 多模态配置
    parser.add_argument("--limit-mm-per-prompt", type=str, default=None,
                        help="限制每个提示的多模态数量，JSON格式如 '{\"image\": 3, \"video\": 1}'")
    
    # 实验性功能配置
    parser.add_argument("--disable-prefix-caching", type=str, default="false",
                        help="禁用前缀缓存（避免Mamba层的实验性警告）, true/false")
    
    # 日志配置
    parser.add_argument("--log-requests", type=str, default="false",
                        help="记录所有API请求到logs/api/api_requests.log文件 (true/false)")
    
    # 服务器配置
    parser.add_argument("--host", type=str, default=HOST, help="服务器主机")
    parser.add_argument("--port", "-p", type=int, default=PORT, help="服务器端口")
    
    args = parser.parse_args()
    
    # 更新全局日志配置
    global LOG_REQUESTS
    LOG_REQUESTS = args.log_requests.lower() == "true"
    
    # 解析disable_prefix_caching配置
    args.disable_prefix_caching = args.disable_prefix_caching.lower() == "true"
    
    # 配置服务器日志
    setup_server_logging()
    
    # 解析多模态配置
    if args.limit_mm_per_prompt is not None:
        try:
            args.limit_mm_per_prompt = json.loads(args.limit_mm_per_prompt)
        except json.JSONDecodeError as e:
            print(f"❌ 错误: --limit-mm-per-prompt 必须是有效的JSON格式")
            print(f"   解析错误: {e}")
            print(f"   示例: '{{\"image\": 3, \"video\": 1}}'")
            exit(1)
    
    # 验证多卡配置
    total_gpus = args.tensor_parallel_size * args.pipeline_parallel_size * args.data_parallel_size
    if total_gpus > 1 and args.data_parallel_size > 1 and args.distributed_executor_backend != "ray":
        print("⚠️  警告: 数据并行需要使用 --distributed-executor-backend ray")
        args.distributed_executor_backend = "ray"
    
    # 设置全局args供lifespan使用
    app.state.args = args
    
    # 启动服务器
    print(f"🚀 启动vLLM Engine Server...")
    print(f"📁 模型: {args.model}")
    print(f"🎯 地址: http://{args.host}:{args.port}")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", log_config=None, access_log=True)


if __name__ == "__main__":
    main()
