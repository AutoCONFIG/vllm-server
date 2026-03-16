"""聊天补全路由

处理聊天补全API请求。
"""

import time
import json
from typing import AsyncGenerator, Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..schemas import ChatRequest

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    http_request: Request,
) -> Any:
    """
    聊天补全端点
    
    Args:
        request: 聊天请求
        http_request: HTTP请求对象
        
    Returns:
        Any: 聊天响应或流式响应
    """
    from services import ChatService
    from core.engine import engine_manager
    
    # 获取配置
    config = http_request.app.state.config
    
    # 创建服务
    service = ChatService(config)
    
    # 构建请求数据
    request_data = {
        "model": request.model,
        "messages": request.messages,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "stream": request.stream,
        "client_ip": http_request.client.host if http_request.client else "unknown",
    }
    
    # 记录请求
    if config.logging.log_requests:
        from utils import log_request
        log_request(
            config.logging.log_dir,
            config.logging.request_log_file,
            request_data=request_data,
        )
    
    # 处理请求
    if request.stream:
        return StreamingResponse(
            _stream_chat_completion(service, request_data),
            media_type="text/event-stream",
        )
    else:
        return await service.non_stream_completion(request_data)


async def _stream_chat_completion(
    service: Any,
    request_data: dict,
) -> AsyncGenerator[str, None]:
    """
    流式聊天补全生成器
    
    Args:
        service: 聊天服务实例
        request_data: 请求数据
        
    Yields:
        str: SSE格式的事件流
    """
    from core.engine import engine_manager
    engine = engine_manager.engine
    from vllm import SamplingParams
    from multimodal import messages_to_multimodal_prompt, MultiModalProcessor
    
    request_id = f"cmpl-{int(time.time() * 1000)}"
    model = request_data.get("model", "qwen-3.5")
    temperature = request_data.get("temperature", 0.7)
    top_p = request_data.get("top_p", 1.0)
    max_tokens = request_data.get("max_tokens")
    messages = request_data.get("messages", [])
    
    # 采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    
    # 转换消息
    prompt, multi_modal_data = messages_to_multimodal_prompt(messages)
    
    # 处理多模态数据
    processor = MultiModalProcessor()
    engine_input = prompt
    if multi_modal_data:
        mm_data = processor.build_multimodal_data(
            images=multi_modal_data.get("image"),
            videos=multi_modal_data.get("video"),
        )
        if mm_data:
            from vllm.inputs.data import TextPrompt
            engine_input = TextPrompt(
                prompt=prompt,
                multi_modal_data=mm_data,
            )
    
    previous_text = ""
    full_response_text = ""
    
    # 执行生成
    async for output in engine.generate(engine_input, sampling_params, request_id):
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
    config = engine_manager.config
    if config.logging.log_requests and full_response_text:
        from utils import log_request
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
        log_request(
            config.logging.log_dir,
            config.logging.request_log_file,
            request_data=request_data,
            response_data=response_data,
        )
