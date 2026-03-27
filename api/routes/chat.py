"""聊天补全路由

处理聊天补全API请求。
"""

import time
import json
from typing import AsyncGenerator, Any

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from ..schemas import ChatRequest

router = APIRouter()


def extract_video_params(messages: list) -> dict:
    """从消息列表中提取视频参数（如 fps）"""
    video_params = {}
    
    if not messages:
        return video_params

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                item_type = item.get("type", "")
                if item_type == "video_url":
                    if "fps" in item:
                        video_params["fps"] = item["fps"]
                elif item_type == "video":
                    if "fps" in item:
                        video_params["fps"] = item["fps"]
                    for key in ["total_pixels", "min_pixels"]:
                        if key in item:
                            video_params[key] = item[key]

    return video_params


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    http_request: Request,
) -> Any:
    from services import ChatService

    config = http_request.app.state.config
    service = ChatService(config)

    request_data = {
        "model": request.model,
        "messages": request.messages,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "stream": request.stream,
        "client_ip": http_request.client.host if http_request.client else "unknown",
    }

    # 验证消息
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    video_params = extract_video_params(request.messages)
    if video_params:
        request_data["media_io_kwargs"] = {"video": video_params}

    if config.logging.log_requests:
        from utils import log_request
        log_request(
            config.logging.log_dir,
            config.logging.request_log_file,
            request_data=request_data,
        )

    if request.stream:
        return StreamingResponse(
            service.stream_completion(request_data),
            media_type="text/event-stream",
        )
    else:
        return await service.non_stream_completion(request_data)
