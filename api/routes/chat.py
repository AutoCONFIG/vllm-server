"""聊天补全路由

处理聊天补全API请求。
"""

import json
from typing import Any

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from ..schemas import ChatRequest

router = APIRouter()


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

    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    if request.media_io_kwargs:
        request_data["media_io_kwargs"] = request.media_io_kwargs
    if request.mm_processor_kwargs:
        request_data["mm_processor_kwargs"] = request.mm_processor_kwargs
    if request.chat_template:
        request_data["chat_template"] = request.chat_template

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