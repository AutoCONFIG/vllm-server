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


def extract_video_params(messages: list) -> dict:
    """从消息列表中提取视频参数（如 fps）"""
    video_params = {}

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
    from core.engine import engine_manager

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
            _stream_chat_completion(service, request_data),
            media_type="text/event-stream",
        )
    else:
        return await service.non_stream_completion(request_data)


async def _stream_chat_completion(
    service: Any,
    request_data: dict,
) -> AsyncGenerator[str, None]:
    from core.engine import engine_manager
    engine = engine_manager.engine
    from vllm import SamplingParams
    from multimodal import messages_to_multimodal_prompt, MultiModalProcessor

    request_id = f"cmpl-{int(time.time() * 1000)}"
    config = engine_manager.config
    default_model = config.model.name if config.model.name else None
    if not default_model and config.model.path:
        default_model = config.model.path.rstrip("/").split("/")[-1]
    model = request_data.get("model", default_model)
    temperature = request_data.get("temperature", 0.7)
    top_p = request_data.get("top_p", 1.0)
    max_tokens = request_data.get("max_tokens")
    messages = request_data.get("messages", [])
    media_io_kwargs = request_data.get("media_io_kwargs")

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    prompt, multi_modal_data, video_params = messages_to_multimodal_prompt(messages)

    processor = MultiModalProcessor()
    engine_input = prompt
    if multi_modal_data:
        mm_processor_kwargs = {}
        if video_params:
            mm_processor_kwargs.update(video_params)
        if media_io_kwargs and "video" in media_io_kwargs:
            mm_processor_kwargs.update(media_io_kwargs["video"])

        mm_data = processor.build_multimodal_data(
            images=multi_modal_data.get("image"),
            videos=multi_modal_data.get("video"),
            mm_processor_kwargs=mm_processor_kwargs if mm_processor_kwargs else None,
        )
        if mm_data:
            from vllm.inputs.data import TextPrompt
            engine_input = TextPrompt(
                prompt=prompt,
                multi_modal_data=mm_data,
            )

    previous_text = ""
    full_response_text = ""

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
