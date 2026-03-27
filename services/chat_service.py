"""聊天服务

处理聊天补全的核心逻辑。
"""

import time
import json
from typing import AsyncGenerator, Optional, Dict, Any, List

from fastapi import HTTPException
from vllm import SamplingParams

from core import engine_manager, EngineNotInitializedError


class ChatService:
    """聊天补全服务"""

    def __init__(self, config: object):
        """
        初始化聊天服务

        Args:
            config: 配置对象
        """
        self.config = config

    async def non_stream_completion(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        非流式聊天补全

        Args:
            request: 请求字典

        Returns:
            ChatResponse: 聊天响应
        """
        engine = engine_manager.engine
        request_id = f"cmpl-{int(time.time() * 1000)}"

        messages = request.get("messages", [])
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p", 1.0)
        max_tokens = request.get("max_tokens")
        
        default_model = self.config.model.name if self.config.model.name else None
        if not default_model and self.config.model.path:
            default_model = self.config.model.path.rstrip("/").split("/")[-1]
        model = request.get("model", default_model)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        engine_prompts = await self._render_messages(messages, request)

        final_output = None
        async for output in engine.generate(engine_prompts[0], sampling_params, request_id):
            final_output = output

        if final_output is None:
            raise HTTPException(status_code=500, detail="Generation failed")

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
            "total_tokens": len(final_output.prompt_token_ids) +
                          sum(len(o.token_ids) for o in final_output.outputs),
        }

        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            "usage": usage,
        }

        if self.config.logging.log_requests:
            from utils import log_request
            log_request(
                self.config.logging.log_dir,
                self.config.logging.request_log_file,
                request_data=request,
                response_data=response,
            )

        return response

    async def stream_completion(
        self,
        request: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """
        流式聊天补全

        Args:
            request: 请求字典

        Yields:
            str: SSE 格式的响应块
        """
        engine = engine_manager.engine
        request_id = f"cmpl-{int(time.time() * 1000)}"
        default_model = self.config.model.name if self.config.model.name else None
        if not default_model and self.config.model.path:
            default_model = self.config.model.path.rstrip("/").split("/")[-1]
        model = request.get("model", default_model)
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p", 1.0)
        max_tokens = request.get("max_tokens")
        messages = request.get("messages", [])

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        engine_prompts = await self._render_messages(messages, request)

        previous_text = ""
        full_response_text = ""

        async for output in engine.generate(engine_prompts[0], sampling_params, request_id):
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

        if self.config.logging.log_requests and full_response_text:
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
                self.config.logging.log_dir,
                self.config.logging.request_log_file,
                request_data=request,
                response_data=response_data,
            )

    async def _render_messages(
        self,
        messages: List[Dict[str, Any]],
        request: Dict[str, Any],
    ) -> List[Any]:
        """
        使用 vllm Renderer 渲染消息

        Args:
            messages: OpenAI 格式的消息列表
            request: 原始请求字典

        Returns:
            List: 渲染后的 engine prompts
        """
        from vllm.renderers import ChatParams
        
        engine = engine_manager.engine
        renderer = engine.renderer
        
        chat_params = ChatParams(
            chat_template=request.get("chat_template"),
            media_io_kwargs=request.get("media_io_kwargs"),
            mm_processor_kwargs=request.get("mm_processor_kwargs"),
        )
        
        _, engine_prompts = await renderer.render_chat_async(
            conversations=[messages],
            chat_params=chat_params,
        )
        
        return engine_prompts