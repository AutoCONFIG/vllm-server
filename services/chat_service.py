"""聊天服务

处理聊天补全的核心逻辑。
"""

import time
import json
from typing import AsyncGenerator, Optional, Dict, Any, Union

from fastapi import HTTPException
from vllm import SamplingParams
from vllm.inputs.data import TextPrompt

from core import engine_manager, EngineNotInitializedError
from multimodal import messages_to_multimodal_prompt, MultiModalProcessor


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

        # 解析请求参数
        messages = request.get("messages", [])
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p", 1.0)
        max_tokens = request.get("max_tokens")
        
        # 获取模型名称
        default_model = self.config.model.name if self.config.model.name else None
        if not default_model and self.config.model.path:
            default_model = self.config.model.path.rstrip("/").split("/")[-1]
        model = request.get("model", default_model)

        # 构建采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        # 转换消息，提取多模态数据和视频参数
        prompt, multi_modal_data, video_params = messages_to_multimodal_prompt(messages)

        # 获取 media_io_kwargs（从请求中）
        media_io_kwargs = request.get("media_io_kwargs", {})

        # 合并 video_params 和 media_io_kwargs
        mm_processor_kwargs = {}
        if video_params:
            mm_processor_kwargs.update(video_params)
        if media_io_kwargs and "video" in media_io_kwargs:
            mm_processor_kwargs.update(media_io_kwargs["video"])

        # 构建Engine输入
        engine_input = self._build_engine_input(
            prompt,
            multi_modal_data,
            mm_processor_kwargs if mm_processor_kwargs else None
        )

        # 生成响应
        final_output = None
        async for output in engine.generate(engine_input, sampling_params, request_id):
            final_output = output

        if final_output is None:
            raise HTTPException(status_code=500, detail="Generation failed")

        # 构建响应
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

        # 记录日志
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
        media_io_kwargs = request.get("media_io_kwargs")

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        prompt, multi_modal_data, video_params = messages_to_multimodal_prompt(messages)

        processor = MultiModalProcessor()
        engine_input: Union[str, TextPrompt] = prompt
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

    def _build_engine_input(
        self,
        prompt: str,
        multi_modal_data: Optional[Dict],
        mm_processor_kwargs: Optional[Dict] = None,
    ) -> Any:
        """
        构建Engine输入

        Args:
            prompt: 文本提示
            multi_modal_data: 多模态数据
            mm_processor_kwargs: 多模态处理器参数（如 fps）

        Returns:
            Any: 文本或TextPrompt字典
        """
        if multi_modal_data:
            processor = MultiModalProcessor()
            mm_data = processor.build_multimodal_data(
                images=multi_modal_data.get("image"),
                videos=multi_modal_data.get("video"),
                mm_processor_kwargs=mm_processor_kwargs,
            )
            if mm_data:
                return TextPrompt(
                    prompt=prompt,
                    multi_modal_data=mm_data,
                )
        return prompt
