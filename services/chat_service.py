"""聊天服务

处理聊天补全的核心逻辑。
"""

import time
import json
from typing import AsyncGenerator, Optional, Dict, Any

from vllm import SamplingParams

from core import engine_manager, EngineNotInitializedError
from multimodal import messages_to_multimodal_prompt, MultiModalProcessor
from api.schemas import ChatResponse, ChatChoice, ChatMessage, Usage


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
    ) -> ChatResponse:
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
        model = request.get("model", "qwen-3.5")
        stream = request.get("stream", False)

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
        if isinstance(engine_input, dict):
            async for output in engine.generate(engine_input, sampling_params, request_id):
                final_output = output
        else:
            async for output in engine.generate(prompt, sampling_params, request_id):
                final_output = output

        if final_output is None:
            return {"error": "Generation failed"}

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
            from vllm.inputs.data import TextPrompt

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
