"""聊天服务

处理聊天补全的核心逻辑。
"""

import time
import json
import logging
from typing import AsyncGenerator, Optional, Dict, Any, List

from fastapi import HTTPException
from vllm.sampling_params import SamplingParams

from core import engine_manager, EngineNotInitializedError

logger = logging.getLogger(__name__)


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
        使用 vllM Renderer 渲染消息

        Args:
            messages: OpenAI 格式的消息列表
            request: 原始请求字典

        Returns:
            List: 渲染后的 engine prompts
        """
        from vllm.renderers.params import ChatParams, TokenizeParams
        from vllm.renderers import merge_kwargs
        from vllm.renderers.hf import resolve_chat_template_content_format
        
        logger.debug(f"[DEBUG] _render_messages: num_messages={len(messages)}")
        
        engine = engine_manager.engine
        renderer = engine.renderer
        model_config = engine.model_config
        
        # 获取 tokenizer
        tokenizer = renderer.get_tokenizer()
        
        # 构建 chat_params（参考 OpenAIServingRender.preprocess_chat）
        content_format = resolve_chat_template_content_format(
            chat_template=request.get("chat_template"),
            tools=None,
            given_format=None,
            tokenizer=tokenizer,
            model_config=model_config,
        )
        
        # 构建 tokenize params
        tok_params = TokenizeParams(max_total_tokens=model_config.max_model_len)
        
        # 构建 chat params，包括 media_io_kwargs 和 mm_processor_kwargs
        mm_config = model_config.multimodal_config
        chat_params = ChatParams(
            chat_template=request.get("chat_template"),
            chat_template_content_format=content_format,
        ).with_defaults(
            default_chat_template_kwargs=None,
            default_media_io_kwargs=(mm_config.media_io_kwargs if mm_config else None),
            default_mm_processor_kwargs=request.get("mm_processor_kwargs"),
        )
        
        # 合并请求中的 media_io_kwargs
        if request.get("media_io_kwargs"):
            # 手动合并 media_io_kwargs
            current_kwargs = chat_params.media_io_kwargs or {}
            request_kwargs = request["media_io_kwargs"]
            
            # 合并所有模态的 kwargs
            for modality, kwargs in request_kwargs.items():
                if modality in current_kwargs:
                    current_kwargs[modality].update(kwargs)
                else:
                    current_kwargs[modality] = kwargs
            
            # 创建新的 chat_params
            chat_params = ChatParams(
                chat_template=chat_params.chat_template,
                chat_template_content_format=chat_params.chat_template_content_format,
                chat_template_kwargs=chat_params.chat_template_kwargs,
                media_io_kwargs=current_kwargs,
                mm_processor_kwargs=chat_params.mm_processor_kwargs,
            )
        
        logger.debug(f"[DEBUG] ChatParams: media_io_kwargs={chat_params.media_io_kwargs}")
        logger.debug(f"[DEBUG] ChatParams: mm_processor_kwargs={chat_params.mm_processor_kwargs}")
        
        # 构建 prompt_extras（关键！）
        prompt_extras = {}
        if request.get("mm_processor_kwargs"):
            prompt_extras["mm_processor_kwargs"] = request["mm_processor_kwargs"]
        
        # 调用 render_chat_async（完整的参数）
        (conversation,), (engine_prompt,) = await renderer.render_chat_async(
            [messages],
            chat_params,
            tok_params,  # 添加 tok_params
            prompt_extras=prompt_extras if prompt_extras else None,  # 添加 prompt_extras
        )
        
        logger.debug(f"[DEBUG] render_chat_async done")
        logger.debug(f"[DEBUG] engine_prompt keys={list(engine_prompt.keys())}")
        logger.debug(f"[DEBUG] engine_prompt type={engine_prompt.get('type')}")

        if "multi_modal_data" in engine_prompt:
            mm_data = engine_prompt["multi_modal_data"]
            logger.debug(f"[DEBUG] ✅ multi_modal_data keys={list(mm_data.keys()) if mm_data else None}")
        else:
            logger.debug(f"[DEBUG] ❌ multi_modal_data NOT in engine_prompt (checking mm_kwargs instead)")

        # Check for processed multimodal data (MultiModalInputs)
        if "mm_kwargs" in engine_prompt:
            mm_kwargs = engine_prompt["mm_kwargs"]
            logger.debug(f"[DEBUG] ✅ mm_kwargs keys={list(mm_kwargs.keys()) if mm_kwargs else None}")
            for k, v in (mm_kwargs.items() if mm_kwargs else []):
                if hasattr(v, 'shape'):
                    logger.debug(f"[DEBUG]   mm_kwargs[{k}].shape={v.shape}")
                elif isinstance(v, list) and len(v) > 0:
                    logger.debug(f"[DEBUG]   mm_kwargs[{k}] is list, len={len(v)}, first_item_type={type(v[0]).__name__}")
                else:
                    logger.debug(f"[DEBUG]   mm_kwargs[{k}]={type(v).__name__}")

        if "mm_hashes" in engine_prompt:
            mm_hashes = engine_prompt["mm_hashes"]
            logger.debug(f"[DEBUG] mm_hashes keys={list(mm_hashes.keys()) if mm_hashes else None}")

        if "mm_placeholders" in engine_prompt:
            mm_placeholders = engine_prompt["mm_placeholders"]
            logger.debug(f"[DEBUG] mm_placeholders keys={list(mm_placeholders.keys()) if mm_placeholders else None}")

        if "prompt_token_ids" in engine_prompt:
            logger.debug(f"[DEBUG] prompt_token_ids length={len(engine_prompt['prompt_token_ids'])}")
        
        return [engine_prompt]