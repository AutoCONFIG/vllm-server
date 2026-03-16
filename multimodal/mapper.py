"""消息到多模态的映射

将OpenAI兼容的消息格式转换为多模态提示和数据。
"""

from typing import List, Tuple, Optional, Dict, Any


def messages_to_multimodal_prompt(
    messages: List[Dict[str, Any]],
) -> Tuple[str, Optional[Dict[str, List[Any]]]]:
    """
    将消息列表转换为提示字符串，并提取多模态数据。
    
    Args:
        messages: OpenAI兼容的消息列表，每个消息包含role和content
        
    Returns:
        Tuple[str, Optional[Dict]]: 
        - prompt: 转换后的提示字符串（包含图像/视频占位符）
        - multi_modal_data: 包含图像/视频URL的字典，如果没有则为None
    """
    prompt_parts = []
    multi_modal_data = {"image": [], "video": []}  # 支持图片和视频
    
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
                elif item_type == "video_url":
                    # 提取视频URL或base64数据
                    video_url = item.get("video_url", {}).get("url", "")
                    if video_url:
                        # 添加视频占位符（Qwen模型使用 <|vision_start|><|video_pad|><|vision_end|>）
                        text_parts.append("<|vision_start|><|video_pad|><|vision_end|>")
                        # 添加到多模态数据列表
                        multi_modal_data["video"].append(video_url)
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
                        # 添加其他参数如 total_pixels, min_pixels, fps
                        for key in ["total_pixels", "min_pixels", "fps"]:
                            if key in item:
                                video_item[key] = item[key]
                        multi_modal_data["video"].append(video_item)
            # 合并所有文本部分
            content = "\n".join(text_parts) if text_parts else ""
        
        # 构建提示词格式
        if role == "system":
            prompt_parts.append(f"<|im_start|>system\n{content} <|end|>")
        elif role == "user":
            prompt_parts.append(f"<|im_start|>user\n{content} <|end|>")
        elif role == "assistant":
            prompt_parts.append(f"<|im_start|>assistant\n{content} <|end|>")
    
    # 添加助手回复起始
    prompt_parts.append("<|im_start|>assistant\n")
    
    # 如果没有多模 fancinal数据，返回None
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
