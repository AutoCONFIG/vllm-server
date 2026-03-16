"""聊天请求和响应模型

定义OpenAI兼容的聊天API模型。
"""

from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """聊天消息"""
    role: str
    content: Any


class Usage(BaseModel):
    """Token使用统计"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatChoice(BaseModel):
    """聊天选择"""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage


class ChatRequest(BaseModel):
    """聊天请求"""
    model: str = Field(default="qwen-3.5")
    messages: List[Dict[str, Any]]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = Field(default=False)
    
    class Config:
        # 允许任意字典类型（用于多模态内容）
        extra = "allow"
