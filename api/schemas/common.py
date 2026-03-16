"""通用API模型

定义健康检查和模型列表响应模型。
"""

from typing import List
from pydantic import BaseModel


class ModelInfo(BaseModel):
    """模型信息"""
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    """模型列表响应"""
    object: str = "list"
    data: List[ModelInfo]


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "healthy"
    model: str
