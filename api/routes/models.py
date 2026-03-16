"""模型列表路由

提供模型列表端点。
"""

from fastapi import APIRouter

from ..schemas import ModelList, ModelInfo

router = APIRouter()


@router.get("/v1/models")
async def list_models() -> ModelList:
    """
    获取可用模型列表
    
    Returns:
        ModelList: 模型列表
    """
    return ModelList(
        data=[
            ModelInfo(
                id="qwen-3.5",
                object="model",
                created=1700000000,
                owned_by="vllm",
            )
        ]
    )
