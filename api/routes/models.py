"""模型列表路由

提供模型列表端点。
"""

from fastapi import APIRouter, Request

from ..schemas import ModelList, ModelInfo

router = APIRouter()


@router.get("/v1/models")
async def list_models(request: Request) -> ModelList:
    """
    获取可用模型列表
    
    Returns:
        ModelList: 模型列表
    """
    config = request.app.state.config
    
    # 获取模型名称，优先使用配置值，否则从model.path提取
    model_name = config.model.name
    if not model_name and config.model.path:
        # 从路径中提取模型名称
        model_name = config.model.path.rstrip("/").split("/")[-1]
    
    return ModelList(
        data=[
            ModelInfo(
                id=model_name,
                object="model",
                created=1700000000,
                owned_by="vllm",
            )
        ]
    )
