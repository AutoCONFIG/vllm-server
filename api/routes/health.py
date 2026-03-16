"""健康检查路由

提供健康检查端点。
"""

from fastapi import APIRouter

from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health")
async def get_health() -> HealthResponse:
    """
    健康检查端点
    
    Returns:
        HealthResponse: 健康状态
    """
    return HealthResponse()
