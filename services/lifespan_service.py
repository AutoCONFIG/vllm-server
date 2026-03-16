"""应用生命周期服务

管理应用的启动和关闭逻辑。
"""

from contextlib import asynccontextmanager

from core import engine_manager, build_engine_args
from config import Settings


@asynccontextmanager
async def lifespan(app, config: Settings):
    """
    应用生命周期管理
    
    Args:
        app: FastAPI应用
        config: 配置对象
    """
    # 启动时
    engine_args = build_engine_args(config)
    await engine_manager.initialize(config, engine_args)
    
    yield
    
    # 关闭时
    await engine_manager.shutdown()
