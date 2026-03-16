"""FastAPI应用创建

创建和配置FastAPI应用实例。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    """
    # 从app.state获取config
    config = app.state.config
    
    # 启动时初始化Engine
    from core import engine_manager, build_engine_args
    print("🚀 正在初始化vLLM Engine...")
    engine_args = build_engine_args(config)
    await engine_manager.initialize(config, engine_args)
    
    yield
    
    # 关闭时清理
    await engine_manager.shutdown()


def create_app(config) -> FastAPI:
    """
    创建FastAPI应用
    """
    # 创建FastAPI实例（此时lifespan会被注册但不会立即执行）
    # 注意：lifespan函数需要访问app.state.config，所以先创建app对象
    app = FastAPI(
        title=config.server.title,
        description=config.server.description,
        version=config.server.version,
    )
    
    # 先设置全局状态（这样lifespan启动时才能访问到config）
    app.state.config = config
    
    # 设置lifespan（在设置config之后）
    app.router.lifespan_context = lifespan
    
    # 注册路由
    from api.routes import health, models, chat
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(chat.router)
    
    return app
