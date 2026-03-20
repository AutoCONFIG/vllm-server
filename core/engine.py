"""Engine管理器模块

管理vLLM AsyncLLMEngine的单例实例。
"""

import asyncio
from typing import Optional

from vllm import AsyncLLMEngine


class EngineNotInitializedError(Exception):
    """Engine未初始化异常"""
    pass


class EngineManager:
    """Engine单例管理器"""
    
    _instance: Optional["EngineManager"] = None
    _engine: Optional[AsyncLLMEngine] = None
    _config: Optional[object] = None
    
    def __new__(cls):
        """确保单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def initialize(
        self,
        config: object,
        engine_args: object,
    ) -> None:
        """
        初始化Engine
        
        Args:
            config: 配置对象
            engine_args: AsyncEngineArgs对象
            
        Raises:
            RuntimeError: 如果Engine已经初始化
        """
        if self._engine is not None:
            print("[WARN] Engine already initialized, skipping")
            return
        
        from .engine_args import print_engine_config
        
        # 打印配置信息
        print_engine_config(config)
        
        # 创建Engine
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._config = config
        
        print("[INFO] Waiting for model to load...")
        # 等待EngineCore准备就绪
        await asyncio.sleep(2)
        print("[INFO] Engine initialized")
    
    @property
    def engine(self) -> AsyncLLMEngine:
        """
        获取Engine实例
        
        Returns:
            AsyncLLMEngine: Engine实例
            
        Raises:
            EngineNotInitializedError: 如果Engine未初始化
        """
        if self._engine is None:
            raise EngineNotInitializedError("Engine未初始化")
        return self._engine
    
    @property
    def config(self) -> object:
        """
        获取配置对象
        
        Returns:
            配置对象
        """
        return self._config
    
    async def shutdown(self) -> None:
        """
        关闭Engine
        """
        if self._engine is not None:
            print("[INFO] Shutting down engine...")
            await self._engine.shutdown()
            self._engine = None
            self._config = None
            print("[INFO] Engine shutdown complete")
    
    def is_initialized(self) -> bool:
        """
        检查Engine是否已初始化
        
        Returns:
            bool: 是否已初始化
        """
        return self._engine is not None


# 全局Engine管理器实例
engine_manager = EngineManager()
