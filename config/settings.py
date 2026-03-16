"""配置模型定义

使用Pydantic定义配置结构，提供类型验证和默认值。
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ServerConfig(BaseModel):
    """服务器配置"""
    host: Optional[str] = None
    port: Optional[int] = None
    title: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None


class ModelConfig(BaseModel):
    """模型配置"""
    path: Optional[str] = None
    dtype: Optional[str] = Field(
        default=None,
        description="数据类型: auto, float16, bfloat16, float32"
    )
    quantization: Optional[str] = Field(
        default=None,
        description="量化类型: fp8, int8, int4, awq, gptq, bitsandbytes 等"
    )
    kv_cache_dtype: Optional[str] = Field(
        default=None,
        description="KV Cache 数据类型: auto, fp8, fp8_e4m3, fp8_e5m2, bfloat16"
    )


class EngineConfig(BaseModel):
    """Engine配置"""
    gpu_memory_utilization: Optional[float] = None
    gpu_ids: Optional[str] = Field(
        default=None,
        description="GPU编号，如 '0' 或 '0,1,2,3'"
    )
    max_model_len: Optional[int] = None
    block_size: Optional[int] = Field(
        default=None,
        description="KV cache block大小，必须能被tensor_parallel_size整除"
    )
    max_num_seqs: Optional[int] = None
    tensor_parallel_size: Optional[int] = None
    pipeline_parallel_size: Optional[int] = None
    data_parallel_size: Optional[int] = None
    distributed_executor_backend: Optional[str] = None
    enable_chunked_prefill: Optional[bool] = None
    max_num_batched_tokens: Optional[int] = Field(
        default=None,
        description="分块预填充的最大batch token数，必须能被tensor_parallel_size整除"
    )
    enable_prefix_caching: Optional[bool] = None
    seed: Optional[int] = None
    disable_custom_all_reduce: Optional[bool] = None
    scheduling_policy: Optional[str] = None


class MultimodalConfig(BaseModel):
    """多模态配置"""
    limit_mm_per_prompt: Optional[Dict[str, int]] = Field(
        default=None,
        description="限制每个提示的多模态数量，如 {'image': 3, 'video': 1}"
    )


class LoggingConfig(BaseModel):
    """日志配置"""
    level: Optional[str] = None
    log_requests: Optional[bool] = None
    log_dir: Optional[str] = None
    request_log_file: Optional[str] = None
    server_log_file: Optional[str] = None


class Settings(BaseSettings):
    """全局配置类
    
    支持从环境变量和YAML文件加载配置。
    所有字段默认为 None，必须显式配置。
    
    环境变量优先级: 环境变量 > YAML文件 > 默认值(None)
    
    环境变量命名规则: VLLM_{SECTION}_{KEY}
    例如: VLLM_SERVER_PORT, VLLM_ENGINE_GPU_UTIL
    """
    
    server: ServerConfig = Field(default_factory=lambda: ServerConfig())
    model: ModelConfig = Field(default_factory=lambda: ModelConfig())
    engine: EngineConfig = Field(default_factory=lambda: EngineConfig())
    multimodal: MultimodalConfig = Field(default_factory=lambda: MultimodalConfig())
    logging: LoggingConfig = Field(default_factory=lambda: LoggingConfig())
    
    # 实验性功能
    disable_prefix_caching: Optional[bool] = None
    
    class Config:
        env_prefix = "vllm_"
        env_nested_delimiter = "_"
        
        @staticmethod
        def from_yaml(yaml_path: str) -> "Settings":
            """从YAML文件加载配置"""
            import yaml
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return Settings(**data)
    
    @classmethod
    def from_yaml_file(cls, yaml_path: str) -> "Settings":
        """从YAML文件加载配置（兼容旧API）"""
        return cls.Config.from_yaml(yaml_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "server": self.server.model_dump(),
            "model": self.model.model_dump(),
            "engine": self.engine.model_dump(),
            "multimodal": self.multimodal.model_dump(),
            "logging": self.logging.model_dump(),
            "disable_prefix_caching": self.disable_prefix_caching,
        }
