"""Engine参数构建模块

根据配置构建AsyncEngineArgs。
"""

from typing import Optional, Dict, Any

from vllm import AsyncEngineArgs


def build_engine_args(
    config: Any,
) -> AsyncEngineArgs:
    """
    根据配置构建对象
    
    Args:
        config: 配置对象（包含engine和model配置）
        
    Returns:
        AsyncEngineArgs: 构建的参数对象
    """
    engine_cfg = config.engine
    model_cfg = config.model
    
    args = {
        # 模型配置
        "model": model_cfg.path,
        "dtype": model_cfg.dtype,
        "quantization": model_cfg.quantization,
        "kv_cache_dtype": model_cfg.kv_cache_dtype,
        
        # 显存配置
        "gpu_memory_utilization": engine_cfg.gpu_memory_utilization,
        "max_model_len": engine_cfg.max_model_len,
        "block_size": engine_cfg.block_size,
        "max_num_seqs": engine_cfg.max_num_seqs,
        
        # 性能优化
        "enable_chunked_prefill": engine_cfg.enable_chunked_prefill,
        "max_num_batched_tokens": engine_cfg.max_num_batched_tokens,
        "enable_prefix_caching": engine_cfg.enable_prefix_caching and not config.disable_prefix_caching,
        
        # 调度配置
        "scheduling_policy": engine_cfg.scheduling_policy,
        "enforce_eager": False,
        "seed": engine_cfg.seed,
        
        # 多卡并行配置
        "tensor_parallel_size": engine_cfg.tensor_parallel_size,
        "pipeline_parallel_size": engine_cfg.pipeline_parallel_size,
        "distributed_executor_backend": engine_cfg.distributed_executor_backend,
        
        # 禁用不支持的优化以避免警告
        "disable_custom_all_reduce": engine_cfg.disable_custom_all_reduce,
        
        # 注意力后端配置
        "attention_backend": engine_cfg.attention_backend,
    }
    
    # 多模态配置
    if config.multimodal.limit_mm_per_prompt is not None:
        args["limit_mm_per_prompt"] = config.multimodal.limit_mm_per_prompt
    
    return AsyncEngineArgs(**args)


def print_engine_config(config: Any) -> None:
    """
    打印Engine配置信息
    
    Args:
        config: 配置对象
    """
    print("🚀 正在初始化vLLM AsyncLLMEngine...")
    print(f"📁 模型路径: {config.model.path}")
    print(f"🎯 GPU利用率: {config.engine.gpu_memory_utilization}")
    print(f"📝 最大长度: {config.engine.max_model_len}")
    
    # 多卡并行信息
    total_gpus = (
        config.engine.tensor_parallel_size *
        config.engine.pipeline_parallel_size *
        config.engine.data_parallel_size
    )
    if total_gpus > 1:
        print(f"🖥️  多卡配置: TP={config.engine.tensor_parallel_size}, "
              f"PP={config.engine.pipeline_parallel_size}, "
              f"DP={config.engine.data_parallel_size}")
        print(f"🖥️  总GPU数: {total_gpus}")
        print(f"🔧 分布式后端: {config.engine.distributed_executor_backend}")
    
    # 注意力后端信息
    if config.engine.attention_backend:
        print(f"⚡ 注意力后端: {config.engine.attention_backend}")
    
    # 多模态信息
    if config.multimodal.limit_mm_per_prompt is not None:
        print(f"🖼️  多模态限制: {config.multimodal.limit_mm_per_prompt}")
    
    # 前缀缓存配置信息
    if config.disable_prefix_caching:
        print(f"⚙️  前缀缓存已禁用（避免Mamba警告）")
