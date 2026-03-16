"""数据验证工具

提供配置和输入数据的验证功能。
"""

import os
from typing import Dict, Any


def validate_gpu_config(
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    distributed_executor_backend: str,
) -> Dict[str, Any]:
    """
    验证GPU并行配置
    
    Args:
        tensor_parallel_size: 张量并行大小
        pipeline_parallel_size: 流水线并行大小
        data_parallel_size: 数据并行大小
        distributed_executor_backend: 分布式执行后端
        
    Returns:
        Dict[str, Any]: 验证后的配置（可能修正）
        
    Raises:
        ValueError: 配置无效时
    """
    # 验证并行大小为正整数
    if tensor_parallel_size < 1:
        raise ValueError(f"tensor_parallel_size 必须 >= 1, 当前: {tensor_parallel_size}")
    if pipeline_parallel_size < 1:
        raise ValueError(f"pipeline_parallel_size 必须 >= 1, 当前: {pipeline_parallel_size}")
    if data_parallel_size < 1:
        raise ValueError(f"data_parallel_size 必须 >= 1, 当前: {data_parallel_size}")
    
    # 检查数据并行是否需要Ray后端
    total_gpus = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
    if total_gpus > 1 and data_parallel_size > 1 and distributed_executor_backend != "ray":
        print("⚠️  警告: 数据并行需要使用 ray 后端，自动切换")
        distributed_executor_backend = "ray"
    
    # 检查CUDA_VISIBLE_DEVICES
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_devices:
        device_count = len(cuda_devices.split(","))
        total_required = tensor_parallel_size * pipeline_parallel_size
        if device_count < total_required:
            print(f"⚠️  警告: CUDA_VISIBLE_DEVICES 指定了 {device_count} 张GPU, "
                  f"但配置需要 {total_required} 张GPU")
    
    return {
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "data_parallel_size": data_parallel_size,
        "distributed_executor_backend": distributed_executor_backend,
    }


def validate_model_path(model_path: str) -> str:
    """
    验证模型路径
    
    Args:
        model_path: 模型路径
        
    Returns:
        str: 验证后的路径
        
    Raises:
        ValueError: 路径无效时
    """
    if not model_path:
        raise ValueError("model_path 不能为空")
    
    # 检查路径是否存在
    if not os.path.exists(model_path):
        print(f"⚠️  警告: 模型路径不存在: {model_path}")
        print(f"   请确保模型已正确下载或挂载")
    
    return os.path.abspath(model_path)


def validate_gpu_utilization(gpu_util: float) -> float:
    """
    验证GPU显存利用率
    
    Args:
        gpu_util: 显存利用率 (0-1)
        
    Returns:
        float: 验证后的值
        
    Raises:
        ValueError: 值超出范围时
    """
    if not (0.0 < gpu_util <= 1.0):
        raise ValueError(f"gpu_memory_utilization 必须在 (0, 1] 范围内, 当前: {gpu_util}")
    return gpu_util


def validate_max_model_len(max_len: int) -> int:
    """
    验证最大模型长度
    
    Args:
        max_len: 最大长度
        
    Returns:
        int: 验证后的值
        
    Raises:
        ValueError: 值无效时
    """
    if max_len < 1:
        raise ValueError(f"max_model_len 必须 >= 1, 当前: {max_len}")
    if max_len > 100000:
        print(f"⚠️  警告: max_model_len 设置为 {max_len}, 这可能需要大量显存")
    return max_len
