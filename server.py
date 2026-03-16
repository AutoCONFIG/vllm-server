#!/usr/bin/env python3
"""
vLLM Engine 服务器 - 模块化版本

这是一个按照分层架构重构的vLLM服务器，
支持多卡并行、多模态输入和完整的API功能。
"""

import argparse
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args)
    
    # 配置日志
    from utils import setup_logging
    setup_logging(
        config.logging.log_dir,
        config.logging.server_log_file,
        config.logging.level,
    )
    
    # 创建并启动应用
    run_server(config)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="vLLM Engine Server - 模块化版本，支持多卡并行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
多卡并行示例:
  # 使用4张卡进行张量并行
  python server.py --tensor-parallel-size 4
  
  # 使用张量并行+流水线并行 (例如4个GPU/节点，2个节点)
  python server.py --tensor-parallel-size 4 --pipeline-parallel-size 2
  
  # 使用数据并行 (需要Ray后端)
  python server.py --data-parallel-size 4 --distributed-executor-backend ray
  
多模态配置示例:
  # 限制最多3张图片，不允许视频
  python server.py --limit-mm-per-prompt '{"image": 3, "video": 0}'
  
  # 限制最多1个视频和2张图片
  python server.py --limit-mm-per-prompt '{"video": 1, "image": 2}'
  
配置管理:
  # 使用YAML配置文件
  python server.py --config config.yaml
  
  # 通过环境变量覆盖配置
  VLLM_SERVER_PORT=8000 python server.py
        """
    )
    
    # 服务器配置
    parser.add_argument("--host", type=str, default=None,
                        help="服务器主机 (环境变量: VLLM_SERVER_HOST)")
    parser.add_argument("--port", "-p", type=int, default=None,
                        help="服务器端口 (环境变量: VLLM_SERVER_PORT)")
    
    # 模型配置
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="模型路径 (环境变量: VLLM_MODEL_PATH)")
    
    # Engine配置
    parser.add_argument("--gpu-util", "-g", type=float, default=None,
                        help="GPU显存利用率 (环境变量: VLLM_ENGINE_GPU_MEMORY_UTILIZATION)")
    parser.add_argument("--max-len", "-l", type=int, default=None,
                        help="最大上下文长度 (环境变量: VLLM_ENGINE_MAX_MODEL_LEN)")
    parser.add_argument("--max-seqs", type=int, default=None,
                        help="最大并发序列数 (环境变量: VLLM_ENGINE_MAX_NUM_SEQS)")
    
    # 多卡并行配置
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=None,
                        help="张量并行大小 (环境变量: VLLM_ENGINE_TENSOR_PARALLEL_SIZE)")
    parser.add_argument("--pipeline-parallel-size", "-pp", type=int, default=None,
                        help="流水线并行大小 (环境变量: VLLM_ENGINE_PIPELINE_PARALLEL_SIZE)")
    parser.add_argument("--data-parallel-size", "-dp", type=int, default=None,
                        help="数据并行大小 (环境变量: VLLM_ENGINE_DATA_PARALLEL_SIZE)")
    parser.add_argument("--distributed-executor-backend", type=str, default=None,
                        choices=["mp", "ray"],
                        help="分布式执行后端 (环境变量: VLLM_ENGINE_DISTRIBUTED_EXECUTOR_BACKEND)")
    
    # 多模态配置
    parser.add_argument("--limit-mm-per-prompt", type=str, default=None,
                        help="限制每个提示的多模态数量，JSON格式")
    
    # 日志配置
    parser.add_argument("--log-requests", type=str, default=None,
                        help="记录所有API请求 (true/false)")
    parser.add_argument("--log-level", type=str, default=None,
                        help="日志级别 (DEBUG/INFO/WARNING/ERROR)")
    
    # 配置文件
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="YAML配置文件路径")
    
    return parser.parse_args()


def load_config(args: argparse.Namespace):
    """
    加载配置
    
    优先级：命令行参数 > 环境变量 > YAML配置 > 默认值
    
    Args:
        args: 命令行参数
        
    Returns:
        Settings: 加载的配置对象
    """
    from config import load_config
    from utils import validate_gpu_config
    
    # 加载基础配置
    settings = load_config(
        yaml_path=args.config,
        fallback_yaml_path="config.yaml",
    )
    
    # 命令行参数覆盖配置
    if args.host is not None and os.getenv("VLLM_SERVER_HOST"):
        settings.server.host = os.getenv("VLLM_SERVER_HOST")
    elif args.host is not None:
        settings.server.host = args.host
    
    if args.port is None and os.getenv("VLLM_SERVER_PORT"):
        settings.server.port = int(os.getenv("VLLM_SERVER_PORT"))
    elif args.port is not None:
        settings.server.port = args.port
    
    if args.model is None and os.getenv("VLLM_MODEL_PATH"):
        settings.model.path = os.getenv("VLLM_MODEL_PATH")
    elif args.model is not None:
        settings.model.path = args.model
    
    # Engine配置覆盖
    if args.gpu_util is not None and os.getenv("VLLM_ENGINE_GPU_MEMORY_UTILIZATION"):
        settings.engine.gpu_memory_utilization = float(os.getenv("VLLM_ENGINE_GPU_MEMORY_UTILIZATION"))
    elif args.gpu_util is not None:
        settings.engine.gpu_memory_utilization = args.gpu_util
    
    if args.max_len is not None and os.getenv("VLLM_ENGINE_MAX_MODEL_LEN"):
        settings.engine.max_model_len = int(os.getenv("VLLM_ENGINE_MAX_MODEL_LEN"))
    elif args.max_len is not None:
        settings.engine.max_model_len = args.max_len
    
    if args.max_seqs is not None and os.getenv("VLLM_ENGINE_MAX_NUM_SEQS"):
        settings.engine.max_num_seqs = int(os.getenv("VLLM_ENGINE_MAX_NUM_SEQS"))
    elif args.max_seqs is not None:
        settings.engine.max_num_seqs = args.max_seqs
    
    # 多卡并行配置覆盖
    if args.tensor_parallel_size is not None and os.getenv("VLLM_ENGINE_TENSOR_PARALLEL_SIZE"):
        settings.engine.tensor_parallel_size = int(os.getenv("VLLM_ENGINE_TENSOR_PARALLEL_SIZE"))
    elif args.tensor_parallel_size is not None:
        settings.engine.tensor_parallel_size = args.tensor_parallel_size
    
    if args.pipeline_parallel_size is not None and os.getenv("VLLM_ENGINE_PIPELINE_PARALLEL_SIZE"):
        settings.engine.pipeline_parallel_size = int(os.getenv("VLLM_ENGINE_PIPELINE_PARALLEL_SIZE"))
    elif args.pipeline_parallel_size is not None:
        settings.engine.pipeline_parallel_size = args.pipeline_parallel_size
    
    if args.data_parallel_size is not None and os.getenv("VLLM_ENGINE_DATA_PARALLEL_SIZE"):
        settings.engine.data_parallel_size = int(os.getenv("VLLM_ENGINE_DATA_PARALLEL_SIZE"))
    elif args.data_parallel_size is not None:
        settings.engine.data_parallel_size = args.data_parallel_size
    
    if args.distributed_executor_backend is not None and os.getenv("VLLM_ENGINE_DISTRIBUTED_EXECUTOR_BACKEND"):
        settings.engine.distributed_executor_backend = os.getenv("VLLM_ENGINE_DISTRIBUTED_EXECUTOR_BACKEND")
    elif args.distributed_executor_backend is not None:
        settings.engine.distributed_executor_backend = args.distributed_executor_backend
    
    # 多模态配置
    if args.limit_mm_per_prompt is not None:
        import json
        try:
            settings.multimodal.limit_mm_per_prompt = json.loads(args.limit_mm_per_prompt)
        except json.JSONDecodeError as e:
            print(f"❌ 错误: --limit-mm-per-prompt 必须是有效的JSON格式")
            print(f"   解析错误: {e}")
            print(f"   示例: '{{\"image\": 3, \"video\": 1}}'")
            sys.exit(1)
    
    # 日志配置
    if args.log_requests is not None and os.getenv("VLLM_LOGGING_LOG_REQUESTS"):
        settings.logging.log_requests = os.getenv("VLLM_LOGGING_LOG_REQUESTS").lower() == "true"
    elif args.log_requests is not None:
        settings.logging.log_requests = args.log_requests.lower() == "true" if args.log_requests else False
    
    if args.log_level is not None and os.getenv("VLLM_LOGGING_LEVEL"):
        settings.logging.level = os.getenv("VLLM_LOGGING_LEVEL")
    elif args.log_level is not None:
        settings.logging.level = args.log_level
    
    # 验证配置
    try:
        from utils import validate_gpu_utilization, validate_max_model_len, validate_model_path
        settings.model.path = validate_model_path(settings.model.path)
        settings.engine.gpu_memory_utilization = validate_gpu_utilization(settings.engine.gpu_memory_utilization)
        settings.engine.max_model_len = validate_max_model_len(settings.engine.max_model_len)
        
        # 自动设置 tensor_parallel_size（根据gpu_ids数量）
        if settings.engine.gpu_ids and settings.engine.tensor_parallel_size is None:
            gpu_count = len(settings.engine.gpu_ids.split(","))
            if gpu_count > 1:
                settings.engine.tensor_parallel_size = gpu_count
        
        validated_gpu = validate_gpu_config(
            settings.engine.tensor_parallel_size,
            settings.engine.pipeline_parallel_size,
            settings.engine.data_parallel_size,
            settings.engine.distributed_executor_backend,
        )
        settings.engine.tensor_parallel_size = validated_gpu["tensor_parallel_size"]
        settings.engine.pipeline_parallel_size = validated_gpu["pipeline_parallel_size"]
        settings.engine.data_parallel_size = validated_gpu["data_parallel_size"]
        settings.engine.distributed_executor_backend = validated_gpu["distributed_executor_backend"]
    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        sys.exit(1)
    
    return settings


def run_server(config):
    """
    运行服务器
    
    Args:
        config: 配置对象
    """
    import uvicorn
    from api.app import create_app
    
    # 设置GPU设备
    gpu_ids = config.engine.gpu_ids
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        print(f"🎮 使用GPU: {gpu_ids}")
        
        # 自动设置tensor_parallel_size（如果未手动设置且配置了多个GPU）
        gpu_count = len(gpu_ids.split(","))
        if config.engine.tensor_parallel_size is None and gpu_count > 1:
            config.engine.tensor_parallel_size = gpu_count
            print(f"⚡ 自动设置 tensor_parallel_size={gpu_count}")
    else:
        print("🎮 使用默认GPU (CUDA_VISIBLE_DEVICES)")
    
    # 创建应用
    app = create_app(config)
    
    # 打印启动信息
    print(f"🚀 启动vLLM Engine Server...")
    print(f"📁 模型: {config.model.path}")
    print(f"🎯 地址: http://{config.server.host}:{config.server.port}")
    
    # 启动服务器
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.logging.level.lower(),
        log_config=None,
        access_log=True,
    )


if __name__ == "__main__":
    main()
