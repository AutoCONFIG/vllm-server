"""配置加载器

提供从YAML文件和环境变量加载配置的功能。
"""

import os
from typing import Optional, Dict, Any

from .settings import Settings


def load_config(
    yaml_path: Optional[str] = None,
    fallback_yaml_path: str = "config.yaml",
) -> Settings:
    """
    加载配置
    
    优先级顺序：
    1. 命令行指定的YAML文件
    2. 默认位置的YAML文件
    3. 环境变量
    4. 默认值
    
    Args:
        yaml_path: 显式指定的YAML配置文件路径
        fallback_yaml_path: 默认YAML配置文件路径
        
    Returns:
        Settings: 加载的配置对象
        
    Raises:
        FileNotFoundError: 当指定的YAML文件不存在时
        ValueError: 当YAML文件格式错误时
    """
    config_data = {}
    yaml_loaded = False
    
    # 尝试加载YAML文件
    try:
        config_path = _resolve_yaml_path(yaml_path, fallback_yaml_path)
        if config_path:
            config_data = _load_yaml_file(config_path)
            yaml_loaded = True
            print(f"📄 从配置文件加载: {config_path}")
    except FileNotFoundError:
        if yaml_path:
            # 如果显式指定了YAML文件但不存在，则报错
            raise
        # 否则继续使用环境变量和默认值
        print("[WARN] Config file not found, using env vars and defaults")
    
    # 创建Settings对象（会自动处理环境变量）
    if yaml_loaded:
        settings = Settings(**config_data)
    else:
        settings = Settings()
    
    return settings


def _resolve_yaml_path(
    yaml_path: Optional[str],
    fallback_yaml_path: str,
) -> Optional[str]:
    """
    解析YAML配置文件路径
    
    Args:
        yaml_path: 显式指定的YAML路径
        fallback_yaml_path: 回退路径
        
    Returns:
        Optional[str]: 解析后的绝对路径，如果找不到则返回None
    """
    # 优先使用显式指定的路径
    if yaml_path:
        if os.path.exists(yaml_path):
            return os.path.abspath(yaml_path)
        else:
            raise FileNotFoundError(f"指定的配置文件不存在: {yaml_path}")
    
    # 检查回退路径
    if os.path.exists(fallback_yaml_path):
        return os.path.abspath(fallback_yaml_path)
    
    # 检查常见的配置目录
    possible_locations = [
        os.path.join(os.path.dirname(__file__), fallback_yaml_path),  # config/目录下
        os.path.join(os.getcwd(), fallback_yaml_path),  # 当前工作目录
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            return os.path.abspath(location)
    
    return None


def _load_yaml_file(yaml_path: str) -> Dict[str, Any]:
    """
    加载YAML文件
    
    Args:
        yaml_path: YAML文件路径
        
    Returns:
        Dict[str, Any]: 加载的配置字典
        
    Raises:
        ValueError: YAML格式错误时
    """
    try:
        import yaml
    except ImportError:
        print("[WARN] PyYAML not installed, trying to install: pip install pyyaml")
        raise ImportError("PyYAML required: pip install pyyaml")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"YAML格式错误: {e}")


def ensure_log_directory(settings: Settings) -> None:
    """
    确保日志目录存在
    
    Args:
        settings: 配置对象
    """
    log_dir = settings.logging.log_dir
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        print(f"[INFO] Created log directory: {log_dir}")
