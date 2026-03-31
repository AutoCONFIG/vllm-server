"""日志工具模块

提供日志配置和请求记录功能。
"""

import json
import logging
import os
from datetime import datetime
from threading import Lock
from typing import Optional, Dict, Any


# 全局日志锁，确保线程安全
LOG_LOCK = Lock()


def setup_logging(
    log_dir: str,
    server_log_file: str = "server.log",
    log_level: str = "INFO",
) -> None:
    """
    配置服务器日志输出到文件和控制台
    
    Args:
        log_dir: 日志目录路径
        server_log_file: 服务器日志文件名
        log_level: 日志级别
    """
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    log_file_path = os.path.join(log_dir, server_log_file)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # 配置 uvicorn 的日志
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.addHandler(console_handler)
    uvicorn_logger.addHandler(file_handler)
    uvicorn_logger.setLevel(log_level)
    
    # 配置 uvicorn.access 的日志
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.addHandler(console_handler)
    access_logger.addHandler(file_handler)
    access_logger.setLevel(log_level)
    
    print(f"[INFO] Logging configured: {log_file_path}")


def _truncate_media_data(data: Any, max_len: int = 30) -> Any:
    """
    截断媒体数据（图片/视频的base64或URL），只保留前后一小段
    
    Args:
        data: 要处理的数据
        max_len: 截断后的最大长度（前后各保留的字符数）
    
    Returns:
        处理后的数据
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            result[key] = _truncate_media_data(value, max_len)
        return result
    elif isinstance(data, list):
        return [_truncate_media_data(item, max_len) for item in data]
    elif isinstance(data, str):
        lower_key_hints = ['image', 'video', 'audio', 'url']
        if any(hint in str(data).lower() for hint in lower_key_hints) and len(data) > max_len * 2:
            if data.startswith('data:') or len(data) > 1000:
                return f"{data[:max_len]}...<truncated {len(data)} chars>...{data[-max_len:]}"
        return data
    else:
        return data


def log_request(
    log_dir: str,
    request_log_file: str = "api_requests.log",
    request_data: Optional[Dict[str, Any]] = None,
    response_data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    将API请求数据记录到日志文件
    
    Args:
        log_dir: 日志目录路径
        request_log_file: 请求日志文件名
        request_data: 请求数据字典
        response_data: 响应数据字典
    """
    try:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "request": _truncate_media_data(request_data) if request_data else None,
            "response": _truncate_media_data(response_data) if response_data else None,
        }
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        log_file_path = os.path.join(log_dir, request_log_file)
        
        with LOG_LOCK:
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] Log write failed: {e}")


def get_logger(name: str) -> logging.Logger:
    """
    获取配置好的日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        logging.Logger: 日志记录器实例
    """
    return logging.getLogger(name)
