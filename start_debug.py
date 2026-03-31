#!/usr/bin/env python3
"""
启动脚本 - 启用 DEBUG 日志
"""

import os
import sys

# 设置日志级别
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

# 启动服务器
from server import main

if __name__ == "__main__":
    main()
