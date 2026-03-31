#!/bin/bash
# vLLM Server 启动脚本

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 设置PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR/vllm-backend:$SCRIPT_DIR:$PYTHONPATH"

echo "PYTHONPATH: $PYTHONPATH"

# 启动服务器
cd "$SCRIPT_DIR"
python server.py --config "$SCRIPT_DIR/config.yaml" "$@"