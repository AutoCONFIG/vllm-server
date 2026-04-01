#!/bin/bash
# vLLM Server 启动脚本

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 创建符号链接 vllm -> vllm-backend/vllm (vllm-backend内部使用from vllm import)
if [ -d "$SCRIPT_DIR/vllm-backend" ]; then
    rm -f "$SCRIPT_DIR/vllm"
    ln -s "$SCRIPT_DIR/vllm-backend/vllm" "$SCRIPT_DIR/vllm"
fi

# 设置PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# 设置 vLLM 日志级别为 DEBUG（用于调试多模态问题）
export VLLM_LOGGING_LEVEL="DEBUG"
export VLLM_SERVER_LOG_LEVEL="DEBUG"

echo "PYTHONPATH: $PYTHONPATH"
echo "VLLM_LOGGING_LEVEL: $VLLM_LOGGING_LEVEL"

# 启动服务器
cd "$SCRIPT_DIR"
python server.py --config "$SCRIPT_DIR/config.yaml" "$@"