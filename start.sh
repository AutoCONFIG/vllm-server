#!/bin/bash
# vLLM Server 启动脚本

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 创建符号链接 vllm -> vllm-backend (vllm-backend内部使用from vllm import)
if [ ! -L "$SCRIPT_DIR/vllm" ] && [ -d "$SCRIPT_DIR/vllm-backend" ]; then
    ln -s "$SCRIPT_DIR/vllm-backend" "$SCRIPT_DIR/vllm"
fi

# 设置PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "PYTHONPATH: $PYTHONPATH"

# 启动服务器
cd "$SCRIPT_DIR"
python server.py --config "$SCRIPT_DIR/config.yaml" "$@"