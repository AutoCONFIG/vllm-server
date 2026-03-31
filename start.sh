#!/bin/bash
# vLLM Server 启动脚本

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 确保vllm-backend以可编辑模式安装
if ! python -c "import vllm" 2>/dev/null; then
    echo "[INFO] Installing vllm-backend in editable mode..."
    VLLM_USE_PRECOMPILED=1 pip install -e "$SCRIPT_DIR/vllm-backend" --no-build-isolation -q
fi

# 设置PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "PYTHONPATH: $PYTHONPATH"

# 启动服务器
cd "$SCRIPT_DIR"
python server.py --config "$SCRIPT_DIR/config.yaml" "$@"