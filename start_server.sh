#!/bin/bash
# =============================================================================
# vLLM Engine Server 启动脚本
# =============================================================================

# 设置环境变量
# 设置OMP线程数以避免CPU竞争和警告
export OMP_NUM_THREADS=1

# 默认配置
MODEL_PATH="/data2/kaiyun/qwen/model"
HOST="0.0.0.0"
PORT=8000
TENSOR_PARALLEL_SIZE=""  # 留空表示根据CUDA_VISIBLE_DEVICES自动设置
PIPELINE_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=1
DISTRIBUTED_EXECUTOR_BACKEND="mp"
CUDA_VISIBLE_DEVICES="2,6"  # 留空表示使用所有可用GPU，例如 "2,6,7" 表示只使用2号、6号、7号卡
LOG_REQUESTS="true"  # 是否记录API请求日志：true 或 false
# 前缀缓存配置：true 禁用，false 启用（禁用可避免Mamba警告）
DISABLE_PREFIX_CACHING="false"

# 多模态输入限制配置（JSON格式）
# 示例: '{"image": 4}' - 限制每个提示最多4张图片
# 示例: '{"image": 4, "video": 1}' - 限制每个提示最多4张图片和1个视频
LIMIT_MM_PER_PROMPT='{"image": 6, "video": 2}'

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  vLLM Engine Server 启动脚本${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查Python环境
echo -e "${YELLOW}▶ 检查Python环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3"
    exit 1
fi

# 检查依赖
echo -e "${YELLOW}▶ 检查依赖...${NC}"
python3 -c "import vllm, fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: 缺少依赖，请安装: pip install fastapi uvicorn pydantic"
    exit 1
fi

# 检查模型路径
echo -e "${YELLOW}▶ 检查模型路径...${NC}"
if [ ! -d "$MODEL_PATH" ]; then
    echo "警告: 模型路径不存在: $MODEL_PATH"
    echo -n "请输入正确的模型路径 (直接回车使用默认): "
    read NEW_PATH
    if [ ! -z "$NEW_PATH" ]; then
        MODEL_PATH="$NEW_PATH"
    fi
fi

echo ""

# 检查GPU数量
echo -e "${YELLOW}▶ 检查GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    TOTAL_GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo -e "${BLUE}系统总GPU数: ${TOTAL_GPU_COUNT} 张${NC}"
    
    # 如果指定了CUDA_VISIBLE_DEVICES
    if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
        # 计算可见GPU数量
        IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
        GPU_COUNT=${#GPUS[@]}
        echo -e "${BLUE}使用GPU: ${CUDA_VISIBLE_DEVICES}${NC}"
        echo -e "${BLUE}可见GPU数: ${GPU_COUNT} 张${NC}"
        
        # 如果TENSOR_PARALLEL_SIZE为空，自动设置为GPU数量
        if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
            TENSOR_PARALLEL_SIZE=$GPU_COUNT
            echo -e "${YELLOW}自动设置张量并行大小: ${TENSOR_PARALLEL_SIZE}${NC}"
        fi
        
        # 导出环境变量
        export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    else
        GPU_COUNT=$TOTAL_GPU_COUNT
        echo -e "${BLUE}使用所有可用GPU${NC}"
        
        # 如果TENSOR_PARALLEL_SIZE为空且使用所有GPU
        if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
            TENSOR_PARALLEL_SIZE=$GPU_COUNT
            echo -e "${YELLOW}自动设置张量并行大小: ${TENSOR_PARALLEL_SIZE}${NC}"
        fi
    fi
else
    echo "警告: 未找到nvidia-smi"
    GPU_COUNT=1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  配置信息${NC}"
echo -e "${GREEN}========================================${NC}"
echo "  模型路径: $MODEL_PATH"
echo "  服务地址: http://$HOST:$PORT"

# 多卡配置
TOTAL_CONFIG_GPUS=$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE))

# 当使用多卡或配置了并行策略时显示多卡信息
if [ $GPU_COUNT -gt 1 ] || [ $TOTAL_CONFIG_GPUS -gt 1 ]; then
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  多卡并行配置${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}张量并行(TP): ${TENSOR_PARALLEL_SIZE}${NC}"
    echo -e "${BLUE}流水线并行(PP): ${PIPELINE_PARALLEL_SIZE}${NC}"
    echo -e "${BLUE}数据并行(DP): ${DATA_PARALLEL_SIZE}${NC}"
    echo -e "${BLUE}配置总GPU数: ${TOTAL_CONFIG_GPUS}${NC}"
    echo -e "${BLUE}可见GPU数: ${GPU_COUNT}${NC}"
    echo -e "${BLUE}分布式后端: ${DISTRIBUTED_EXECUTOR_BACKEND}${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ $TOTAL_CONFIG_GPUS -gt $GPU_COUNT ]; then
        echo -e "${YELLOW}⚠️  警告: 配置的GPU数(${TOTAL_CONFIG_GPUS})超过可见GPU数(${GPU_COUNT})${NC}"
        echo -e "${YELLOW}⚠️  请检查CUDA_VISIBLE_DEVICES设置或调整并行配置${NC}"
    fi
fi
echo ""
echo -e "${BLUE}按 Ctrl+C 停止服务${NC}"
echo ""

# 启动服务
python3 vllm_engine_server.py \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" \
    --data-parallel-size "$DATA_PARALLEL_SIZE" \
    --distributed-executor-backend "$DISTRIBUTED_EXECUTOR_BACKEND" \
    --gpu-util 0.95 \
    --max-len 262144 \
    --max-seqs 64 \
    --log-requests "$LOG_REQUESTS" \
    --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
    --disable-prefix-caching "$DISABLE_PREFIX_CACHING"
