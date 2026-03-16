#!/bin/bash
# =============================================================================
# vLLM Engine Server Docker构建脚本
# =============================================================================
#
# 使用方法:
#   ./build.sh [选项]
#
# 选项:
#   -t, --tag          镜像标签 (默认: latest)
#   -v, --vllm         vLLM基础镜像 (默认: vllm/vllm-openai:latest)
#   --no-cache         不使用缓存
#   --push             构建后推送
#   -h, --help         显示帮助信息
#
# 示例:
#   ./build.sh                              # 使用默认配置
#   ./build.sh -t v1.0.0 -v vllm/vllm-openai:v0.11.0
#   ./build.sh --no-cache
# =============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 默认值
IMAGE_NAME="vllm-engine-server"
IMAGE_TAG="latest"
VLLM_IMAGE="vllm/vllm-openai:latest"
NO_CACHE=false
PUSH=false

# 帮助信息
show_help() {
    echo "vLLM Engine Server Docker构建脚本"
    echo ""
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -t, --tag          镜像标签 (默认: latest)"
    echo "  -v, --vllm         vLLM基础镜像 (默认: vllm/vllm-openai:latest)"
    echo "  --no-cache         不使用缓存"
    echo "  --push             构建后推送"
    echo "  -h, --help         显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0"
    echo "  $0 -t v1.0.0 -v vllm/vllm-openai:v0.11.0"
    echo "  $0 --no-cache"
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -v|--vllm)
            VLLM_IMAGE="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 打印配置
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  vLLM Engine Server Docker构建${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}构建配置:${NC}"
echo "  镜像: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  基础镜像: ${VLLM_IMAGE}"
echo "  不使用缓存: ${NO_CACHE}"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

# 构建参数
BUILD_ARGS="--build-arg VLLM_IMAGE=${VLLM_IMAGE}"
if [ "$NO_CACHE" = true ]; then
    BUILD_ARGS="${BUILD_ARGS} --no-cache"
fi

# 构建镜像
echo -e "${GREEN}开始构建...${NC}"
docker build \
    -f docker/Dockerfile.custom \
    -t ${IMAGE_NAME}:${IMAGE_TAG} \
    ${BUILD_ARGS} \
    .

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}构建成功!${NC}"
    echo ""
    echo "镜像: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "运行示例:"
    echo "  docker run --gpus all -p 8000:8000 \\"
    echo "    -v /path/to/model:/model \\"
    echo "    -e VLLM_MODEL_PATH=/model \\"
    echo "    ${IMAGE_NAME}:${IMAGE_TAG}"

    if [ "$PUSH" = true ]; then
        echo ""
        echo -e "${YELLOW}推送镜像...${NC}"
        docker push ${IMAGE_NAME}:${IMAGE_TAG}
    fi
else
    echo -e "${RED}构建失败!${NC}"
    exit 1
fi
