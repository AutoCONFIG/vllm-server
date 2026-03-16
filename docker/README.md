# vLLM Engine Server Docker部署

基于vLLM官方镜像构建，支持GPU加速推理。

## 快速开始

### 1. 构建镜像

```bash
# 默认构建
docker build -t vllm-engine-server:latest -f docker/Dockerfile.custom .

# 指定vLLM版本
docker build -t vllm-engine-server:v0.11.0 \
    --build-arg VLLM_IMAGE=vllm/vllm-openai:v0.11.0 \
    -f docker/Dockerfile.custom .
```

### 2. 运行容器

```bash
# 基本运行
docker run --gpus all -p 8000:8000 \
    -v /path/to/model:/model \
    -e VLLM_MODEL_PATH=/model \
    --shm-size=8g --ipc=host \
    vllm-engine-server:latest

# 多GPU运行
docker run --gpus '"device=0,1,2,3"' -p 8000:8000 \
    -v /path/to/model:/model \
    -e VLLM_MODEL_PATH=/model \
    -e VLLM_ENGINE_TENSOR_PARALLEL_SIZE=4 \
    --shm-size=16g --ipc=host \
    vllm-engine-server:latest
```

### 3. 使用Docker Compose

```bash
# 复制环境变量配置
cp .env.example .env

# 编辑.env，设置MODEL_PATH
vim .env

# 启动
docker-compose up -d
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `VLLM_MODEL_PATH` | 模型路径 | - |
| `VLLM_ENGINE_TENSOR_PARALLEL_SIZE` | 张量并行大小 | 1 |
| `VLLM_ENGINE_GPU_MEMORY_UTILIZATION` | GPU显存利用率 | 0.95 |
| `VLLM_ENGINE_MAX_MODEL_LEN` | 最大上下文长度 | 8192 |

## 注意事项

1. **GPU支持**: 需要安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
2. **共享内存**: 多卡并行时设置 `--shm-size` 或使用 `--ipc=host`
3. **模型路径**: 必须通过 `-v` 挂载模型目录并设置 `VLLM_MODEL_PATH`

## API测试

```bash
# 健康检查
curl http://localhost:8000/health

# 聊天补全
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "qwen", "messages": [{"role": "user", "content": "你好"}]}'
```
