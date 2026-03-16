# vLLM Engine Server - 模块化版本

一个模块化、易维护的vLLM推理服务器，支持多卡并行、多模态输入和完整的API功能。

## 项目结构

```
vllm_engine_server/
├── config/                    # 配置管理模块
│   ├── __init__.py
│   ├── config.py             # 配置加载器
│   ├── settings.py            # 配置模型类
│   └── config.yaml           # 默认配置文件
│
├── core/                      # 核心引擎模块
│   ├── __init__.py
│   ├── engine.py              # Engine管理器（单例）
│   └── engine_args.py         # Engine参数构建
│
├── api/                       # API层
│   ├── __init__.py
│   ├── app.py                # FastAPI应用创建
│   ├── routes/               # 路由处理
│   │   ├── __init__.py
│   │   ├── chat.py            # 聊天补全路由
│   │   ├── health.py          # 健康检查路由
│   │   └── models.py          # 模型列表路由
│   └── schemas/              # Pydantic模型
│       ├── __init__.py
│       ├── chat.py             # 聊天请求/响应模型
│       └── common.py           # 通用响应模型
│
├── services/                  # 业务服务层
│   ├── __init__.py
│   ├── chat_service.py        # 聊天服务
│   └── lifespan_service.py    # 应用生命周期服务
│
├── multimodal/               # 多模态处理模块
│   ├── __init__.py
│   ├── base.py               # 加载器基类
│   ├── image_loader.py        # 图像加载器
│   ├── video_loader.py        # 视频加载器
│   ├── processor.py           # 多模态数据处理器
│   └── mapper.py             # 消息到多模态映射
│
├── utils/                     # 工具模块
│   ├── __init__.py
│   ├── logger.py             # 日志工具
│   └── validators.py         # 数据验证工具
│
├── logs/                      # 日志目录
│   ├── api_requests.log
│   └── server.log
│
├── server.py                 # 入口文件
├── config.yaml               # 运行时配置文件
└── start_server.sh          # 启动脚本
```

## 快速开始

### 1. 使用配置文件启动

```bash
# 默认配置启动
python server.py

# 使用自定义配置文件
python server.py --config custom_config.yaml
```

### 2. 使用命令行参数启动

```bash
# 指定模型路径
python server.py --model /path/to/model

# 调整服务器端口
python server.py --port 8080

# 启用请求日志
python server.py --log-requests true
```

### 3. 使用环境变量启动

```bash
# 设置服务器地址
VLLM_SERVER_PORT=8000 python server.py

# 设置模型路径
VLLM_MODEL_PATH=/path/to/model python server.py

# 设置Engine配置
VLLM_ENGINE_GPU_MEMORY_UTILIZATION=0.9 python server.py
VILLM_ENGINE_MAX_MODEL_LEN=4096 python server.py
```

### 4. 多卡并行启动

```bash
# 张量并行（单节点多GPU）
python server.py --tensor-parallel-size 4

# 流水线并行（多节点）
python server.py --tensor-parallel-size 4 --pipeline-parallel-size 2

# 数据并行（需要Ray后端）
python server.py --data-parallel-size 4 --distributed-executor-backend ray
```

## 配置说明

配置加载优先级：**环境变量 > 命令行参数 > YAML配置文件 > 默认值**

### 配置文件 (config.yaml)

```yaml
server:
  host: "0.0.0.0"
  port: 8000

model:
  path: "/path/to/model"
  dtype: "bfloat16"

engine:
  gpu_memory_utilization: 0.95
  max_model_len: 8192
  max_num_seqs: 64
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  distributed_executor_backend: "mp"

multimodal:
  limit_mm_per_prompt:
    image: 4
    video: 2

logging:
  level: "INFO"
  log_requests: false
  log_dir: "logs"
```

### 环境变量命名规则

| 配置项 | 环境变量 |
|---------|-----------|
| 服务器主机 | `VLLM_SERVER_HOST` |
| 服务器端口 | `VLLM_SERVER_PORT` |
| 模型路径 | `VLLM_MODEL_PATH` |
| GPU显存利用率 | `VLLM_ENGINE_GPU_MEMORY_UTILIZATION` |
| 最大模型长度 | `VLLM_ENGINE_MAX_MODEL_LEN` |
| 最大序列数 | `VLLM_ENGINE_MAX_NUM_SEQS` |
| 张量并行大小 | `VLLM_ENGINE_TENSOR_PARALLEL_SIZE` |
| 流水线并行大小 | `VLLM_ENGINE_PIPELINE_PARALLEL_SIZE` |
| 数据并行大小 | `VLLM_ENGINE_DATA_PARALLEL_SIZE` |
| 分布式后端 | `VLLM_ENGINE_DISTRIBUTED_EXECUTOR_BACKEND` |
| 日志级别 | `VLLM_LOGGING_LEVEL` |
| 记录请求 | `VLLM_LOGGING_LOG_REQUESTS` |

## API 端点

### 健康检查
```bash
curl http://localhost:8000/health
```

### 模型列表
```bash
curl http://localhost:8000/v1/models
```

### 聊天补全（非流式）
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-3.5",
    "messages": [
      {"role": "user", "content": "你好！"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

### 聊天补全（流式）
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-3.5",
    "messages": [
      {"role": "user", "content": "你好！"}
    ],
    "stream": true
  }'
```

### 多模态请求（图像）
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-3.5",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "描述这张图片"},
          {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
      }
    ]
  }'
```

### 多模态请求（视频）
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-3.5",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "分析这段视频"},
          {
            "type": "video",
            "video": "https://example.com/video.mp4",
            "total_pixels": 720 * 1280,
            "fps": 30
          }
        ]
      }
    ]
  }'
```

## 模块开发指南

### 添加新功能

1. **确定模块位置**
   - 核心功能 → `core/`
   - API功能 → `api/routes/` 或 `api/schemas/`
   - 业务逻辑 → `services/`
   - 多模态处理 → `multimodal/`
   - 工具函数 → `utils/`

2. **实现代码**
   - 遵循项目规范（见 [`ARCHITECTURE.md`](ARCHITECTURE.md)）
   - 使用类型注解
   - 添加文档字符串

3. **导出接口**
   - 在 `__init__.py` 中导出公共接口

### 添加新配置项

1. **在 `config/settings.py` 中添加配置模型**
   ```python
   class MyConfig(BaseModel):
       my_setting: str = "default_value"
   ```

2. **在 `config.yaml` 中添加默认值**
   ```yaml
   my_config:
       my_setting: "default_value"
   ```

3. **添加环境变量支持**
   - 环境变量名：`VLLM_MY_CONFIG_MY_SETTING`
   - 在 `server.py` 的 `load_config()` 中添加解析逻辑

## 迁移指南

### 从旧版本迁移

1. **备份旧文件**
   ```bash
   cp vllm_engine_server.py vllm_engine_server.py.bak
   ```

2. **使用新入口启动**
   ```bash
   python server.py
   ```

3. **API兼容性**
   - 新版本完全兼容旧版本的API接口
   - 相同的端点和请求/响应格式

### 配置迁移

旧版本的命令行参数可以直接映射到新版本：

| 旧参数 | 新参数/配置 |
|---------|---------------|
| `--model, -m` | `--model`, `VLLM_MODEL_PATH` |
| `--host` | `--host`, `VLLM_SERVER_HOST` |
| `--port, -p` | `--port`, `VLLM_SERVER_PORT` |
| `--gpu-util, -g` | `--gpu-util`, `VLLM_ENGINE_GPU_MEMORY_UTILIZATION` |
| `--max-len, -l` | `--max-len`, `VLLM_ENGINE_MAX_MODEL_LEN` |
| `--max-seqs` | `--max-seqs`, `VLLM_ENGINE_MAX_NUM_SEQS` |
| `--tensor-parallel-size, -tp` | `--tensor-parallel-size`, `VLLM_ENGINE_TENSOR_PARALLEL_SIZE` |
| `--pipeline-parallel-size, -pp` | `--pipeline-parallel-size`, `VLLM_ENGINE_PIPELINE_PARALLEL_SIZE` |
| `--data-parallel-size, -dp` | `--data-parallel-size`, `VLLM_ENGINE_DATA_PARALLEL_SIZE` |
| `--distributed-executor-backend` | `--distributed-executor-backend`, `VLLM_ENGINE_DISTRIBUTED_EXECUTOR_BACKEND` |
| `--log-requests` | `--log-requests`, `VLLM_LOGGING_LOG_REQUESTS` |

## 调试

### 启用调试日志

```bash
# 方法1：命令行参数
python server.py --log-level DEBUG

# 方法2：环境变量
VLLM_LOGGING_LEVEL=DEBUG python server.py

# 方法3：配置文件
# 编辑 config.yaml
logging:
  level: "DEBUG"
```

### 查看日志

```bash
# 查看请求日志
tail -f logs/api_requests.log

# 查看服务器日志
tail -f logs/server.log

# 统计请求数量
wc -l logs/api_requests.log
```

## 依赖安装

```bash
# 基础依赖
pip install fastapi uvicorn pydantic pydantic-settings pyyaml

# vLLM依赖
pip install vllm

# 可选依赖（Ray后端）
pip install ray
```

## 架构设计

本模块化项目遵循以下设计原则：

1. **分层架构**
   - API层：处理HTTP请求/响应
   - Services层：业务逻辑编排
   - Core层：引擎核心管理
   - Utils层：通用工具

2. **配置管理**
   - 统一的配置加载器
   - 支持多种配置源（YAML、环境变量、命令行）
   - 类型安全的配置模型

3. **模块独立性**
   - 每个模块职责单一
   - 模块间依赖最小化
   - 易于测试和维护

更多详细设计规范，请参阅 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

## License

MIT
