# vLLM Engine Server 模块化架构规范

## 1. 设计原则

### 1.1 核心原则
- **单一职责**: 每个模块只负责一项明确的功能
- **高内聚低耦合**: 模块内部紧密度高，模块间依赖最小化
- **清晰层次**: 严格遵循分层架构，数据流向清晰
- **可扩展性**: 新功能可以轻松添加而不影响现有代码

### 1.2 分层架构
```
┌─────────────────────────────────────────┐
│           入口层 (entrypoint)            │
│    server.py / main.py                  │
├─────────────────────────────────────────┤
│           API层 (api)                    │
│    路由定义 / 请求模型 / 响应模型         │
├─────────────────────────────────────────┤
│         Services层 (services)            │
│    业务逻辑 / 编排调度 / 数据处理         │
├─────────────────────────────────────────┤
│          Core层 (core)                   │
│    Engine管理 / 推理核心 / 配置          │
├─────────────────────────────────────────┤
│       Multimodal层 (multimodal)          │
│    图像处理 / 视频处理 / 媒体转换         │
├─────────────────────────────────────────┤
│          Utils层 (utils)                 │
│    日志 / 配置加载 / 公共工具            │
└─────────────────────────────────────────┘
```

## 2. 目录结构

```
vllm_engine_server/
├── config/                    # 配置管理模块
│   ├── __init__.py
│   ├── config.py             # 配置加载器
│   ├── settings.py            # 配置模型类
│   └── templates/
│       └── config.yaml        # 配置文件模板
│
├── core/                      # 核心引擎模块
│   ├── __init__.py
│   ├── engine.py             # Engine管理器（单例）
│   ├── engine_args.py        # Engine参数构建
│   └── constants.py          # 常量定义
│
├── api/                       # API层
│   ├── __init__.py
│   ├── app.py                # FastAPI应用创建
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── chat.py           # 聊天补全路由
│   │   ├── health.py         # 健康检查路由
│   │   └── models.py         # 模型列表路由
│   └── schemas/
│       ├── __init__.py
│       ├── chat.py           # 聊天请求/响应模型
│       └── common.py         # 通用模型
│
├── services/                  # 业务服务层
│   ├── __init__.py
│   ├── chat_service.py       # 聊天服务
│   ├── prompt_service.py     # 提示词服务
│   └── lifespan_service.py   # 应用生命周期服务
│
├── multimodal/               # 多模态处理模块
│   ├── __init__.py
│   ├── base.py               # 基类定义
│   ├── image_loader.py       # 图像加载器
│   ├── video_loader.py       # 视频加载器
│   ├── processor.py          # 多模态数据处理
│   └── mapper.py             # 消息到多模态映射
│
├── utils/                     # 工具模块
│   ├── __init__.py
│   ├── logger.py             # 日志工具
│   ├── validators.py         # 数据验证工具
│   └── helpers.py             # 辅助函数
│
├── logs/                      # 日志目录
│   ├── api_requests.log
│   └── server.log
│
├── config.yaml               # 运行时配置文件
├── server.py                 # 入口文件
├── start_server.sh          # 启动脚本
├── tests/                    # 测试目录
│   └── ...
│
└── README.md                 # 项目说明文档
```

## 3. 模块详细规范

### 3.1 config/ 模块

**职责**: 统一管理所有配置，支持YAML文件和环境变量

**文件说明**:
- `config.py`: 配置加载器，自动合并YAML和环境变量
- `settings.py`: Pydantic配置模型，定义配置结构
- `config.yaml`: 默认配置文件模板

**配置优先级**: 环境变量 > YAML配置 > 默认值

### 3.2 core/ 模块

**职责**: 管理vLLM Engine的核心功能

**文件说明**:
- `engine.py`: EngineManager单例类，负责Engine初始化、 shutdown
- `engine_args.py`: AsyncEngineArgs构建器
- `constants.py`: 硬编码常量（默认路径、超时等）

**核心类**:
```python
class EngineManager:
    """Engine单例管理器"""
    
    async def initialize(config: ServerConfig) -> None:
        """初始化Engine"""
        
    async def shutdown() -> None:
        """关闭Engine"""
        
    @property
    def engine(self) -> AsyncLLMEngine:
        """获取Engine实例"""
```

### 3.3 api/ 模块

**职责**: 定义FastAPI路由和数据模型

**目录结构**:
```
api/
├── routes/           # 路由处理
│   ├── chat.py       # /v1/chat/completions
│   ├── health.py     # /health
│   └── models.py     # /v1/models
└── schemas/          # Pydantic模型
    ├── chat.py        # ChatRequest, ChatResponse
    └── common.py      # 通用响应模型
```

### 3.4 services/ 模块

**职责**: 业务逻辑编排，不直接涉及底层实现

**文件说明**:
- `chat_service.py`: 聊天补全业务逻辑
- `prompt_service.py`: 消息到Prompt的转换
- `lifespan_service.py`: 应用生命周期管理

### 3.5 multimodal/ 模块

**职责**: 统一处理多模态数据（图像、视频）

**文件说明**:
- `base.py`: 抽象基类，定义加载器接口
- `image_loader.py`: 图像加载（URL/base64/本地文件）
- `video_loader.py`: 视频加载（URL/base64/本地文件）
- `processor.py`: 多模态数据预处理
- `mapper.py`: 消息格式到多模态数据的映射

### 3.6 utils/ 模块

**职责**: 通用工具函数

**文件说明**:
- `logger.py`: 日志配置和工具函数
- `validators.py`: 数据验证工具
- `helpers.py`: 其他辅助函数

## 4. 模块依赖关系

```
server.py (入口)
    │
    ├── api/app.py (创建FastAPI)
    │       │
    │       ├── api/routes/* (路由处理)
    │       │       │
    │       │       └── services/* (业务逻辑)
    │       │               │
    │       │               ├── core/* (Engine管理)
    │       │               │
    │       │               └── multimodal/* (多模态处理)
    │       │
    │       └── services/lifespan_service.py
    │               │
    │               └── core/engine.py
    │
    └── config/* (配置管理)
            │
            └── utils/logger.py
```

## 5. 命名规范

### 5.1 文件命名
- **模块目录**: 小写下划线分隔 `multimodal/`
- **Python文件**: 小写下划线分隔 `image_loader.py`
- **测试文件**: `test_` 前缀 `test_image_loader.py`

### 5.2 类命名
- **普通类**: 大驼峰 `class EngineManager`
- **异常类**: `Error` 后缀 `class MediaLoadError`
- **基类**: `Base` 前缀 `class BaseLoader`

### 5.3 函数命名
- **公开函数**: 小写下划线 `def load_image()`
- **内部函数**: 单下划线前缀 `def _internal_func()`
- **异步函数**: `async def` 语法

### 5.4 常量命名
- **模块常量**: 大写下划线 `DEFAULT_MAX_TOKENS`
- **配置键**: 小写下划线 `max_model_len`

## 6. 导入规范

### 6.1 绝对导入
```python
# 推荐使用绝对导入
from core.engine import EngineManager
from api.routes import chat
```

### 6.2 相对导入（模块内）
```python
# 仅在同一包内使用相对导入
from ..services import ChatService
from .schemas import ChatRequest
```

### 6.3 导入顺序
1. 标准库
2. 第三方库
3. 项目内部模块
4. 本地模块

```python
# 标准库
import asyncio
import json
from typing import Optional

# 第三方库
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# 项目内部模块（按层次从外到内）
from config import Settings
from core import EngineManager
from services import ChatService
from multimodal import ImageLoader

# 本地
from .routes import chat
```

## 7. 错误处理规范

### 7.1 异常定义
在 `core/exceptions.py` 中定义项目专用异常：
```python
class EngineError(Exception):
    """Engine相关基础异常"""
    pass

class EngineNotInitializedError(EngineError):
    """Engine未初始化异常"""
    pass

class MediaLoadError(Exception):
    """媒体加载失败异常"""
    pass

class ValidationError(Exception):
    """数据验证失败异常"""
    pass
```

### 7.2 全局异常处理器
在 `api/app.py` 中统一处理：
```python
@app.exception_handler(EngineNotInitializedError)
async def engine_not_initialized_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"error": "Engine not ready"}
    )
```

## 8. 配置规范

### 8.1 config.yaml 结构
```yaml
server:
  host: "0.0.0.0"
  port: 8000
  title: "vLLM Engine Server"
  version: "1.0.0"

model:
  path: "/data2/kaiyun/qwen/model"
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

### 8.2 环境变量覆盖
```bash
# 通过环境变量覆盖配置
VLLM_SERVER_PORT=8000
VLLM_MODEL_PATH=/custom/path
VLLM_GPU_UTIL=0.9
```

## 9. 测试规范

### 9.1 测试目录结构
```
tests/
├── unit/                  # 单元测试
│   ├── test_core/
│   ├── test_multimodal/
│   └── test_services/
├── integration/           # 集成测试
│   └── test_api/
└── fixtures/             # 测试数据
    ├── images/
    └── videos/
```

### 9.2 测试命名
```python
# test_<模块名>_<功能名>
def test_engine_args_build_with_gpu_util():
    """测试Engine参数构建"""
    pass

def test_image_loader_from_base64():
    """测试Base64图像加载"""
    pass
```

## 10. 文档规范

### 10.1 模块文档字符串
```python
"""
多模态图像加载模块

提供从多种来源（URL、Base64、本地文件）加载图像的功能。
支持JPEG、PNG、GIF等常见格式。

Classes:
    ImageLoader: 图像加载器类
    ImageLoadError: 图像加载异常

Example:
    >>> loader = ImageLoader()
    >>> image = loader.from_url("http://example.com/image.jpg")
"""

class ImageLoader:
    """图像加载器"""
    
    def from_url(self, url: str) -> Image:
        """从URL加载图像
        
        Args:
            url: 图像URL
            
        Returns:
            PIL.Image: 加载的图像对象
            
        Raises:
            ImageLoadError: 加载失败时抛出
        """
        pass
```

## 11. 迁移检查清单

模块化迁移时需要确保：
- [ ] 所有现有功能保持不变
- [ ] API接口完全兼容
- [ ] 命令行参数保持向后兼容
- [ ] 配置文件可平滑迁移
- [ ] 日志格式保持一致
- [ ] 错误消息保持一致

## 12. 版本兼容性

- Python: >= 3.10
- FastAPI: >= 0.100
- vLLM: >= 0.4
- Pydantic: >= 2.0
