# vLLM 多模态视频 FPS 支持 - OpenAI 兼容 API 扩展

本项目在标准 OpenAI 兼容 API 基础上，扩展支持阿里云 DashScope 的视频 FPS 功能。

## 概述

本质上是 OpenAI 兼容格式的扩展，完全兼容标准 OpenAI API，同时增加以下扩展格式：

### 1. 标准 OpenAI 格式（已支持）

#### 视频文件 URL
```python
{
    "type": "video_url",
    "video_url": {
        "url": "https://example.com/video.mp4"
    }
}
```

#### 图片 URL
```python
{
    "type": "image_url",
    "image_url": {
        "url": "https://example.com/image.jpg"
    }
}
```

### 2. DashScope 扩展格式（新增支持）

#### 扩展1：视频文件 URL + FPS 参数
```python
{
    "type": "video_url",
    "video_url": {
        "url": "https://example.com/video.mp4"
    },
    "fps": 2  # 扩展参数：控制抽帧频率
}
```

**fps 参数说明**：
- 控制抽帧频率，每隔 `1/fps` 秒抽取一帧
- 取值范围：[0.1, 10]
- 默认值：2.0
- 高速运动场景建议设置较高的 fps 值
- 静态或长视频建议设置较低的 fps 值

#### 扩展2：图片列表作为视频帧 + FPS 参数
```python
{
    "type": "video",
    "video": [
        "https://example.com/frame1.jpg",
        "https://example.com/frame2.jpg",
        "https://example.com/frame3.jpg",
        "https://example.com/frame4.jpg"
    ],
    "fps": 2  # 扩展参数：告知模型帧之间的时间间隔
}
```

**用途**：
- 当视频以图像列表（预先抽取的视频帧）传入时
- 通过 fps 参数告知模型视频帧之间的时间间隔
- 帮助模型更准确地理解事件的顺序、持续时间和动态变化

## 技术实现

### 核心修改文件

1. **vllm/entrypoints/chat_utils.py**
   - 新增 `parse_video_from_image_urls` 方法处理图片列表格式
   - 扩展 `_parse_chat_message_content_part` 处理 `type=video` 格式
   - 支持 fps 参数传递到 `media_io_kwargs`

2. **vllm/multimodal/media/video.py**
   - `VideoMediaIO.load_base64` 支持 `video/jpeg` 格式
   - 从 kwargs 中获取 fps 参数构建 metadata
   - metadata 包含：fps, duration, frames_indices, do_sample_frames

3. **vllm/model_executor/models/qwen3_vl.py**
   - `_get_video_second_idx` 使用 metadata 中的 fps 计算时间戳
   - `do_sample_frames=False` 时直接使用提供的帧，不再重新采样

### 数据流

```
Request (DashScope format)
    ↓
_parse_chat_message_content_part (检测 type=video)
    ↓
parse_video_from_image_urls (获取图片列表)
    ↓
_video_from_image_urls_async (并发获取图片，组合为 video/jpeg 格式)
    ↓
VideoMediaIO.load_base64 (解析帧数据，构建 metadata)
    ↓
Qwen3VLMultiModalProcessor._call_hf_processor (处理视频数据)
    ↓
_get_video_second_idx (使用 fps 计算时间戳)
```

### video/jpeg 格式

内部使用的合成格式，用于表示图片帧序列：

```
data:video/jpeg;base64,<frame1_base64>,<frame2_base64>,...
```

metadata 结构：
```python
{
    "total_num_frames": 4,
    "fps": 2.0,           # 从请求参数获取
    "duration": 2.0,      # total_num_frames / fps
    "video_backend": "jpeg_sequence",
    "frames_indices": [0, 1, 2, 3],
    "do_sample_frames": False  # 表示已经预先采样，不需要再采样
}
```

## 使用示例

### 标准 OpenAI 格式
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")

# 标准 OpenAI 格式
response = client.chat.completions.create(
    model="qwen3-vl",
    messages=[{
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}},
            {"type": "text", "text": "描述这个视频"}
        ]
    }]
)
```

### DashScope 扩展格式1：视频 URL + FPS
```python
response = client.chat.completions.create(
    model="qwen3-vl",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {"url": "https://example.com/video.mp4"},
                "fps": 2
            },
            {"type": "text", "text": "描述这个视频"}
        ]
    }]
)
```

### DashScope 扩展格式2：图片列表 + FPS
```python
response = client.chat.completions.create(
    model="qwen3-vl",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": [
                    "https://example.com/frame1.jpg",
                    "https://example.com/frame2.jpg",
                    "https://example.com/frame3.jpg",
                    "https://example.com/frame4.jpg"
                ],
                "fps": 2
            },
            {"type": "text", "text": "描述这个视频的具体过程"}
        ]
    }]
)
```

## 兼容性

- **向后兼容**：完全兼容标准 OpenAI API 格式
- **扩展兼容**：DashScope 扩展格式作为增强功能，不影响原有功能
- **模型支持**：Qwen3-VL、Qwen2.5-VL 等支持视频理解的模型
