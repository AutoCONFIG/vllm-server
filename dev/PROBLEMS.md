# Chat Template 视频占位符丢失问题

## 问题描述

在使用 DashScope 扩展格式（`type=video` 或 `type=video_url` 带 fps 参数）时，视频占位符 `<##VIDEO##>` 在 chat template 处理过程中丢失，导致模型无法正确处理视频内容。

## 问题表现

### 1. 模型输出异常

测试日志显示模型输出为：
```
The video shows a person performing... (但视频实际上不包含这些内容)
```

模型似乎在"编造"视频内容，而不是正确理解实际视频帧。

### 2. mm_kwargs items 类型异常

测试代码显示：
```python
first_item_type=Nonetype
```

预期应该是 `MultiModalKwargsItem` 类型，但实际为 `NoneType`，说明多模态数据处理链路存在问题。

## 测试结果对比

| Test | Format | prompt_tokens | Model Output | Status |
|------|--------|---------------|--------------|--------|
| 1 | 4 images (OpenAI) | 8184 | 空 `\n` | ❌ |
| 2 | 1 video base64 (OpenAI) | 11661 | `二` | ❌ |
| 3 | 4 images as video (fps=2) | 4109 | `user` | ❌ |
| 4 | URL images as video (fps=2) | 4109 | `user` | ❌ |
| 5 | URL video (fps=2) | 12581 | **正确描述** | ✓ |

**关键发现：只有第五轮测试（URL视频带fps参数）正常工作。**

## 根本原因

### 1. Chat Template 只处理 text 类型

Qwen 的 chat template 只处理 `content['type'] == 'text'` 的内容：

```jinja2
{%- for content in message['content'] -%}
  {%- if content['type'] == 'text' -%}
    {{ content['text'] }}
  {%- endif -%}
{%- endfor -%}
```

### 2. `_get_modality_placeholder` 返回值问题

在 `chat_utils.py` 中，当 `wrap_dicts=True`（OpenAI 格式）时，原代码返回：
```python
if wrap_dicts:
    return {"type": modality}  # modality = "video"
```

这导致 video 内容项变成 `{"type": "video"}`，但 chat template 会忽略这个类型，只处理 `{"type": "text"}` 的内容。

### 3. 占位符丢失的后果

由于 chat template 没有输出 `<##VIDEO##>` 占位符：
- 后续的多模态 token 替换流程无法找到替换位置
- 视频帧数据虽然被正确加载和解析（VideoMediaIO.load_base64 正常工作）
- 但这些数据无法被正确注入到模型的输入序列中

## 解决方案

修改 `vllm/entrypoints/chat_utils.py` 中 `_get_modality_placeholder` 函数：

```python
# 修改前
if wrap_dicts:
    return {"type": modality}

# 修改后
if wrap_dicts:
    # For multimodal content (image, video, audio), return as type='text'
    # with the placeholder as the text content. This ensures the chat template
    # will output the placeholder, which will be replaced later by actual
    # multimodal tokens during processing.
    if modality in ("image", "video", "audio"):
        placeholder = MODALITY_PLACEHOLDERS_MAP[modality]
        return {"type": "text", "text": placeholder}
    return {"type": modality}
```

## 数据流对比

### 修复前
```
Request (type=video) 
    → wrap_dicts=True 
    → {"type": "video"} 
    → chat template 忽略 
    → 占位符丢失 
    → 模型无视频输入
```

### 修复后
```
Request (type=video) 
    → wrap_dicts=True 
    → {"type": "text", "text": "<##VIDEO##>"} 
    → chat template 输出占位符 
    → 多模态 token 替换 
    → 模型正确处理视频
```

## 验证方法

1. 添加调试日志，检查 chat template 输出是否包含 `<##VIDEO##>`
2. 检查 mm_kwargs items 是否为正确的 `MultiModalKwargsItem` 类型
3. 确认 VideoMediaIO.load_base64 被调用且 metadata 正确

## 相关文件

- `vllm/entrypoints/chat_utils.py`: `_get_modality_placeholder` 函数 (约 1763-1772 行)
- `vllm/multimodal/media/video.py`: `VideoMediaIO.load_base64` 处理 video/jpeg 格式
- `vllm/model_executor/models/qwen3_vl.py`: `_get_video_second_idx` 使用 fps 计算 timestamp

## 状态

- [x] 问题分析完成
- [x] 修复代码已应用（占位符处理）
- [ ] 发现新问题：Tests 1-4 模型输出异常
- [ ] 进一步调查多模态处理链路

## 新发现的问题

### Tests 1-4 异常分析

从日志看，数据流层面是正确的：
- `VideoMediaIO.load_base64` 正确处理：`fps=2.0, duration=2.0, frames_shape=(4, 1080, 1920, 3)`
- `mm_kwargs` 和 `mm_placeholders` 都存在
- `prompt_token_ids` 从 16 扩展到 4109（Test 3/4）或 11661（Test 2）
- `mm_kwargs[video] is list, len=1, first_item_type=MultiModalKwargsItem` ✓

但模型输出却是：
- Test 1 (4 images): 空 `\n`
- Test 2 (video base64): `二`
- Test 3/4 (image list as video): `user`

**可能的根本原因：**

1. **Test 2 缺少 fps 参数**：测试代码没有传入 fps，导致视频处理使用默认值
2. **Test 1 图片处理问题**：4张图片处理可能存在问题
3. **模型生成异常**：输入数据正确，但模型生成逻辑有问题

### 下一步调查方向

1. 检查 `_call_hf_processor` 中 video 数据的实际处理
2. 对比 Test 5（成功）和 Tests 1-4（失败）的数据差异
3. 检查 prompt_token_ids 序列是否正确包含多模态 token
