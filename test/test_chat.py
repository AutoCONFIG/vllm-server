#!/usr/bin/env python3
"""
vLLM 多模态推理客户端测试 - 对话测试

测试轮次：
第一轮：本地4张图片（OpenAI标准格式，base64编码）
第二轮：本地1个视频（OpenAI标准格式，base64编码）
第三轮：本地4图图片列表（DashScope扩展格式，type=video）
第四轮：URL 4图列表（DashScope扩展格式，type=video）
第五轮：URL视频（OpenAI标准格式，type=video_url）
"""

import base64
import os
import sys
import time
from pathlib import Path

# ============ 配置项，请根据实际调整 =============
BASE_URL = "http://localhost:8000/v1"
API_KEY = "token-abc123"
MODEL_NAME = "qwen3-vl"

# 本地图片路径
LOCAL_IMAGE_PATHS = [
    "./1.jpg",
    "./2.jpg",
    "./3.jpg",
    "./4.jpg",
]

# 本地视频路径
LOCAL_VIDEO_PATHS = [
    "./test_video1.mp4",
]

# ===================================================
# URL配置 - 请在此处填入您的URL
# ===================================================

# 图片URL（第四轮使用）
IMAGE_URLS = [
    "http://cavinet-traffic.oss-cn-hangzhou.aliyuncs.com/event/2026-03-20/image/69bd12a1e4b04abf63d4f9fc.jpg",
    "http://cavinet-traffic.oss-cn-hangzhou.aliyuncs.com/event/2026-03-20/image/69bd11bde4b04abf63d4f981.jpg",
    "http://cavinet-traffic.oss-cn-hangzhou.aliyuncs.com/event/2026-03-20/image/69bd1279e4b04abf63d4f9e7.jpg",
    "http://cavinet-traffic.oss-cn-hangzhou.aliyuncs.com/event/2026-03-20/image/69bd1259e4b04abf63d4f9d1.jpg",
]

# 视频URL（第五轮使用）
VIDEO_URLS = [
    "http://cavinet-traffic.oss-cn-hangzhou.aliyuncs.com/event/2026-03-20/video/69bd1258e4b04abf63d4f9ce.mp4",
]

# ===================================================

TEMPERATURE = 0.7
FPS = 2  # 视频帧率


def encode_file_to_base64(file_path: str) -> str:
    """将本地文件转换为base64 URL"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(file_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".avi": "video/avi",
    }
    mime_type = mime_types.get(ext, "application/octet-stream")
    return f"data:{mime_type};base64,{data}"


def get_images_base64(image_paths: list) -> list:
    """获取本地图片的base64 URL列表"""
    images = []
    for path in image_paths:
        try:
            images.append(encode_file_to_base64(path))
        except FileNotFoundError:
            print(f"图片文件不存在: {path}")
            return None
    return images


def get_videos_base64(video_paths: list) -> list:
    """获取本地视频的base64 URL列表"""
    videos = []
    for path in video_paths:
        try:
            videos.append(encode_file_to_base64(path))
        except FileNotFoundError:
            print(f"视频文件不存在: {path}")
            return None
    return videos


def estimate_tokens(text: str) -> int:
    """简单估算token数，1 token 约等于 1.5字符"""
    return max(1, int(len(text) / 1.5))


def inference_with_messages(client, messages: list, test_name: str):
    """使用对话信息进行推理"""
    start_time = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
        )
        total_time = time.perf_counter() - start_time
        text = response.choices[0].message.content
        token_count = estimate_tokens(text)
        token_per_sec = token_count / total_time if total_time > 0 else 0
        return True, text, total_time, token_per_sec, response
    except Exception as e:
        total_time = time.perf_counter() - start_time
        return False, str(e), total_time, 0, None


def test_round_1_images_base64(client):
    """第一轮测试：本地4张图片（OpenAI标准格式，base64编码）"""
    print("\n" + "=" * 60)
    print("第一轮测试：本地4张图片（OpenAI标准格式，base64编码）")
    print("=" * 60)
    
    # 获取本地图片的base64
    images = get_images_base64(LOCAL_IMAGE_PATHS)
    if images is None:
        print("图片文件不存在，测试终止")
        return None
    
    # 构建OpenAI标准格式的消息
    user_content = [
        {"type": "text", "text": "请详细描述这4张图片的内容。"}
    ]
    for img in images:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": img}
        })
    
    messages = [
        {"role": "user", "content": user_content}
    ]
    
    print(f"图片数量: {len(images)}")
    print("格式: OpenAI标准 (type=image_url)")
    print("发送请求中...")
    
    success, text, total_time, token_per_sec, response = inference_with_messages(
        client, messages, "第一轮4图片(base64)"
    )
    
    if success:
        print(f"\n响应时间: {total_time:.3f} 秒")
        print(f"Token生成速率: {token_per_sec:.1f} tokens/秒")
        print(f"\n模型回复: {text}")
    else:
        print(f"\n请求失败: {text}")
    
    return messages, response


def test_round_2_video_base64(client):
    """第二轮测试：本地1个视频（OpenAI标准格式，base64编码）"""
    print("\n" + "=" * 60)
    print("第二轮测试：本地1个视频（OpenAI标准格式，base64编码）")
    print("=" * 60)
    
    # 获取本地视频的base64
    videos = get_videos_base64(LOCAL_VIDEO_PATHS)
    if videos is None:
        print("视频文件不存在，测试终止")
        return None
    
    # 构建OpenAI标准格式的消息
    user_content = [
        {"type": "text", "text": "请详细描述这个视频的内容。"}
    ]
    for video in videos:
        user_content.append({
            "type": "video_url",
            "video_url": {"url": video}
        })
    
    messages = [
        {"role": "user", "content": user_content}
    ]
    
    print(f"视频数量: {len(videos)}")
    print("格式: OpenAI标准 (type=video_url)")
    print("发送请求中...")
    
    success, text, total_time, token_per_sec, response = inference_with_messages(
        client, messages, "第二轮视频(base64)"
    )
    
    if success:
        print(f"\n响应时间: {total_time:.3f} 秒")
        print(f"Token生成速率: {token_per_sec:.1f} tokens/秒")
        print(f"\n模型回复: {text}")
    else:
        print(f"\n请求失败: {text}")
    
    return messages, response


def test_round_3_local_image_list(client):
    """第三轮测试：本地4图图片列表（DashScope扩展格式，type=video）"""
    print("\n" + "=" * 60)
    print("第三轮测试：本地4图图片列表（DashScope扩展格式，type=video）")
    print("=" * 60)
    
    # 获取本地图片的base64
    images = get_images_base64(LOCAL_IMAGE_PATHS)
    if images is None:
        print("图片文件不存在，测试终止")
        return None
    
    # 构建DashScope扩展格式的消息
    # 将图片列表作为视频帧传入，使用fps参数
    user_content = [
        {
            "type": "video",
            "video": images,
            "fps": FPS
        },
        {"type": "text", "text": "请详细描述这个视频的具体过程。"}
    ]
    
    messages = [
        {"role": "user", "content": user_content}
    ]
    
    print(f"图片数量: {len(images)}")
    print(f"格式: DashScope扩展 (type=video, fps={FPS})")
    print("发送请求中...")
    
    success, text, total_time, token_per_sec, response = inference_with_messages(
        client, messages, "第三轮图片列表(base64)"
    )
    
    if success:
        print(f"\n响应时间: {total_time:.3f} 秒")
        print(f"Token生成速率: {token_per_sec:.1f} tokens/秒")
        print(f"\n模型回复: {text}")
    else:
        print(f"\n请求失败: {text}")
    
    return messages, response


def test_round_4_url_image_list(client):
    """第四轮测试：URL 4图列表（DashScope扩展格式，type=video）"""
    print("\n" + "=" * 60)
    print("第四轮测试：URL 4图列表（DashScope扩展格式，type=video）")
    print("=" * 60)
    
    print(f"图片URL列表:")
    for i, url in enumerate(IMAGE_URLS):
        print(f"  {i+1}. {url}")
    
    # 构建DashScope扩展格式的消息
    # 将图片URL列表作为视频帧传入，使用fps参数
    user_content = [
        {
            "type": "video",
            "video": IMAGE_URLS,
            "fps": FPS
        },
        {"type": "text", "text": "请详细描述这个视频的具体过程。"}
    ]
    
    messages = [
        {"role": "user", "content": user_content}
    ]
    
    print(f"\n格式: DashScope扩展 (type=video, fps={FPS})")
    print("发送请求中...")
    
    success, text, total_time, token_per_sec, response = inference_with_messages(
        client, messages, "第四轮URL图片列表"
    )
    
    if success:
        print(f"\n响应时间: {total_time:.3f} 秒")
        print(f"Token生成速率: {token_per_sec:.1f} tokens/秒")
        print(f"\n模型回复: {text}")
    else:
        print(f"\n请求失败: {text}")
    
    return messages, response


def test_round_5_url_video(client):
    """第五轮测试：URL视频（OpenAI标准格式，type=video_url）"""
    print("\n" + "=" * 60)
    print("第五轮测试：URL视频（OpenAI标准格式，type=video_url）")
    print("=" * 60)
    
    print(f"视频URL:")
    for i, url in enumerate(VIDEO_URLS):
        print(f"  {i+1}. {url}")
    
    # 构建OpenAI标准格式的消息
    user_content = [
        {"type": "text", "text": "请详细描述这个视频的内容。"}
    ]
    for url in VIDEO_URLS:
        user_content.append({
            "type": "video_url",
            "video_url": {"url": url},
            "fps": FPS  # DashScope扩展：在video_url类型中也支持fps参数
        })
    
    messages = [
        {"role": "user", "content": user_content}
    ]
    
    print(f"\n格式: OpenAI标准 + DashScope扩展fps参数")
    print("发送请求中...")
    
    success, text, total_time, token_per_sec, response = inference_with_messages(
        client, messages, "第五轮URL视频"
    )
    
    if success:
        print(f"\n响应时间: {total_time:.3f} 秒")
        print(f"Token生成速率: {token_per_sec:.1f} tokens/秒")
        print(f"\n模型回复: {text}")
    else:
        print(f"\n请求失败: {text}")
    
    return messages, response


def main():
    from openai import OpenAI
    from pathlib import Path

    print("=" * 60)
    print("vLLM 多模态对话测试")
    print("=" * 60)
    print(f"服务器地址: {BASE_URL}")
    print(f"API Key: {API_KEY}")
    print(f"模型名称: {MODEL_NAME}")
    print(f"温度参数: {TEMPERATURE}")
    print(f"视频帧率: {FPS}")
    
    print(f"\n本地图片文件 ({len(LOCAL_IMAGE_PATHS)} 张):")
    for i, p in enumerate(LOCAL_IMAGE_PATHS):
        print(f"  {i+1}. {p}")
    
    print(f"\n本地视频文件 ({len(LOCAL_VIDEO_PATHS)} 个):")
    for i, p in enumerate(LOCAL_VIDEO_PATHS):
        print(f"  {i+1}. {p}")
    
    print(f"\n图片URL ({len(IMAGE_URLS)} 张):")
    for i, url in enumerate(IMAGE_URLS):
        print(f"  {i+1}. {url}")
    
    print(f"\n视频URL ({len(VIDEO_URLS)} 个):")
    for i, url in enumerate(VIDEO_URLS):
        print(f"  {i+1}. {url}")

    print("\n测试计划:")
    print("  第一轮: 本地4图 (OpenAI标准, base64)")
    print("  第二轮: 本地1视频 (OpenAI标准, base64)")
    print("  第三轮: 本地4图列表 (DashScope扩展, type=video)")
    print("  第四轮: URL 4图列表 (DashScope扩展, type=video)")
    print("  第五轮: URL视频 (OpenAI标准 + fps)")

    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )

    try:
        _ = client.models.list()
        print("\n✓ 服务器连接成功")
    except Exception as e:
        print(f"\n✗ 无法连接到服务器: {e}")
        print("请确保vLLM服务正在运行!")
        sys.exit(1)

    # 第一轮：本地4张图片（OpenAI标准格式，base64编码）
    result1 = test_round_1_images_base64(client)
    if result1 is None:
        sys.exit(1)
    
    # 第二轮：本地1个视频（OpenAI标准格式，base64编码）
    result2 = test_round_2_video_base64(client)
    if result2 is None:
        print("第二轮视频测试失败")
    
    # 第三轮：本地4图图片列表（DashScope扩展格式）
    result3 = test_round_3_local_image_list(client)
    if result3 is None:
        print("第三轮图片列表测试失败")
    
    # 第四轮：URL 4图列表（DashScope扩展格式）
    result4 = test_round_4_url_image_list(client)
    if result4 is None:
        print("第四轮URL图片列表测试失败")
    
    # 第五轮：URL视频（OpenAI标准格式）
    result5 = test_round_5_url_video(client)
    if result5 is None:
        print("第五轮URL视频测试失败")

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
