#!/usr/bin/env python3
"""
vLLM 多模态推理客户端并发测试示例 - 完整版，含多维性能指标
"""

import base64
import os
import sys
import time
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============ 配置项，请根据实际调整 =============
BASE_URL = "http://36.155.152.28:31002/v1"
API_KEY = "token-abc123"
MODEL_NAME = "qwen-3-vl"

IMAGE_PATHS = [
    "./test_image1.jpg",
    "./test_image2.jpg",
    "./test_image3.jpg",
    "./test_image4.jpg",
]

VIDEO_PATHS = [
    "./test_video1.mp4",
    "./test_video2.mp4",
]

LIMIT_MM_PER_PROMPT = {"image": 4, "video": 2}

TEMPERATURE = 0.7
CONCURRENCY = 32
# ===================================================


def encode_file_to_base64(file_path: str) -> str:
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
    images = []
    for path in image_paths:
        try:
            images.append(encode_file_to_base64(path))
        except FileNotFoundError:
            print(f"图片文件不存在: {path}")
            return None
    return images


def get_videos_base64(video_paths: list) -> list:
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
    # 对中文可直接用字符数或者用专门Tokenizer替代
    return max(1, int(len(text) / 1.5))


def inference_4_images(client, image_paths):
    start_time = time.perf_counter()
    images = get_images_base64(image_paths)
    if images is None:
        return False, "图片文件不存在", 0, 0, 0

    content = [{"type": "text", "text": "请详细描述这4张图片的内容，包括每张图片的细节和特点。"}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": img}})
    messages = [{"role": "user", "content": content}]
    try:
        # 同步接口近似首字时间 = 全响应时间
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
        )
        total_time = time.perf_counter() - start_time
        first_byte_time = total_time  # 近似
        text = response.choices[0].message.content
        token_count = estimate_tokens(text)
        token_per_sec = token_count / total_time if total_time > 0 else 0
        return True, text, total_time, first_byte_time, token_per_sec
    except Exception as e:
        total_time = time.perf_counter() - start_time
        return False, str(e), total_time, 0, 0


def inference_3_images_1_video(client, image_paths, video_path):
    start_time = time.perf_counter()
    images = get_images_base64(image_paths)
    if images is None:
        return False, "图片文件不存在", 0, 0, 0
    videos = get_videos_base64([video_path])
    if videos is None:
        return False, "视频文件不存在", 0, 0, 0

    content = [{"type": "text", "text": "请详细描述这些图片和视频的内容。"}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": img}})
    content.append({"type": "video_url", "video_url": {"url": videos[0]}})
    messages = [{"role": "user", "content": content}]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
        )
        total_time = time.perf_counter() - start_time
        first_byte_time = total_time  # 近似
        text = response.choices[0].message.content
        token_count = estimate_tokens(text)
        token_per_sec = token_count / total_time if total_time > 0 else 0
        return True, text, total_time, first_byte_time, token_per_sec
    except Exception as e:
        total_time = time.perf_counter() - start_time
        return False, str(e), total_time, 0, 0


def run_concurrent_tasks(executor, futures):
    total_times = []
    first_byte_times = []
    token_speeds = []
    success_count = 0
    failure_count = 0
    failure_reasons = {}

    for future in as_completed(futures):
        try:
            success, msg, full_time, first_byte_time, tokens_per_sec = future.result()
            if full_time > 0:
                total_times.append(full_time)
            if first_byte_time > 0:
                first_byte_times.append(first_byte_time)
            if tokens_per_sec > 0:
                token_speeds.append(tokens_per_sec)
            if success:
                success_count += 1
            else:
                failure_count += 1
                failure_reasons[msg] = failure_reasons.get(msg, 0) + 1
        except Exception as e:
            failure_count += 1
            failure_reasons[f"异常: {e}"] = failure_reasons.get(f"异常: {e}", 0) + 1

    return {
        "total_times": total_times,
        "first_byte_times": first_byte_times,
        "token_speeds": token_speeds,
        "success": success_count,
        "failure": failure_count,
        "failure_reasons": failure_reasons,
    }


def percentile(arr, pct):
    if not arr:
        return 0
    k = int(len(arr) * pct)
    s = sorted(arr)
    return s[min(k, len(s) - 1)]


def print_report(title, stats, total_duration):
    print("\n" + "=" * 80)
    print(f"{title} 性能指标汇总")
    print("=" * 80)

    total_requests = stats["success"] + stats["failure"]
    print(f"请求总数: {total_requests}")
    print(f"成功数: {stats['success']}")
    print(f"失败数: {stats['failure']}")
    print(f"总测试耗时: {total_duration:.3f} 秒")

    if stats["total_times"]:
        print(f"响应时间（秒）:")
        print(f"  最小: {min(stats['total_times']):.3f}")
        print(f"  最大: {max(stats['total_times']):.3f}")
        print(f"  平均: {statistics.mean(stats['total_times']):.3f}")
        print(f"  P90: {percentile(stats['total_times'], 0.9):.3f}")
        print(f"  P99: {percentile(stats['total_times'], 0.99):.3f}")
    else:
        print("无响应时间数据")

    if stats["first_byte_times"]:
        print(f"首字时间（秒）:")
        print(f"  最小: {min(stats['first_byte_times']):.3f}")
        print(f"  最大: {max(stats['first_byte_times']):.3f}")
        print(f"  平均: {statistics.mean(stats['first_byte_times']):.3f}")
    else:
        print("无首字时间数据")

    if stats["token_speeds"]:
        print(f"Token生成速率（tokens/秒）:")
        print(f"  最小: {min(stats['token_speeds']):.1f}")
        print(f"  最大: {max(stats['token_speeds']):.1f}")
        print(f"  平均: {statistics.mean(stats['token_speeds']):.1f}")
    else:
        print("无Token速率数据")

    # 计算整体吞吐量QPS = 总请求数 / 总测试耗时
    qps = total_requests / total_duration if total_duration > 0 else 0
    print(f"整体吞吐量(QPS): {qps:.2f}")

    if stats['failure'] > 0:
        print("\n失败原因分布:")
        for reason, count in stats['failure_reasons'].items():
            print(f"  {count} 次: {reason}")

def main():
    from openai import OpenAI

    print("="*60)
    print("vLLM 多模态推理客户端并发测试 (含多维性能指标)")
    print("="*60)
    print(f"服务器地址: {BASE_URL}")
    print(f"API Key: {API_KEY}")
    print(f"模型名称: {MODEL_NAME}")
    print(f"多模态限制: {LIMIT_MM_PER_PROMPT}")
    print(f"温度参数: {TEMPERATURE}")
    print(f"并发数量: {CONCURRENCY}")
    print(f"\n图片文件 ({len(IMAGE_PATHS)} 张):")
    for i, p in enumerate(IMAGE_PATHS):
        print(f"  {i+1}. {p}")
    print(f"\n视频文件 ({len(VIDEO_PATHS)} 个):")
    for i, p in enumerate(VIDEO_PATHS):
        print(f"  {i+1}. {p}")

    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )

    try:
        _ = client.models.list()
        print("\n服务器连接成功")
    except Exception as e:
        print(f"\n无法连接到服务器: {e}")
        print("请确保vLLM服务正在运行!")
        sys.exit(1)

    import time
    from concurrent.futures import ThreadPoolExecutor

    print("\n" + "="*60)
    print("第一轮测试：4张图片，32并发")
    print("="*60)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(inference_4_images, client, IMAGE_PATHS) for _ in range(CONCURRENCY)]
        stats1 = run_concurrent_tasks(executor, futures)
    duration1 = time.perf_counter() - start
    print_report("第一轮4图片测试", stats1, duration1)

    print("\n" + "="*60)
    print("第二轮测试：3张图片 + 1个视频，32并发")
    print("="*60)

    three_images = IMAGE_PATHS[:3]
    video = VIDEO_PATHS[0]

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(inference_3_images_1_video, client, three_images, video) for _ in range(CONCURRENCY)]
        stats2 = run_concurrent_tasks(executor, futures)
    duration2 = time.perf_counter() - start
    print_report("第二轮3图片+1视频测试", stats2, duration2)

    print("\n" + "="*60)
    print("整体测试总结")
    print("="*60)
    total_reqs = stats1['success'] + stats1['failure'] + stats2['success'] + stats2['failure']
    total_succ = stats1['success'] + stats2['success']
    total_fail = stats1['failure'] + stats2['failure']
    total_duration = duration1 + duration2
    print(f"总请求数: {total_reqs}")
    print(f"成功数: {total_succ}")
    print(f"失败数: {total_fail}")
    print(f"总耗时: {total_duration:.3f} 秒")
    print(f"整体吞吐量(QPS): {total_reqs / total_duration:.2f}")

    if total_fail > 0:
        print("\n失败原因汇总（示例）：")
        fr_all = {}
        for fdict in [stats1['failure_reasons'], stats2['failure_reasons']]:
            for k, v in fdict.items():
                fr_all[k] = fr_all.get(k, 0) + v
        for reason, count in fr_all.items():
            print(f"  {count} 次: {reason}")

if __name__ == "__main__":
    main()