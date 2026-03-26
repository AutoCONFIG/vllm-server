"""视频加载器模块

提供从URL、Base64等来源加载视频的功能。
"""

from typing import Any, Optional, Tuple, Dict

from .base import BaseLoader, MediaLoadError


class VideoLoadError(MediaLoadError):
    """视频加载失败异常"""
    pass


class VideoLoader(BaseLoader):
    """视频加载器"""

    def load(
        self,
        source: str,
        fps: Optional[int] = None,
        total_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        加载视频

        Args:
            source: 视频来源（URL或Base64数据）
            fps: 帧率
            total_pixels: 总像素数
            min_pixels: 最小像素数

        Returns:
            tuple: (video_data, video_metadata)

        Raises:
            VideoLoadError: 加载失败时抛出
        """
        try:
            from vllm.multimodal.utils import fetch_video

            # 构建 kwargs 参数
            fetch_kwargs = {}
            if fps is not None:
                fetch_kwargs["fps"] = fps
            if total_pixels is not None:
                fetch_kwargs["total_pixels"] = total_pixels
            if min_pixels is not None:
                fetch_kwargs["min_pixels"] = min_pixels

            if fetch_kwargs:
                video_data, video_meta = fetch_video(source, **fetch_kwargs)
            else:
                video_data, video_meta = fetch_video(source)

            return (video_data, video_meta)
        except ImportError:
            raise VideoLoadError("vllm的fetch_video不可用，请确保vllm版本支持视频处理")
        except Exception as e:
            raise VideoLoadError(f"从URL加载视频失败: {e}")
