"""视频加载器模块

提供从URL、Base64等来源加载视频的功能。
"""

from typing import Any, Optional

from .base import BaseLoader, MediaLoadError


class VideoLoadError(MediaLoadError):
    """视频加载失败异常"""
    pass


class VideoLoader(BaseLoader):
    """视频加载器"""
    
    def from_url(self, url: str) -> Any:
        """
        从URL加载视频
        
        Args:
            url: 视频URL
            
        Returns:
            tuple: (video_data, video_metadata)
            
        Raises:
            VideoLoadError: 加载失败时抛出
        """
        try:
            from vllm.multimodal.utils import fetch_video
            video_data, video_meta = fetch_video(url)
            return (video_data, video_meta)
        except ImportError:
            raise VideoLoadError("vllm的fetch_video不可用，请确保vllm版本支持视频处理")
        except Exception as e:
            raise VideoLoadError(f"从URL加载视频失败: {e}")
    
    def from_base64(self, data_url: str, mime_type: str) -> Any:
        """
        从Base64编码的数据加载视频
        
        直接传递data URL给fetch_video，让其自动处理
        
        Args:
            data_url: data:video/xxx;base64, 数据
            mime_type: 期望的MIME类型
            
        Returns:
            tuple: (video_data, video_metadata)
            
        Raises:
            VideoLoadError: 加载失败时抛出
        """
        try:
            from vllm.multimodal.utils import fetch_video
            # 直接传递data URL给fetch_video（与原始版本一致）
            video_data, video_meta = fetch_video(data_url)
            return (video_data, video_meta)
        except Exception as e:
            raise VideoLoadError(f"从Base64加载视频失败: {e}")
    
    def load_with_params(
        self,
        source: str,
        total_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        fps: Optional[int] = None,
    ) -> Any:
        """
        加载视频并附加参数
        
        Args:
            source: 视频来源（URL或Base64数据）
            total_pixels: 总像素数
            min_pixels: 最小像素数
            fps: 帧率
            
        Returns:
            dict: 包含视频数据和元数据的字典
        """
        self.load(source)  # 验证视频可以加载
        
        result = {"url": source if not source.startswith("data:") else "data:..."}
        if total_pixels is not None:
            result["total_pixels"] = str(total_pixels)
        if min_pixels is not None:
            result["min_pixels"] = str(min_pixels)
        if fps is not None:
            result["fps"] = str(fps)
        
        return result
