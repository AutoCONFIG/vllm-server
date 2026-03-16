"""图像加载器模块

提供从URL、Base64等来源加载图像的功能。
"""

from typing import Any
from io import BytesIO

from PIL import Image

from .base import BaseLoader, MediaLoadError


class ImageLoadError(MediaLoadError):
    """图像加载失败异常"""
    pass


class ImageLoader(BaseLoader):
    """图像加载器"""
    
    def from_url(self, url: str) -> Image.Image:
        """
        从URL加载图像
        
        Args:
            url: 图像URL
            
        Returns:
            PIL.Image: 加载的图像对象
            
        Raises:
            ImageLoadError: 加载失败时抛出
        """
        try:
            # 尝试使用vllm的fetch_image
            from vllm.multimodal.utils import fetch_image
            return fetch_image(url)
        except ImportError:
            # vllm工具不可用，使用requests
            try:
                import requests
                response = requests.get(url, timeout=30)
                return Image.open(BytesIO(response.content))
            except Exception as e:
                raise ImageLoadError(f"从URL加载图像失败: {e}")
        except Exception as e:
            raise ImageLoadError(f"从URL加载图像失败: {e}")
    
    def from_base64(self, data_url: str, mime_type: str) -> Image.Image:
        """
        从Base64编码的数据加载图像
        
        Args:
            data_url: data:image/xxx;base64, 数据
            mime_type: 期望的MIME类型（忽略）
            
        Returns:
            PIL.Image: 加载的图像对象
            
        Raises:
            ImageLoadError: 加载失败时抛出
        """
        try:
            bytes_io = self._decode_base64(data_url)
            image = Image.open(bytes_io)
            return image
        except Exception as e:
            raise ImageLoadError(f"从Base64加载图像失败: {e}")
    
    def from_bytes(self, data: bytes) -> Image.Image:
        """
        从字节数据加载图像
        
        Args:
            data: 图像字节数据
            
        Returns:
            PIL.Image: 加载的图像对象
            
        Raises:
            ImageLoadError: 加载失败时抛出
        """
        try:
            return Image.open(BytesIO(data))
        except Exception as e:
            raise ImageLoadError(f"从字节数据加载图像失败: {e}")
