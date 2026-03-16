"""多模态加载器基类

定义图像和视频加载器的公共接口。
"""

from abc import ABC, abstractmethod
from typing import Any
from io import BytesIO


class MediaLoadError(Exception):
    """媒体加载失败异常"""
    pass


class BaseLoader(ABC):
    """媒体加载器基类"""
    
    @abstractmethod
    def from_url(self, url: str) -> Any:
        """
        从URL加载媒体
        
        Args:
            url: 媒体URL
            
        Returns:
            加载的媒体对象
            
        Raises:
            MediaLoadError: 加载失败时抛出
        """
        pass
    
    @abstractmethod
    def from_base64(self, data_url: str, mime_type: str) -> Any:
        """
        从Base64编码的数据加载媒体
        
        Args:
            data_url: data: MIME类型;base64, 数据
            mime_type: 期望的MIME类型
            
        Returns:
            加载的媒体对象
            
        Raises:
            MediaLoadError: 加载失败时抛出
        """
        pass
    
    def load(self, source: str, expected_mime: str = "") -> Any:
        """
        自动检测并加载媒体
        
        Args:
            source: 媒体来源（URL或Base64数据）
            expected_mime: 期望的MIME类型
            
        Returns:
            加载的媒体对象
            
        Raises:
            MediaLoadError: 加载失败时抛出
        """
        if source.startswith("data:"):
            # Base64编码的数据
            if not expected_mime:
                # 从数据URL中提取MIME类型
                expected_mime = source.split(":")[1].split(";")[0]
            return self.from_base64(source, expected_mime)
        else:
            # URL
            return self.from_url(source)
    
    @staticmethod
    def _decode_base64(data_url: str) -> BytesIO:
        """
        解码Base64数据
        
        Args:
            data_url: data: MIME类型;base64, 数据
            
        Returns:
            BytesIO: 解码后的字节数据
        """
        import base64
        
        # 提取Base64数据部分
        if "," in data_url:
            base64_data = data_url.split(",", 1)[1]
        else:
            base64_data = data_url
        
        # 解码
        decoded = base64.b64decode(base64_data)
        return BytesIO(decoded)
