"""多模态数据处理器

处理图像和视频数据的聚合和转换。
"""

from typing import Optional, Dict, Any, List, Tuple

from .image_loader import ImageLoader, ImageLoadError
from .video_loader import VideoLoader, VideoLoadError


class MultiModalProcessor:
    """多模态数据处理器"""
    
    def __init__(self):
        """初始化处理器"""
        self.image_loader = ImageLoader()
        self.video_loader = VideoLoader()
    
    def process_images(
        self,
        image_sources: List[str],
    ) -> List[Any]:
        """
        批量加载图像
        
        Args:
            image_sources: 图像来源列表（URL或Base64）
            
        Returns:
            List[Any]: 加载的图像对象列表
        """
        images = []
        for source in image_sources:
            try:
                image = self.image_loader.load(source)
                images.append(image)
            except ImageLoadError as e:
                print(f"[WARN] Image load failed: {e}")
                continue
        return images
    
    def process_videos(
        self,
        video_sources: List[str],
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        """
        批量加载视频
        
        Args:
            video_sources: 视频来源列表（URL或Base64）
            
        Returns:
            List[Tuple[Any, Dict]]: 加载的视频数据列表，每个元素为(data, metadata)
        """
        videos = []
        for source in video_sources:
            try:
                video_data, video_meta = self.video_loader.load(source)
                videos.append((video_data, video_meta))
            except VideoLoadError as e:
                print(f"[WARN] Video load failed: {e}")
                continue
        return videos
    
    def build_multimodal_data(
        self,
        images: Optional[List[str]] = None,
        videos: Optional[List[Dict[str, Any]]] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        构建多模态数据字典

        Args:
            images: 图像来源列表
            videos: 视频字典列表（包含url或其他参数）
            mm_processor_kwargs: 多模态处理器参数（如 fps）

        Returns:
            Optional[Dict]: 多模态数据字典，如果没有数据则返回None
        """
        mm_data = {}

        # 处理图像
        if images:
            loaded_images = self.process_images(images)
            if loaded_images:
                mm_data["image"] = loaded_images[0] if len(loaded_images) == 1 else loaded_images

        # 处理视频
        if videos:
            loaded_videos = []
            for video_item in videos:
                if isinstance(video_item, str):
                    video_url = video_item
                    video_kwargs = mm_processor_kwargs.copy() if mm_processor_kwargs else {}
                elif isinstance(video_item, dict):
                    video_url = video_item.get("url", "")
                    video_kwargs = mm_processor_kwargs.copy() if mm_processor_kwargs else {}
                    for key in ["fps", "total_pixels", "min_pixels"]:
                        if key in video_item:
                            video_kwargs[key] = video_item[key]
                else:
                    continue

                if not video_url:
                    continue

                try:
                    # 处理视频帧列表（URL列表）情况
                    if isinstance(video_url, list) and video_url:
                        # 加载每个图片URL
                        frame_images = self.process_images(video_url)
                        if frame_images:
                            # 转换为numpy数组并构建metadata
                            import numpy as np
                            frames = np.stack([np.asarray(img) for img in frame_images])
                            total = int(frames.shape[0])
                            fps = float(video_kwargs.get("fps", 1))
                            duration = total / fps if fps > 0 else 0.0
                            metadata = {
                                "total_num_frames": total,
                                "fps": fps,
                                "duration": duration,
                                "video_backend": "jpeg_sequence",
                                "frames_indices": list(range(total)),
                                "do_sample_frames": False,
                            }
                            loaded_videos.append((frames, metadata))
                    else:
                        # 处理普通视频URL（字符串）
                        if video_kwargs:
                            video_data, video_meta = self.video_loader.load(
                                video_url,
                                **video_kwargs
                            )
                        else:
                            video_data, video_meta = self.video_loader.load(video_url)
                        loaded_videos.append((video_data, video_meta))
                except VideoLoadError as e:
                    print(f"[WARN] Video load failed: {e}")
                    continue

            if loaded_videos:
                mm_data["video"] = loaded_videos[0] if len(loaded_videos) == 1 else loaded_videos

        return mm_data if mm_data else None
    
    def process_for_engine(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        videos: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """
        为Engine构建输入
        
        Args:
            prompt: 文本提示
            images: 图像来源列表
            videos: 视频字典列表
            
        Returns:
            Any: 纯文本提示或TextPrompt字典
        """
        mm_data = self.build_multimodal_data(images, videos)
        
        if mm_data:
            from vllm.inputs.data import TextPrompt
            return TextPrompt(
                prompt=prompt,
                multi_modal_data=mm_data,
            )
        else:
            return prompt
