from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from loguru import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from app.models.domain import Segment
from app.models.schemas import BBox


class SegmentationService(ABC):
    @abstractmethod
    async def segment(self, img: Image.Image, bbox: BBox) -> Segment:
        """
        Returns a precise object mask aligned to the bbox.
        """
        raise NotImplementedError


@dataclass
class SAM2SegmentationService(SegmentationService):
    """
    SAM2.1-based segmentation using Ultralytics.
    Provides high-quality object masks using bounding box prompts.
    Simpler and more integrated than Facebook's implementation.
    """
    model_path: str = r"D:\image_image_search\backend\app\models\sam2.1_l.pt"
    
    def __post_init__(self) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "torch not installed. Install with: pip install torch torchvision "
                "or use DETECTION_MODE=runpod for RunPod RF-DETR API with polygon segmentation"
            )
        
        try:
            from ultralytics import SAM
        except ImportError:
            raise ImportError(
                "Ultralytics not installed. Install with: pip install ultralytics>=8.3.0 "
                "or use DETECTION_MODE=runpod for RunPod RF-DETR API with polygon segmentation"
            )
        
        # Check if model file exists
        if not Path(self.model_path).exists():
            logger.warning(
                f"SAM2.1 model not found at {self.model_path}. "
                f"Please download from: https://docs.ultralytics.com/models/sam-2/"
            )
            raise FileNotFoundError(f"SAM2.1 model not found: {self.model_path}")
        
        # Determine device (GPU preferred, CPU fallback)
        if torch.cuda.is_available():
            self._device = 'cuda'
            logger.info("SAM2.1 using GPU (CUDA)")
        else:
            self._device = 'cpu'
            logger.info("SAM2.1 using CPU (GPU not available)")
        
        # Load SAM2.1 model from Ultralytics
        self._model = SAM(self.model_path)
        
        # Set device
        self._model.to(self._device)
        
        logger.info(f"SAM2.1 loaded successfully from {self.model_path} on {self._device}")
        
        # Display model info (optional, for debugging)
        try:
            self._model.info()
        except Exception:
            pass  # Some versions might not have info()

    async def segment(self, img: Image.Image, bbox: BBox) -> Segment:
        """
        Segment object using SAM2.1 with bounding box prompt.
        
        Args:
            img: Input image (PIL Image)
            bbox: Bounding box around the object
            
        Returns:
            Segment with high-quality mask
        """
        # Convert PIL to numpy (RGB) for Ultralytics
        img_array = np.array(img)
        
        # Prepare bbox in format [x1, y1, x2, y2]
        bboxes = [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
        
        # Run inference with bboxes prompt
        # Ultralytics SAM expects image array or path, and bboxes parameter
        results = self._model(img_array, bboxes=bboxes, verbose=False)
        
        # Extract mask from results
        # Ultralytics returns results with masks attribute
        if results and len(results) > 0:
            result = results[0]
            
            # Get masks from result
            if hasattr(result, 'masks') and result.masks is not None:
                # Get the first (best) mask
                masks_data = result.masks.data
                
                # Convert to numpy boolean array
                if torch.is_tensor(masks_data):
                    mask = masks_data[0].cpu().numpy().astype(bool)
                else:
                    mask = masks_data[0].astype(bool)
                
                # Calculate coverage and confidence
                coverage = mask.sum() / mask.size * 100
                
                # Get confidence if available
                confidence = 0.0
                if hasattr(result.masks, 'conf') and result.masks.conf is not None:
                    confidence = float(result.masks.conf[0])
                
                logger.debug(
                    f"SAM2.1 segmentation: mask coverage {coverage:.1f}%, "
                    f"confidence {confidence:.3f}"
                )
                
                return Segment(
                    bbox=(bbox.x1, bbox.y1, bbox.x2, bbox.y2),
                    mask=mask,
                )
        
        # Fallback: if segmentation fails, return rectangle mask
        logger.warning("SAM2.1 segmentation failed, falling back to rectangle")
        h, w = img_array.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = True
        
        return Segment(
            bbox=(bbox.x1, bbox.y1, bbox.x2, bbox.y2),
            mask=mask,
        )


@dataclass  
class SAMLikeSegmentationService(SegmentationService):
    """
    Fallback segmentation using GrabCut (OpenCV).
    Used when SAM2 is not available or as backup.
    Better than rectangle but not as good as SAM2.
    """
    
    async def segment(self, img: Image.Image, bbox: BBox) -> Segment:
        import cv2
        
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # Initialize mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Define rectangle for GrabCut
        rect = (bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1)
        
        # GrabCut algorithm (better than rectangle!)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        
        try:
            cv2.grabCut(
                img_array, 
                mask, 
                rect, 
                bgd_model, 
                fgd_model, 
                5,  # iterations
                cv2.GC_INIT_WITH_RECT
            )
            
            # Convert to binary mask (1 and 3 are foreground in GrabCut)
            mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(bool)
            
            logger.debug(f"GrabCut segmentation: mask coverage {mask.sum() / mask.size * 100:.1f}%")
            
        except Exception as e:
            # Fallback to rectangle if GrabCut fails
            logger.warning(f"GrabCut failed: {e}, falling back to rectangle")
            mask = np.zeros((h, w), dtype=bool)
            mask[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = True
        
        return Segment(
            bbox=(bbox.x1, bbox.y1, bbox.x2, bbox.y2),
            mask=mask,
        )