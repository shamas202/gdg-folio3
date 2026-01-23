from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from ultralytics import RTDETR
    RTDETR_AVAILABLE = True
except ImportError:
    RTDETR_AVAILABLE = False
    RTDETR = None

from app.models.domain import Detection


class DetectionService(ABC):
    @abstractmethod
    async def detect(self, img: Image.Image, category_hint: str | None) -> list[Detection]:
        raise NotImplementedError


@dataclass
class RTDETRDetectionService(DetectionService):
    """
    RT-DETR (Real-Time Detection Transformer) detector for product detection.
    Uses ultralytics RT-DETR model for accurate object detection.
    """
    model_path: str
    score_threshold: float = 0.30

    def __post_init__(self) -> None:
        # Load RT-DETR model
        if not RTDETR_AVAILABLE:
            raise RuntimeError(
                "ultralytics not installed. Install with: pip install ultralytics>=8.3.0 "
                "or use DETECTION_MODE=runpod for RunPod RF-DETR API"
            )
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"RT-DETR model not found at {self.model_path}")
        
        self._model = RTDETR(self.model_path)
        
        # COCO label map (minimal; expand as needed)
        # RT-DETR uses COCO classes, mapping class IDs to category names
        self._coco = {
            59: "bed",
            61: "dining table",
            62: "toilet",
            63: "tv",
            56: "chair",
            57: "couch",
        }

    async def detect(self, img: Image.Image, category_hint: str | None) -> list[Detection]:
        # Convert PIL Image to numpy array for ultralytics
        img_array = np.array(img)
        
        # Run inference
        results = self._model(img_array, verbose=False)
        
        dets: list[Detection] = []
        
        # Process results (RT-DETR returns results similar to YOLO)
        if results and len(results) > 0:
            result = results[0]
            
            # Get boxes, scores, and class IDs
            boxes = result.boxes.xyxy.cpu().numpy()  # xyxy format
            scores = result.boxes.conf.cpu().numpy()  # confidence scores
            labels = result.boxes.cls.cpu().numpy().astype(int)  # class IDs
            
            for box, score, label in zip(boxes, scores, labels, strict=False):
                if float(score) < self.score_threshold:
                    continue
                
                # Map COCO class ID to category name
                cat = self._coco.get(int(label), "unknown")
                
                # Filter by category hint if provided
                if category_hint and cat != category_hint:
                    continue
                
                # Convert box from xyxy to (x1, y1, x2, y2) format
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                dets.append(Detection(category=cat, bbox=(x1, y1, x2, y2), score=float(score)))

        dets.sort(key=lambda d: d.score, reverse=True)
        return dets


@dataclass
class TorchvisionDetectionService(DetectionService):
    """
    Legacy Faster R-CNN detector (kept for backward compatibility).
    Use RTDETRDetectionService for better product detection.
    """
    score_threshold: float = 0.30

    def __post_init__(self) -> None:
        import torch
        import torchvision
        
        self._model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        self._model.eval()

        # COCO label map (minimal; expand as needed)
        self._coco = {
            59: "bed",
            61: "dining table",
            62: "toilet",
            63: "tv",
            56: "chair",
            57: "couch",
        }

    async def detect(self, img: Image.Image, category_hint: str | None) -> list[Detection]:
        import torch
        import torchvision
        
        t = torchvision.transforms.ToTensor()(img)
        with torch.no_grad():
            out = self._model([t])[0]

        dets: list[Detection] = []
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels, strict=False):
            if float(score) < self.score_threshold:
                continue
            cat = self._coco.get(int(label), "unknown")
            if category_hint and cat != category_hint:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            dets.append(Detection(category=cat, bbox=(x1, y1, x2, y2), score=float(score)))

        dets.sort(key=lambda d: d.score, reverse=True)
        return dets