from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from loguru import logger

from app.models.domain import Detection
from app.services.detection import DetectionService


# ---------------------------------------------------------------------------
# YOLO COCO class name → our internal category name
# YOLO11n returns class names directly from model.names (e.g. "chair", "couch").
# Extend this map when supporting more categories.
# ---------------------------------------------------------------------------
YOLO_CLASS_TO_CATEGORY: dict[str, str] = {
    "chair": "chair",
    "couch": "couch",       # maps to both "couch" and "sofa"
    "bed": "bed",
    "dining table": "dining-table",
    "tv": "tv",
    "clock": "wall-clock",
    "vase": "vase",
    "laptop": "laptop",
    "tennis racket": "tennis-racket",
}

# Reverse: our category name → YOLO class name (used for category_hint filtering)
CATEGORY_TO_YOLO_CLASS: dict[str, str] = {
    "chair": "chair",
    "couch": "couch",
    "sofa": "couch",
    "bed": "bed",
    "dining-table": "dining table",
    "tv": "tv",
    "wall-clock": "clock",
    "clock": "clock",
    "vase": "vase",
    "laptop": "laptop",
    "tennis-racket": "tennis racket",
}


@dataclass
class YOLODetectionService(DetectionService):
    """
    YOLO11n-based object detection (CPU-friendly, no external server needed).

    Downloads yolo11n.pt (~2.6MB) on first use and caches it locally.
    Returns Detection objects with mask_polygon=None (no segmentation).

    Replaces RF-DETR HTTP endpoint for local / demo deployments.
    """
    model_name: str = "yolo11n.pt"
    confidence_threshold: float = 0.15
    _model: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
            logger.success(
                f"YOLODetectionService ready — model: {self.model_name}, "
                f"conf_threshold: {self.confidence_threshold}"
            )
        except ImportError:
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install ultralytics>=8.3.0"
            )

    async def detect(
        self,
        img: Image.Image,
        category_hint: str | None = None,
    ) -> list[Detection]:
        """
        Run YOLO11n inference on img and return matching detections.

        Args:
            img: PIL RGB image.
            category_hint: Optional category name (our internal name).
                           If provided, only detections matching this category
                           are returned. Otherwise all supported classes returned.

        Returns:
            List of Detection objects sorted by score desc, mask_polygon=None.
        """
        img_array = np.array(img)
        results = self._model(img_array, verbose=False)

        # Resolve which YOLO class name to filter for (if hint given)
        target_yolo_class: str | None = None
        if category_hint:
            hint_lower = category_hint.lower().strip()
            target_yolo_class = CATEGORY_TO_YOLO_CLASS.get(hint_lower)
            if target_yolo_class is None:
                logger.warning(
                    f"YOLODetectionService: unknown category_hint '{category_hint}', "
                    f"will return all detections"
                )

        dets: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0].item())
                if conf < self.confidence_threshold:
                    continue

                cls_id = int(box.cls[0].item())
                yolo_class = self._model.names[cls_id].lower()

                # Apply category hint filter
                if target_yolo_class and yolo_class != target_yolo_class.lower():
                    continue

                # Map YOLO class → our category name (skip unmapped classes)
                category = YOLO_CLASS_TO_CATEGORY.get(yolo_class)
                if category is None:
                    continue

                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                dets.append(
                    Detection(
                        category=category,
                        bbox=(x1, y1, x2, y2),
                        score=conf,
                        mask_polygon=None,  # YOLO11n detection only — no segmentation
                    )
                )

        dets.sort(key=lambda d: d.score, reverse=True)
        logger.info(
            f"YOLO detected {len(dets)} object(s)"
            + (f" matching hint='{category_hint}'" if category_hint else "")
        )
        return dets
