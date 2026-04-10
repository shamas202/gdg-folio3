from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image
from loguru import logger

from app.models.domain import Segment
from app.models.schemas import BBox


class SegmentationService(ABC):
    @abstractmethod
    async def segment(self, img: Image.Image, bbox: BBox) -> Segment:
        raise NotImplementedError


@dataclass
class BBoxSegmentationService(SegmentationService):
    """
    Fills the bounding box as a solid rectangle mask.
    Used with YOLO11n (detection-only, no polygon mask).
    """

    async def segment(self, img: Image.Image, bbox: BBox) -> Segment:
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[
            max(0, bbox.y1): min(h, bbox.y2),
            max(0, bbox.x1): min(w, bbox.x2),
        ] = True
        logger.debug(f"BBoxSegmentation: {bbox.x1},{bbox.y1} → {bbox.x2},{bbox.y2}")
        return Segment(
            bbox=(bbox.x1, bbox.y1, bbox.x2, bbox.y2),
            mask=mask,
        )
