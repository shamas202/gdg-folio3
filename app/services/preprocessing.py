from __future__ import annotations

from dataclasses import dataclass

from PIL import Image
from loguru import logger

from app.models.schemas import BBox


@dataclass
class PreprocessingService:
    """
    Image preprocessing — tight bbox crop only.
    No masking, no padding, no medium crop.
    """

    def clamp_bbox(self, bbox: BBox, w: int, h: int) -> BBox:
        x1 = max(0, min(w - 1, bbox.x1))
        y1 = max(0, min(h - 1, bbox.y1))
        x2 = max(0, min(w, bbox.x2))
        y2 = max(0, min(h, bbox.y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bbox after clamping")
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def tight_crop(self, img: Image.Image, bbox: BBox) -> Image.Image:
        """Exact bbox crop — no padding, no mask."""
        crop = img.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
        logger.debug(
            f"Tight crop: {bbox.x2 - bbox.x1}×{bbox.y2 - bbox.y1}px"
        )
        return crop
