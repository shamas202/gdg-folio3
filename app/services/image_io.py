from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import UploadFile
from loguru import logger

from app.core.constants import (
    MAX_FILE_SIZE_MB,
    MIN_DIMENSION_UNIVERSAL,
    MAX_DIMENSION,
    CONTRAST_THRESHOLDS,
)


@dataclass
class ImageIOService:
    catalog_images_dir: Path = Path("images/catalog")
    search_images_dir: Path = Path("images/search")
    min_dimension: int = MIN_DIMENSION_UNIVERSAL
    max_dimension: int = MAX_DIMENSION
    max_file_size_mb: int = MAX_FILE_SIZE_MB

    def __post_init__(self) -> None:
        self.catalog_images_dir.mkdir(parents=True, exist_ok=True)
        self.search_images_dir.mkdir(parents=True, exist_ok=True)

    async def read_upload_as_rgb(
        self,
        file: UploadFile,
        category: str | None = None,  # accepted but unused (kept for API compat)
    ) -> Image.Image:
        """
        Read and validate an uploaded image file.

        Checks:
        - File size (≤ max_file_size_mb)
        - Format (JPEG, PNG, WebP)
        - Minimum dimensions
        - Auto-resize if larger than max_dimension
        - Contrast (rejects blank/uniform images)
        """
        data = await file.read()

        size_mb = len(data) / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            raise ValueError(
                f"Image too large ({size_mb:.1f}MB). "
                f"Maximum: {self.max_file_size_mb}MB."
            )

        try:
            img = Image.open(BytesIO(data))
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image: {e}")

        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        if w < self.min_dimension or h < self.min_dimension:
            raise ValueError(
                f"Image too small ({w}×{h}px). Minimum: {self.min_dimension}px."
            )

        if w > self.max_dimension or h > self.max_dimension:
            img.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)
            logger.info(f"Resized image from {w}×{h} to {img.size}")

        # Contrast check — reject blank/uniform images
        img_array = np.array(img)
        std_dev = img_array.std()
        if std_dev < CONTRAST_THRESHOLDS["reject_min"]:
            raise ValueError(
                f"Image rejected: almost blank or uniform (std={std_dev:.2f}). "
                f"Please upload a valid product image."
            )
        if std_dev < CONTRAST_THRESHOLDS["warn_min"]:
            logger.warning(f"Low contrast image (std={std_dev:.1f})")

        logger.debug(f"Image loaded: {img.size}, std={std_dev:.1f}")
        return img

    def pil_to_numpy_rgb(self, img: Image.Image) -> np.ndarray:
        return np.array(img, dtype=np.uint8)

    def get_image_path(self, image_id: str, image_type: str | None = None) -> Path | None:
        if image_type == "search":
            directories = [self.search_images_dir]
        elif image_type == "catalog":
            directories = [self.catalog_images_dir]
        else:
            directories = [self.catalog_images_dir, self.search_images_dir]

        for directory in directories:
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                p = directory / f"{image_id}{ext}"
                if p.exists():
                    return p
        return None
