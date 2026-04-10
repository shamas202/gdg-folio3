from __future__ import annotations

import asyncio
import hashlib
from io import BytesIO

import httpx
import numpy as np
from fastapi import APIRouter, Depends
from loguru import logger
from PIL import Image
from pydantic import BaseModel, HttpUrl

from app.core.constants import CONTRAST_THRESHOLDS, MIN_DIMENSION_UNIVERSAL
from app.core.errors import BadRequest
from app.dependencies.container import Container, get_container
from app.models.schemas import BBox, CatalogAddResponse

router = APIRouter()

# ---------------------------------------------------------------------------
# Supported categories (must match detection_yolo.py CATEGORY_TO_YOLO_CLASS)
# ---------------------------------------------------------------------------
SUPPORTED_CATEGORIES = {
    "chair", "couch", "sofa", "bed", "dining-table",
    "tv", "clock", "wall-clock", "vase", "laptop", "tennis-racket",
}


class CatalogAddRequest(BaseModel):
    image_url: HttpUrl
    product_name: str
    category: str


def _load_image_from_bytes(data: bytes) -> Image.Image:
    """Open, convert to RGB, and run basic quality checks."""
    try:
        img = Image.open(BytesIO(data))
    except Exception as e:
        raise BadRequest(f"Invalid or corrupted image: {e}")

    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if w < MIN_DIMENSION_UNIVERSAL or h < MIN_DIMENSION_UNIVERSAL:
        raise BadRequest(
            f"Image too small ({w}×{h}px). Minimum: {MIN_DIMENSION_UNIVERSAL}px on each side."
        )

    std_dev = float(np.array(img).std())
    if std_dev < CONTRAST_THRESHOLDS["reject_min"]:
        raise BadRequest(
            f"Image rejected: almost blank or uniform (std={std_dev:.2f}). "
            "Please provide a valid product image."
        )
    if std_dev < CONTRAST_THRESHOLDS["warn_min"]:
        logger.warning(f"Low contrast image (std={std_dev:.1f})")

    return img


@router.post("/catalog/add", response_model=CatalogAddResponse)
async def add_product(
    body: CatalogAddRequest,
    container: Container = Depends(get_container),
) -> CatalogAddResponse:
    """
    Add a single product to the catalog via its image URL.

    Pipeline:
      1. Download image from image_url
      2. Validate (dimensions, contrast)
      3. YOLO11n detect → pick best bbox matching category
      4. Tight bbox crop (no masking)
      5. Gemini embed (3072-dim)
      6. Upsert to Pinecone (namespace = category)

    pinecone_id = md5(image_url)[:16] — same URL always maps to the same ID,
    making repeated calls idempotent (upsert overwrites the existing vector).
    """
    from app.utils.timing import timed

    image_url = str(body.image_url)
    category = body.category.lower().strip()
    product_name = body.product_name.strip()

    if not product_name:
        raise BadRequest("product_name cannot be empty")

    if category not in SUPPORTED_CATEGORIES:
        raise BadRequest(
            f"Unsupported category '{category}'. "
            f"Supported: {sorted(SUPPORTED_CATEGORIES)}"
        )

    pinecone_id = hashlib.md5(image_url.encode()).hexdigest()[:16]
    logger.info(
        f"Catalog add: pinecone_id={pinecone_id}, "
        f"product='{product_name}', category='{category}', url={image_url}"
    )

    # 1. Download image
    with timed("Image download"):
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.get(image_url)
                resp.raise_for_status()
                image_bytes = resp.content
        except httpx.HTTPStatusError as e:
            raise BadRequest(f"Failed to download image (HTTP {e.response.status_code}): {image_url}")
        except httpx.RequestError as e:
            raise BadRequest(f"Failed to reach image URL: {e}")

    # 2. Load + validate
    with timed("Image load"):
        img = await asyncio.to_thread(_load_image_from_bytes, image_bytes)

    w, h = img.size

    # 3. Detect
    with timed("Detection"):
        detections = await container.detection.detect(img, category_hint=category)

    # Pick best bbox matching category (largest area); fall back to largest overall
    matching = [d for d in detections if d.category == category]
    pool = matching if matching else detections

    det = max(pool, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])) if pool else None

    if det is not None:
        bbox = BBox(x1=det.bbox[0], y1=det.bbox[1], x2=det.bbox[2], y2=det.bbox[3])
        logger.info(
            f"Detected '{det.category}' "
            f"{det.bbox[2]-det.bbox[0]}×{det.bbox[3]-det.bbox[1]}px "
            f"(score={det.score:.2f})"
        )
        try:
            bbox = container.preprocessing.clamp_bbox(bbox, w, h)
        except ValueError as e:
            raise BadRequest(f"Invalid bounding box: {e}")

        with timed("Crop"):
            crop = container.preprocessing.tight_crop(img, bbox)
    else:
        # No detection — use full image as crop (works well for clean product shots)
        logger.warning(f"No '{category}' detected — embedding full image as fallback")
        crop = img

    # 3. Embed
    with timed("Embedding"):
        vector = await asyncio.to_thread(
            container.embedding.embed_crops, {"tight": crop}
        )

    # 4. Upsert to Pinecone
    metadata = {
        "product_name": product_name,
        "category": category,
        "image_url": image_url,
    }

    with timed("Pinecone upsert"):
        await asyncio.to_thread(
            container.vectors.upsert,
            vector_id=pinecone_id,
            vector=vector,
            metadata=metadata,
            namespace=category,
        )

    logger.info(f"Upserted '{product_name}' → namespace '{category}' (id={pinecone_id})")

    detection_note = (
        f"Detected '{det.category}' at {det.score:.0%} confidence."
        if det is not None else
        "No object detected — full image used."
    )

    return CatalogAddResponse(
        pinecone_id=pinecone_id,
        success=True,
        message=f"'{product_name}' added to catalog. {detection_note}",
    )
