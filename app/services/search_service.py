from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from fastapi import UploadFile
from loguru import logger

from app.core.errors import BadRequest
from app.core.config import Settings
from app.models.schemas import BBox, SearchHit, SearchResponse
from app.services.image_io import ImageIOService
from app.services.preprocessing import PreprocessingService
from app.services.detection import DetectionService
from app.services.embedding import EmbeddingService
from app.repositories.pinecone_repo import PineconeVectorRepository


@dataclass
class SearchService:
    settings: Settings
    image_io: ImageIOService
    preprocessing: PreprocessingService
    detection: DetectionService
    embedding: EmbeddingService
    vectors: PineconeVectorRepository

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_best_detection(
        self,
        detections: list,
        category_hint: str | None,
    ):
        """Return the best detection: matching category (largest bbox) or overall largest."""
        if not detections:
            return None

        if category_hint:
            matching = [d for d in detections if d.category == category_hint.lower().strip()]
            pool = matching if matching else detections
            if not matching:
                available = list({d.category for d in detections})
                logger.warning(
                    f"No '{category_hint}' detected. Found: {available}. "
                    f"Using largest object instead."
                )
        else:
            pool = detections

        return max(pool, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))

    def _build_hits(self, candidates: list[dict[str, Any]], top_k: int) -> list[SearchHit]:
        """Deduplicate candidates and build SearchHit list."""
        seen: dict[str, dict] = {}
        for c in candidates:
            pid = c.get("id")
            if not pid:
                continue
            if pid not in seen or c["score"] > seen[pid]["score"]:
                seen[pid] = c

        top = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        hits = []
        for c in top:
            m = c.get("metadata", {})
            hits.append(
                SearchHit(
                    pinecone_id=c["id"],
                    score=float(c["score"]),
                    image_url=m.get("image_url") or "",
                    product_name=m.get("product_name") or "",
                    name_english=m.get("name_english") or "",
                    name_arabic=m.get("name_arabic") or "",
                    category=m.get("category") or "",
                )
            )
        return hits

    # ------------------------------------------------------------------
    # Search with pre-supplied bbox (fast path — user clicked a detection)
    # ------------------------------------------------------------------

    async def search_with_bbox(
        self,
        *,
        room_image_file: UploadFile,
        bbox: BBox,
        category: str | None,
        top_k: int,
        mask_polygon: list | None = None,  # accepted but ignored (no masking)
    ) -> SearchResponse:
        from app.utils.timing import timed

        logger.info(
            f"Fast search — bbox: ({bbox.x1},{bbox.y1},{bbox.x2},{bbox.y2}), "
            f"category: {category}, top_k: {top_k}"
        )

        with timed("Image load"):
            img = await self.image_io.read_upload_as_rgb(room_image_file)

        w, h = img.size
        query_category = category.lower().strip() if category else None

        try:
            bbox = self.preprocessing.clamp_bbox(bbox, w, h)
        except ValueError as e:
            raise BadRequest(f"Invalid bounding box: {e}")

        with timed("Crop"):
            crop = self.preprocessing.tight_crop(img, bbox)

        with timed("Embedding"):
            query_vector = await asyncio.to_thread(
                self.embedding.embed_crops, {"tight": crop}
            )

        with timed("Vector search"):
            candidates = await asyncio.to_thread(
                self.vectors.query,
                vector=query_vector,
                top_k=top_k,
                category=query_category,
            )

        logger.info(f"Retrieved {len(candidates)} candidates")

        if not candidates:
            msg = "No products found in the catalog"
            if query_category:
                msg += f" for category '{query_category}'"
            return SearchResponse(query_category=query_category, hits=[], message=msg)

        return SearchResponse(
            query_category=query_category,
            hits=self._build_hits(candidates, top_k),
        )

    # ------------------------------------------------------------------
    # Full search (detect first, then search)
    # ------------------------------------------------------------------

    async def search_room_image(
        self,
        *,
        room_image_file: UploadFile,
        assigned_category: str | None,
        top_k: int,
    ) -> SearchResponse:
        from app.utils.timing import timed

        logger.info(f"Full search — category: {assigned_category}, top_k: {top_k}")

        with timed("Image load"):
            img = await self.image_io.read_upload_as_rgb(room_image_file)

        w, h = img.size

        with timed("Detection"):
            detections = await self.detection.detect(
                img,
                category_hint=assigned_category.lower().strip() if assigned_category else None,
            )

        logger.info(f"Detected {len(detections)} objects")

        if not detections:
            raise BadRequest(
                "No objects detected in the image. "
                "Please ensure the image shows a clear view of the product."
            )

        det = self._pick_best_detection(detections, assigned_category)
        query_category = det.category.lower().strip()
        bbox = BBox(x1=det.bbox[0], y1=det.bbox[1], x2=det.bbox[2], y2=det.bbox[3])

        logger.info(
            f"Selected: '{det.category}' {det.bbox[2]-det.bbox[0]}×"
            f"{det.bbox[3]-det.bbox[1]}px (score: {det.score:.3f})"
        )

        try:
            bbox = self.preprocessing.clamp_bbox(bbox, w, h)
        except ValueError as e:
            raise BadRequest(f"Invalid bounding box: {e}")

        with timed("Crop"):
            crop = self.preprocessing.tight_crop(img, bbox)

        with timed("Embedding"):
            query_vector = await asyncio.to_thread(
                self.embedding.embed_crops, {"tight": crop}
            )

        with timed("Vector search"):
            candidates = await asyncio.to_thread(
                self.vectors.query,
                vector=query_vector,
                top_k=top_k,
                category=query_category,
            )

        logger.info(f"Retrieved {len(candidates)} candidates from namespace '{query_category}'")

        if not candidates:
            msg = f"No products found for '{query_category}'. Add products to catalog first."
            return SearchResponse(query_category=query_category, hits=[], message=msg)

        return SearchResponse(
            query_category=query_category,
            hits=self._build_hits(candidates, top_k),
        )
