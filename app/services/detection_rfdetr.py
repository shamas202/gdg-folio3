from __future__ import annotations

from dataclasses import dataclass
import asyncio
import base64
from io import BytesIO
from typing import Optional, Any

import httpx
from PIL import Image
from loguru import logger

from app.models.domain import Detection


@dataclass
class RFDETRDetectionService:
    """
    RF-DETR detection via local GPU service (GB10).

    Endpoint (server.py on GB10):
      POST /predict
      Body: {"image": "<base64>", "confidence_threshold": 0.25}
      Response: {"status":"success","result":{"detected_objects":[...],...}}

    Concurrency: detect_batch() fires many /predict calls concurrently
    via semaphore (max_concurrency). The GPU box serializes internally;
    multiple uvicorn workers on the GPU box let the kernel overlap I/O.
    """
    api_url: str
    status_url: str = ""
    api_key: str = ""
    confidence_threshold: float = 0.10
    max_wait_seconds: int = 60
    max_concurrency: int = 8
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None

    def _image_to_base64(self, img: Image.Image) -> str:
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
                http2=False,
            )
        return self._client

    async def _close_client(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def detect(self, img: Image.Image, category_hint: str | None = None) -> list[Detection]:
        client = await self._get_client()

        payload = {
            "image": self._image_to_base64(img),
            "confidence_threshold": float(self.confidence_threshold),
        }

        try:
            logger.debug("Sending image to local /predict")
            response = await client.post(self.api_url, json=payload)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict) and "error" in data:
                logger.error(f"Local RF-DETR error: {data['error']}")
                return []

            detections = self._parse_detections(data, category_hint)
            logger.info(f"RF-DETR detected {len(detections)} objects")
            return detections

        except httpx.HTTPStatusError as e:
            logger.error(f"Local RF-DETR HTTP error {e.response.status_code}: {e}")
            return []
        except httpx.RequestError as e:
            logger.error(f"Local RF-DETR request error: {e}")
            return []
        except Exception as e:
            logger.error(f"Local RF-DETR unexpected error: {e}")
            return []

    async def detect_batch(
        self,
        images: list[Image.Image],
        category_hints: list[str | None] | None = None,
    ) -> list[list[Detection]]:
        if category_hints is None:
            category_hints = [None] * len(images)
        if len(images) != len(category_hints):
            raise ValueError(
                f"Images ({len(images)}) and category_hints ({len(category_hints)}) must have same length"
            )
        if not images:
            return []

        sem = asyncio.Semaphore(self.max_concurrency)

        async def _one(img: Image.Image, hint: str | None) -> list[Detection]:
            async with sem:
                try:
                    return await self.detect(img, hint)
                except Exception as e:
                    logger.error(f"Batch detect failed: {e}")
                    return []

        logger.info(f"Processing {len(images)} images (max_concurrency={self.max_concurrency})")
        tasks = [_one(img, hint) for img, hint in zip(images, category_hints, strict=False)]
        return await asyncio.gather(*tasks)

    def _parse_detections(self, data: Any, category_hint: str | None) -> list[Detection]:
        if not isinstance(data, dict):
            return []

        result = data.get("result", {})
        if not isinstance(result, dict):
            return []

        detected_objects = result.get("detected_objects", [])
        detections: list[Detection] = []

        for det in detected_objects:
            if not isinstance(det, dict):
                continue

            conf = float(det.get("confidence", 0.0))
            if conf < self.confidence_threshold:
                continue

            bbox = det.get("bbox_from_mask", [0, 0, 0, 0])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            detections.append(
                Detection(
                    category=str(det.get("label", "unknown")).lower().strip(),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    score=conf,
                    mask_polygon=det.get("mask_polygon"),
                )
            )

        if category_hint:
            hint = category_hint.lower().strip()
            matching = [d for d in detections if d.category == hint]
            if matching:
                detections = matching

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections
