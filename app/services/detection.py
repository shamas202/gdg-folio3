from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from app.models.domain import Detection


class DetectionService(ABC):
    @abstractmethod
    async def detect(self, img: Image.Image, category_hint: str | None) -> list[Detection]:
        raise NotImplementedError
