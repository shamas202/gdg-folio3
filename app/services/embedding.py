from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class EmbeddingService(ABC):
    @abstractmethod
    async def load(self) -> None: ...

    @abstractmethod
    async def unload(self) -> None: ...

    @abstractmethod
    def embed_crops(self, crops: dict[str, Image.Image]) -> list[float]:
        """Return a single L2-normalized vector for the given crops."""
        raise NotImplementedError


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-12)