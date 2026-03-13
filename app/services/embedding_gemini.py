from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
from PIL import Image
from loguru import logger

from app.services.embedding import EmbeddingService


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-12
    return x / n


def _image_to_base64_jpeg(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@dataclass
class GeminiEmbeddingService(EmbeddingService):
    """
    Image embedding service using Google's gemini-embedding-2-preview model.

    Replaces the dual-tower ViT + CLIP setup with a single multimodal API call.

    Per object:
      - Sends tight crop  → Gemini API → 3072-dim vector
      - Sends medium crop → Gemini API → 3072-dim vector
      - Averages and L2-normalizes the two vectors

    Retry logic handles transient API rate-limit / server errors.
    """

    api_key: str
    model: str = "models/gemini-embedding-2-preview"
    output_dimensionality: int = 3072
    max_retries: int = 4
    initial_backoff: float = 1.0

    _client: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._loaded = False
        self._client = None

    async def load(self) -> None:
        if self._loaded:
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._genai = genai
            self._loaded = True
            logger.success(
                f"GeminiEmbeddingService ready — model: {self.model}, "
                f"dim: {self.output_dimensionality}"
            )
        except ImportError:
            raise RuntimeError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai"
            )

    async def unload(self) -> None:
        self._loaded = False
        self._client = None
        logger.info("GeminiEmbeddingService unloaded")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed_crops(self, crops: dict[str, Image.Image]) -> list[float]:
        """
        Embed tight + medium crops and return the averaged 3072-dim vector.

        Called via asyncio.to_thread() from the async search/upsert path.
        """
        if not self._loaded:
            raise RuntimeError("GeminiEmbeddingService not loaded. Call load() first.")

        priority_keys = ["tight", "medium"]
        selected = {k: crops[k] for k in priority_keys if k in crops}

        if not selected:
            selected = {k: v for k, v in list(crops.items())[:2]}
            logger.warning(
                f"Expected 'tight'/'medium' keys, got: {list(crops.keys())}. "
                f"Using first two crops instead."
            )

        vectors: list[np.ndarray] = []
        for crop_name, crop_img in selected.items():
            try:
                vec = self._embed_single_image(crop_img, crop_name)
                vectors.append(vec)
            except Exception as e:
                logger.error(f"Failed to embed '{crop_name}' crop: {e}")
                raise

        if not vectors:
            raise RuntimeError("No crops could be embedded — all API calls failed.")

        averaged = _l2_normalize(np.mean(vectors, axis=0))
        logger.debug(
            f"Embedded {len(vectors)} crops ({list(selected.keys())}), "
            f"final dim: {len(averaged)}"
        )
        return averaged.astype(np.float32).tolist()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_single_image(self, img: Image.Image, name: str) -> np.ndarray:
        """Send one image to the Gemini Embedding API with retry/backoff."""
        b64_data = _image_to_base64_jpeg(img)

        content = {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64_data,
                    }
                }
            ]
        }

        backoff = self.initial_backoff
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                result = self._genai.embed_content(
                    model=self.model,
                    content=content,
                    task_type="SEMANTIC_SIMILARITY",
                    output_dimensionality=self.output_dimensionality,
                )
                vec = np.array(result["embedding"], dtype=np.float32)
                logger.debug(
                    f"Embedded '{name}' crop in attempt {attempt}, dim={len(vec)}"
                )
                return _l2_normalize(vec)

            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_retryable = any(
                    token in err_str
                    for token in ("rate", "quota", "429", "503", "timeout", "unavailable")
                )

                if not is_retryable or attempt == self.max_retries:
                    logger.error(
                        f"Gemini embed failed for '{name}' crop "
                        f"(attempt {attempt}/{self.max_retries}): {e}"
                    )
                    raise

                logger.warning(
                    f"Gemini embed attempt {attempt}/{self.max_retries} failed "
                    f"for '{name}' crop: {e}. Retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

        raise last_error  # type: ignore[misc]
