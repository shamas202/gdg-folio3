from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.services.visual_matcher import VisualCrossEncoder


class RerankService(ABC):
    @abstractmethod
    def rerank(
        self,
        *,
        query_vector: list[float],
        candidates: list[dict[str, Any]],
        top_k: int,
        exact_first: bool,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


@dataclass
class VisualCrossEncoderRerankService(RerankService):
    """
    Single-stage reranker using plain cosine similarity on 3072-dim Gemini vectors.
    """

    matcher: VisualCrossEncoder

    def rerank(
        self,
        *,
        query_vector: list[float],
        candidates: list[dict[str, Any]],
        top_k: int,
        exact_first: bool,
    ) -> list[dict[str, Any]]:
        if not query_vector:
            return []

        q = np.array(query_vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)

        reranked = []
        for c in candidates:
            if "values" not in c or not c["values"]:
                continue

            vec = np.array(c["values"], dtype=np.float32)
            if vec.shape[0] == 0 or vec.shape[0] != q.shape[0]:
                continue

            score = float(np.dot(q, vec / (np.linalg.norm(vec) + 1e-12)))
            c["final_score"] = max(0.0, score)
            reranked.append(c)

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked[:top_k]
