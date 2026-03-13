from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from loguru import logger

from app.services.visual_matcher import VisualCrossEncoder


@dataclass
class MultiStageRerankService:
    """
    Two-stage reranker for large catalogs using plain cosine similarity on
    3072-dim Gemini embeddings.

    Stage 1 — fast dot-product filter: keep top N% candidates.
    Stage 2 — full cosine similarity re-score on the filtered set.
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
        if not query_vector or not candidates:
            return []

        q = np.array(query_vector, dtype=np.float32)
        if q.shape[0] == 0:
            logger.error("Query vector has zero length")
            return []

        q = q / (np.linalg.norm(q) + 1e-12)

        logger.debug(
            f"Two-stage reranking: {len(candidates)} candidates → top {top_k}"
        )

        # ── Stage 1: fast dot-product filter ────────────────────────────
        stage1_results = []
        for c in candidates:
            if "values" not in c or not c["values"]:
                continue

            vec = np.array(c["values"], dtype=np.float32)
            if vec.shape[0] == 0 or vec.shape[0] != q.shape[0]:
                logger.debug(
                    f"Skipping candidate with invalid vector shape {vec.shape[0]} "
                    f"(expected {q.shape[0]})"
                )
                continue

            c["stage1_score"] = float(np.dot(q, vec / (np.linalg.norm(vec) + 1e-12)))
            stage1_results.append(c)

        if not stage1_results:
            logger.warning("No valid candidates after Stage 1 filtering")
            return []

        # Adaptive cutoff based on catalog size
        n = len(stage1_results)
        if n > 10000:
            ratio = 0.2
        elif n > 1000:
            ratio = 0.3
        else:
            ratio = 0.5

        stage1_k = max(int(n * ratio), top_k * 3, 50)
        stage1_results.sort(key=lambda x: x["stage1_score"], reverse=True)
        filtered = stage1_results[:stage1_k]

        logger.debug(
            f"Stage 1 done: {n} → {len(filtered)} "
            f"(ratio={ratio:.0%}, min_score={filtered[-1]['stage1_score']:.3f})"
        )

        # ── Stage 2: full cosine re-score ────────────────────────────────
        for c in filtered:
            vec = np.array(c["values"], dtype=np.float32)
            score = float(np.dot(q, vec / (np.linalg.norm(vec) + 1e-12)))
            c["final_score"] = max(0.0, score)

        filtered.sort(key=lambda x: x["final_score"], reverse=True)

        if filtered:
            top = filtered[0]
            logger.debug(
                f"Stage 2 done: best={top['final_score']:.3f}, "
                f"worst={filtered[-1]['final_score']:.3f}, "
                f"sku={top.get('metadata', {}).get('sku_id', 'unknown')}"
            )

        return filtered[:top_k]


@dataclass
class HybridRerankService:
    """
    Adaptive reranker: uses MultiStageRerankService for >1000 candidates,
    falls back to simple single-stage cosine reranking for smaller sets.
    """

    matcher: VisualCrossEncoder

    def __post_init__(self) -> None:
        self._multistage = MultiStageRerankService(matcher=self.matcher)

    def rerank(
        self,
        *,
        query_vector: list[float],
        candidates: list[dict[str, Any]],
        top_k: int,
        exact_first: bool,
    ) -> list[dict[str, Any]]:
        if len(candidates) > 1000:
            logger.debug("Using multi-stage reranking (large candidate set)")
            return self._multistage.rerank(
                query_vector=query_vector,
                candidates=candidates,
                top_k=top_k,
                exact_first=exact_first,
            )

        logger.debug("Using single-stage reranking (small candidate set)")
        return self._simple_rerank(
            query_vector=query_vector,
            candidates=candidates,
            top_k=top_k,
        )

    def _simple_rerank(
        self,
        *,
        query_vector: list[float],
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not query_vector or not candidates:
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
