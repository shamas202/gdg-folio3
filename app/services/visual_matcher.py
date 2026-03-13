from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class VisualCrossEncoder:
    """
    Visual similarity scorer using plain cosine similarity.

    Both query and candidate vectors are L2-normalized 3072-dim Gemini embeddings,
    so np.dot is equivalent to cosine similarity and ranges in [-1, 1].
    """

    def score(
        self,
        query_vec: np.ndarray,
        candidate_vec: np.ndarray,
        *,
        exact_weight: float = 1.0,
        semantic_weight: float = 1.0,
    ) -> float:
        """
        Return the cosine similarity between two L2-normalized vectors.

        The exact_weight / semantic_weight parameters are kept for API
        compatibility but are not used — with a single unified Gemini embedding
        there is no meaningful instance/semantic split.
        """
        q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        c = candidate_vec / (np.linalg.norm(candidate_vec) + 1e-12)
        score = float(np.dot(q, c))
        return max(0.0, score)
