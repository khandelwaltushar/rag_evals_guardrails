"""Embedding similarity metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cosine_sim(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    a = a.flatten()
    b = b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


async def answer_context_similarity(
    answer_vec: NDArray[np.float32],
    context_vecs: list[NDArray[np.float32]],
) -> float:
    if not context_vecs:
        return 0.0
    sims = [cosine_sim(answer_vec, cv) for cv in context_vecs]
    return float(max(sims))
