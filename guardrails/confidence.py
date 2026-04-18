"""Aggregate confidence from retrieval scores and evaluation signals."""

from __future__ import annotations

from core.models import EvaluationMetrics, RetrievedChunk


def score_confidence(
    retrieved: list[RetrievedChunk],
    metrics: EvaluationMetrics | None,
    min_sim: float,
) -> float:
    ret_score = 0.0
    if retrieved:
        ret_score = sum(c.score for c in retrieved) / max(len(retrieved), 1)
        ret_score = min(1.0, max(0.0, ret_score))

    if metrics is None:
        return 0.5 * ret_score + 0.5 * (1.0 if retrieved else 0.0)

    emb = metrics.embedding_answer_context_sim or 0.0
    faith = metrics.faithfulness_score
    rel = metrics.relevance_score
    # Penalize if embedding says answer drifts from context
    grounded = emb if emb >= min_sim else emb * 0.5
    return float(0.25 * ret_score + 0.25 * grounded + 0.25 * faith + 0.25 * rel)
