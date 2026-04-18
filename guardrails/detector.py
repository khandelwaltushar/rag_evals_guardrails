"""Hallucination heuristics: low similarity + optional self-check."""

from __future__ import annotations

from dataclasses import dataclass

from core.config import Settings
from core.models import EvaluationMetrics


@dataclass
class HallucinationSignals:
    low_context_similarity: bool
    judge_flags_faithfulness: bool


def detect(
    settings: Settings,
    metrics: EvaluationMetrics | None,
) -> HallucinationSignals:
    emb = metrics.embedding_answer_context_sim if metrics else None
    low_sim = emb is not None and emb < settings.min_context_similarity
    judge_bad = bool(metrics and metrics.faithfulness_score < settings.min_confidence_threshold)
    return HallucinationSignals(
        low_context_similarity=low_sim,
        judge_flags_faithfulness=judge_bad,
    )
