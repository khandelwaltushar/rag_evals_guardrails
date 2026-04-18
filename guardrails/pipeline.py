"""Orchestrates detection, self-check, and fallbacks."""

from __future__ import annotations

from core.config import Settings
from core.models import EvaluationMetrics, RetrievedChunk
from core.tokens import TokenLedger
from guardrails.confidence import score_confidence
from guardrails.detector import detect
from guardrails import fallback as fb
from guardrails.self_check import self_evaluate


class GuardrailPipeline:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def apply(
        self,
        query: str,
        draft_answer: str,
        retrieved: list[RetrievedChunk],
        metrics: EvaluationMetrics | None,
        *,
        ledger: TokenLedger | None = None,
    ) -> tuple[str, float, str]:
        """
        Returns (final_answer, confidence, action).
        action: none | insufficient_info | clarify
        """
        context = "\n\n".join(c.text for c in retrieved[:8])
        signals = detect(self._settings, metrics)

        self_score = 1.0
        if signals.low_context_similarity or signals.judge_flags_faithfulness:
            ss, usage = await self_evaluate(self._settings, draft_answer, context)
            self_score = ss
            if ledger:
                ledger.merge(usage)

        confidence = score_confidence(retrieved, metrics, self._settings.min_context_similarity)
        confidence = confidence * 0.7 + self_score * 0.3

        action = "none"
        answer = draft_answer

        if not retrieved:
            answer = fb.insufficient_information_message()
            action = "insufficient_info"
            confidence = min(confidence, 0.2)
        elif confidence < self._settings.min_confidence_threshold or (
            signals.low_context_similarity and signals.judge_flags_faithfulness
        ):
            if self_score < 0.4:
                answer = fb.insufficient_information_message()
                action = "insufficient_info"
            else:
                answer = fb.clarifying_prompt(query, retrieved)
                action = "clarify"
            confidence = min(confidence, self._settings.min_confidence_threshold)

        return answer, float(confidence), action
