"""LLM-as-judge for faithfulness / relevance / retrieval — uses same provider as chat (OpenAI or Anthropic)."""

from __future__ import annotations

import json
import re
from typing import Any

from core.config import Settings
from core.logging_config import get_logger
from evaluation.schemas import JudgeResult

logger = get_logger(__name__)


class LLMJudge:
    """Delegates to the injected LLM client (OpenAI or Anthropic) so keys match `LLM_PROVIDER`."""

    def __init__(self, settings: Settings, llm: Any) -> None:
        self._settings = settings
        self._llm = llm

    async def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> tuple[JudgeResult, dict[str, int]]:
        user = f"""Query: {query}

Context (retrieved):
{context}

Answer:
{answer}

Respond with JSON only:
{{
  "faithfulness": <0-1 float, is the answer supported by context>,
  "relevance": <0-1 float, does the answer address the query>,
  "retrieval_quality": <0-1 float, does the context contain what is needed>,
  "rationale": "<short>"
}}
"""
        text, usage = await self._llm.complete(
            system="You are an evaluation assistant. Output only valid JSON.",
            user=user,
            temperature=0.0,
            max_tokens=512,
            model=self._settings.judge_model,
        )
        data = _extract_json(text)
        try:
            jr = JudgeResult.model_validate(data)
        except Exception as e:
            logger.warning("judge_parse_failed", error=str(e), text_preview=text[:300])
            jr = JudgeResult(faithfulness=0.5, relevance=0.5, rationale="parse_error")
        return jr, usage


def _extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    return json.loads(m.group())
