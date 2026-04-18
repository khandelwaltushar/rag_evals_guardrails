"""Optional cross-encoder style reranking via LLM scoring (lightweight)."""

from __future__ import annotations

import json
import re
from typing import Any

from core.config import Settings
from core.models import RetrievedChunk
from core.logging_config import get_logger
logger = get_logger(__name__)


class LLMReranker:
    """Uses small LLM to assign relevance 0-1 per candidate — use sparingly for cost."""

    def __init__(self, settings: Settings, llm: Any) -> None:
        self._settings = settings
        self._llm = llm

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        subset = candidates[:20]
        lines = []
        for i, c in enumerate(subset):
            snippet = c.text[:500].replace("\n", " ")
            lines.append(f"[{i}] {snippet}")
        prompt = (
            "Score each passage for answering the query. "
            'Reply with JSON only: {"scores": [<float>, ...]} in same order.\n\n'
            f"Query: {query}\n\n" + "\n".join(lines)
        )
        text, _usage = await self._llm.complete(
            system="You output only valid JSON.",
            user=prompt,
            temperature=0.0,
            max_tokens=256,
        )
        scores = self._parse_scores(text, len(subset))
        for c, s in zip(subset, scores, strict=True):
            c.metadata = {**c.metadata, "rerank_score": s}
        ranked = sorted(
            zip(subset, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        return [r[0] for r in ranked[:top_n]]

    def _parse_scores(self, text: str, n: int) -> list[float]:
        try:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return [1.0] * n
            data = json.loads(m.group())
            arr = data.get("scores", [])
            out = [float(x) for x in arr[:n]]
            while len(out) < n:
                out.append(0.0)
            return out[:n]
        except Exception:
            logger.warning("rerank_parse_failed", text_preview=text[:200])
            return [1.0] * n
