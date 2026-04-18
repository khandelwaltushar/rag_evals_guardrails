"""LLM self-evaluation of groundedness."""

from __future__ import annotations

import re

from core.config import Settings
from llm.factory import get_llm_client


async def self_evaluate(
    settings: Settings,
    answer: str,
    context: str,
) -> tuple[float, dict[str, int]]:
    llm = get_llm_client(settings)
    text, usage = await llm.complete(
        system="Reply with a single float 0-1 only: groundedness of the answer in the context.",
        user=f"Context:\n{context[:6000]}\n\nAnswer:\n{answer}\n\nGroundedness:",
        temperature=0.0,
        max_tokens=16,
    )
    m = re.search(r"0?\.\d+|1\.0|1|0", text.strip())
    score = float(m.group()) if m else 0.5
    return min(1.0, max(0.0, score)), usage
