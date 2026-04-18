"""Query rewriting for retrieval."""

from __future__ import annotations

from typing import Any

from core.config import Settings
from core.logging_config import get_logger

logger = get_logger(__name__)


async def rewrite_query(settings: Settings, llm: Any, query: str) -> tuple[str, dict[str, int]]:
    text, usage = await llm.complete(
        system=(
            "Rewrite the user question into a standalone search query for document retrieval. "
            "Output only the rewritten query, no quotes or explanation."
        ),
        user=query,
        temperature=0.1,
        max_tokens=128,
    )
    rewritten = text.strip().split("\n")[0].strip()
    if len(rewritten) < 2:
        rewritten = query
    logger.debug("query_rewritten", original=query[:200], rewritten=rewritten[:200])
    return rewritten, usage
