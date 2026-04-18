"""Fallback strategies when confidence is low or hallucination suspected."""

from __future__ import annotations

from core.models import RetrievedChunk


def insufficient_information_message() -> str:
    return (
        "I do not have enough information in the retrieved sources to answer "
        "this question confidently. Please try rephrasing or provide more context."
    )


def clarifying_prompt(query: str, retrieved: list[RetrievedChunk]) -> str:
    topics = ", ".join({c.doc_id for c in retrieved[:5]}) or "your documents"
    return (
        f"Your question may be broad or ambiguous in relation to {topics}. "
        "Could you specify which aspect you care about most, or narrow the timeframe or entity?"
    )
