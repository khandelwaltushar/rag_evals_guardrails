"""Structured LLM-as-judge output."""

from typing import Any

from pydantic import BaseModel, Field


class JudgeResult(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    retrieval_quality: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Whether retrieved passages contain answer evidence",
    )
    rationale: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump()
