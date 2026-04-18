"""Request-scoped trace for observability."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from core.models import PipelineTrace


@dataclass
class TraceContext:
    """Collects structured steps for API responses when debug is on."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: list[dict[str, Any]] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    prompt_preview: str | None = None

    def add_step(self, name: str, data: dict[str, Any]) -> None:
        self.steps.append({"step": name, **data})

    def set_prompt_preview(self, prompt: str, max_len: int = 4000) -> None:
        self.prompt_preview = prompt[:max_len] + ("..." if len(prompt) > max_len else "")

    def to_pipeline_trace(self, debug: bool) -> PipelineTrace:
        return PipelineTrace(
            trace_id=self.trace_id,
            steps=self.steps,
            retrieved_chunk_ids=self.retrieved_chunk_ids,
            prompt_preview=self.prompt_preview if debug else None,
            debug=debug,
        )
