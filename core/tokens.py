"""Token usage aggregation."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenLedger:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, usage: dict[str, int]) -> None:
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }

    def merge(self, other: dict[str, Any]) -> None:
        self.prompt_tokens += int(other.get("prompt_tokens", 0))
        self.completion_tokens += int(other.get("completion_tokens", 0))
