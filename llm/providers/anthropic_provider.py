"""Anthropic chat client."""

from __future__ import annotations

from anthropic import AsyncAnthropic

from core.config import Settings


class AnthropicClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY required")
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key, timeout=settings.request_timeout_s)
        self._model = settings.chat_model

    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> tuple[str, dict[str, int]]:
        mt = max_tokens or 4096
        resp = await self._client.messages.create(
            model=model or self._model,
            max_tokens=mt,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = []
        for b in resp.content:
            if hasattr(b, "text"):
                parts.append(b.text)
        text = "".join(parts)
        u = {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
        }
        return text, u
