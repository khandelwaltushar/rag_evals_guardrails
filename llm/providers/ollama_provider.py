"""Ollama chat client via its OpenAI-compatible endpoint."""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from core.config import Settings


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self._client = AsyncOpenAI(
            api_key="ollama",
            base_url=settings.ollama_base_url,
            timeout=settings.request_timeout_s,
        )
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
        resp = await self._client.chat.completions.create(
            model=model or self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        u = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
        }
        return text, u

    async def stream_complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=model or self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
