"""Pluggable LLM provider."""

from __future__ import annotations

from typing import Any

from core.config import LLMProviderName, Settings
from llm.providers.anthropic_provider import AnthropicClient
from llm.providers.ollama_provider import OllamaClient
from llm.providers.openai_provider import OpenAIClient


def get_llm_client(settings: Settings) -> Any:
    if settings.llm_provider == LLMProviderName.ANTHROPIC:
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic "
                "(chat, judge, rerank, guardrails). Embeddings still use OPENAI_API_KEY."
            )
        return AnthropicClient(settings)
    if settings.llm_provider == LLMProviderName.OLLAMA:
        return OllamaClient(settings)
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
    return OpenAIClient(settings)
