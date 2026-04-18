"""Application settings — config-driven pipeline."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve `.env` from the project root so keys load regardless of cwd (e.g. uvicorn from another dir).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ChunkingStrategy(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class LLMProviderName(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class EmbedderProviderName(str, Enum):
    OPENAI = "openai"
    LOCAL = "local"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(
            _PROJECT_ROOT / ".env",
            _PROJECT_ROOT / ".env.local",
        ),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # Feature flags
    rag_debug: bool = Field(default=False, description="Verbose traces and prompts in responses")
    use_reranker: bool = Field(default=False)
    hybrid_alpha: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Blend weight for dense vs sparse: score = alpha*dense + (1-alpha)*sparse",
    )

    # Providers
    llm_provider: LLMProviderName = Field(default=LLMProviderName.OPENAI)
    embedder_provider: EmbedderProviderName = Field(default=EmbedderProviderName.OPENAI)
    embedding_model: str = Field(default="text-embedding-3-small")
    chat_model: str = Field(default="gpt-4o-mini")
    judge_model: str = Field(default="gpt-4o-mini")

    # Ollama — when LLM_PROVIDER=ollama
    ollama_base_url: str = Field(default="http://localhost:11434/v1")

    # OpenAI — reads `OPENAI_API_KEY` from environment and from `.env` at project root
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key"),
    )

    # Anthropic
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "anthropic_api_key"),
    )

    # Redis
    redis_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("REDIS_URL", "redis_url"),
    )

    # Storage
    faiss_index_path: Path = Field(default=Path("./data/faiss_index"))
    bm25_corpus_path: Path = Field(default=Path("./data/bm25_corpus.json"))
    metadata_path: Path = Field(default=Path("./data/chunk_metadata.json"))

    # Chunking
    chunking_strategy: ChunkingStrategy = Field(default=ChunkingStrategy.RECURSIVE)
    chunk_size: int = Field(default=512, description="Target chunk size in tokens (approx)")
    chunk_overlap: int = Field(default=64)
    semantic_similarity_threshold: float = Field(default=0.75)

    # Retrieval
    top_k: int = Field(default=8)
    rerank_top_n: int = Field(default=4)

    # Guardrails
    min_context_similarity: float = Field(default=0.35)
    min_confidence_threshold: float = Field(default=0.45)

    # HTTP
    request_timeout_s: float = Field(default=120.0)

    @field_validator("redis_url", mode="before")
    @classmethod
    def _empty_redis_url(cls, v: str | None) -> str | None:
        if v is not None and str(v).strip() == "":
            return None
        return v

    @model_validator(mode="after")
    def _anthropic_model_ids(self) -> Settings:
        if self.llm_provider != LLMProviderName.ANTHROPIC:
            return self
        for name, val in (("CHAT_MODEL", self.chat_model), ("JUDGE_MODEL", self.judge_model)):
            if val.startswith("gpt") or val.startswith("o1") or val.startswith("o3"):
                raise ValueError(
                    f"When LLM_PROVIDER=anthropic, {name} must be a Claude model id "
                    f"(e.g. claude-3-5-haiku-20241022), not {val!r}."
                )
        return self

    def effective_llm_provider(self) -> Literal["openai", "anthropic", "ollama"]:
        return self.llm_provider.value


def get_settings() -> Settings:
    return Settings()
