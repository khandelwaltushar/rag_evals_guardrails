"""OpenAI embeddings with batching and cache deduplication."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from openai import AsyncOpenAI

from core.cache import EmbeddingCache
from core.config import Settings
from core.logging_config import get_logger

logger = get_logger(__name__)


def _l2_normalize(x: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / norms).astype(np.float32)


class OpenAIEmbedder:
    embedding_dim: int = 1536

    def __init__(self, settings: Settings, cache: EmbeddingCache | None = None) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for embeddings")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=settings.request_timeout_s)
        self._model = settings.embedding_model
        self._cache = cache or EmbeddingCache()

    async def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        cached, _missing_list = await self._cache.get(texts)
        results: list[list[float] | None] = list(cached)

        need_idx = [i for i, v in enumerate(results) if v is None]
        if need_idx:
            unique_texts = list(dict.fromkeys(texts[i] for i in need_idx))
            new_vectors: dict[str, list[float]] = {}
            batch_size = 64
            for start in range(0, len(unique_texts), batch_size):
                batch = unique_texts[start : start + batch_size]
                resp = await self._client.embeddings.create(model=self._model, input=batch)
                for item in resp.data:
                    new_vectors[batch[item.index]] = item.embedding
                if resp.usage:
                    logger.debug(
                        "embedding_tokens",
                        prompt_tokens=resp.usage.prompt_tokens,
                        batch=len(batch),
                    )
            await self._cache.set_many(list(new_vectors.keys()), list(new_vectors.values()))
            for i in need_idx:
                results[i] = new_vectors[texts[i]]

        arr = np.asarray([r for r in results if r is not None], dtype=np.float32)
        if arr.size and arr.ndim == 2 and self.embedding_dim != arr.shape[1]:
            self.embedding_dim = int(arr.shape[1])
        return _l2_normalize(arr)

    async def embed_query(self, text: str) -> NDArray[np.float32]:
        return await self.embed_texts([text])
