"""Local sentence-transformers embedder — no API, CPU/GPU-local."""

from __future__ import annotations

import asyncio

import numpy as np
from numpy.typing import NDArray

from core.cache import EmbeddingCache
from core.config import Settings
from core.logging_config import get_logger

logger = get_logger(__name__)


def _l2_normalize(x: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / norms).astype(np.float32)


class LocalEmbedder:
    """Mirrors the OpenAIEmbedder interface but runs a sentence-transformers model locally."""

    def __init__(self, settings: Settings, cache: EmbeddingCache | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = settings.embedding_model
        self._model = SentenceTransformer(self._model_name)
        self.embedding_dim = int(self._model.get_sentence_embedding_dimension())
        self._cache = cache or EmbeddingCache()
        logger.info("local_embedder_loaded", model=self._model_name, dim=self.embedding_dim)

    def _encode_sync(self, texts: list[str]) -> NDArray[np.float32]:
        vecs = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    async def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        cached, _missing = await self._cache.get(texts)
        results: list[list[float] | None] = list(cached)

        need_idx = [i for i, v in enumerate(results) if v is None]
        if need_idx:
            unique_texts = list(dict.fromkeys(texts[i] for i in need_idx))
            loop = asyncio.get_running_loop()
            arr = await loop.run_in_executor(None, self._encode_sync, unique_texts)
            new_vectors = {t: arr[i].tolist() for i, t in enumerate(unique_texts)}
            await self._cache.set_many(list(new_vectors.keys()), list(new_vectors.values()))
            for i in need_idx:
                results[i] = new_vectors[texts[i]]

        arr = np.asarray([r for r in results if r is not None], dtype=np.float32)
        if arr.size and arr.ndim == 2 and self.embedding_dim != arr.shape[1]:
            self.embedding_dim = int(arr.shape[1])
        return _l2_normalize(arr)

    async def embed_query(self, text: str) -> NDArray[np.float32]:
        return await self.embed_texts([text])
