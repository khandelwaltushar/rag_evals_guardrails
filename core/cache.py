"""Embedding and response cache — Redis with in-memory fallback."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from core.logging_config import get_logger

logger = get_logger(__name__)


def _key(prefix: str, payload: str) -> str:
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}:{h}"


class EmbeddingCache:
    """Caches embedding vectors as JSON lists to survive process restarts when using Redis."""

    def __init__(self, redis_client: Any | None = None, *, prefix: str = "emb") -> None:
        self._redis = redis_client
        self._prefix = prefix
        self._mem: dict[str, list[float]] = {}

    async def get(self, texts: list[str]) -> tuple[list[list[float] | None], list[str]]:
        """Returns parallel lists: cached vectors (or None) and texts that need embedding."""
        keys = [_key(self._prefix, t) for t in texts]
        out: list[list[float] | None] = []
        missing: list[str] = []
        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                for k in keys:
                    pipe.get(k)
                raw = await pipe.execute()
                for i, val in enumerate(raw):
                    if val is None:
                        out.append(None)
                        missing.append(texts[i])
                    else:
                        out.append(json.loads(val))
                return out, missing
            except Exception as e:
                logger.warning("redis_embedding_cache_fallback", error=str(e))
        for i, k in enumerate(keys):
            v = self._mem.get(k)
            if v is None:
                out.append(None)
                missing.append(texts[i])
            else:
                out.append(v)
        return out, missing

    async def set_many(self, texts: list[str], vectors: list[list[float]]) -> None:
        if len(texts) != len(vectors):
            raise ValueError("texts and vectors length mismatch")
        pairs = list(zip(texts, vectors, strict=True))
        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                for text, vec in pairs:
                    k = _key(self._prefix, text)
                    pipe.set(k, json.dumps(vec), ex=86400 * 7)
                await pipe.execute()
                return
            except Exception as e:
                logger.warning("redis_embedding_cache_set_fallback", error=str(e))
        for text, vec in pairs:
            self._mem[_key(self._prefix, text)] = vec
