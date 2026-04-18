"""BM25 lexical index — persisted as tokenized corpus + rank_bm25."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from core.config import Settings
from core.models import ChunkRecord
from core.logging_config import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25SparseIndex:
    def __init__(self, settings: Settings) -> None:
        self._path = settings.bm25_corpus_path
        self._chunk_ids: list[str] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._chunk_ids = raw.get("chunk_ids", [])
            self._tokenized = raw.get("tokenized", [])
            if self._tokenized:
                self._bm25 = BM25Okapi(self._tokenized)
            logger.info("bm25_loaded", docs=len(self._chunk_ids))

    def _persist_unlocked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(
                {"chunk_ids": self._chunk_ids, "tokenized": self._tokenized},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    async def index_documents(self, chunks: list[ChunkRecord]) -> None:
        with self._lock:
            for ch in chunks:
                self._chunk_ids.append(ch.chunk_id)
                self._tokenized.append(_tokenize(ch.text))
            self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
            self._persist_unlocked()

    async def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        with self._lock:
            if not self._bm25 or not self._chunk_ids:
                return []
            q = _tokenize(query)
            scores = self._bm25.get_scores(q)
            ranked = sorted(
                enumerate(scores),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]
            return [(self._chunk_ids[i], float(s)) for i, s in ranked]
