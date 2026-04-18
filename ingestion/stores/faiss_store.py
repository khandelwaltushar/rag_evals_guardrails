"""FAISS IndexFlatIP with L2-normalized vectors (cosine via inner product)."""

from __future__ import annotations

import json
import threading
from typing import Any

import faiss
import numpy as np
from numpy.typing import NDArray

from core.config import Settings
from core.logging_config import get_logger

logger = get_logger(__name__)


class FaissVectorStore:
    """Append-only dense index. Chunk IDs must be stable for updates; re-embed flows use new IDs."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._path = settings.faiss_index_path
        self._meta_path = settings.metadata_path
        self._dim: int | None = None
        self._index: faiss.Index | None = None
        self._id_to_row: dict[str, int] = {}
        self._row_to_id: list[str] = []
        self._payloads: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            self._dim = dim
            self._index = faiss.IndexFlatIP(dim)
            logger.info("faiss_index_created", dim=dim)
        elif self._dim != dim:
            raise ValueError(f"Embedding dim mismatch: index {self._dim} vs {dim}")

    def load(self) -> None:
        if self._path.exists():
            self._index = faiss.read_index(str(self._path))
            self._dim = self._index.d
            if self._meta_path.exists():
                raw = json.loads(self._meta_path.read_text(encoding="utf-8"))
                self._row_to_id = raw.get("row_to_id", [])
                self._id_to_row = {cid: i for i, cid in enumerate(self._row_to_id)}
                self._payloads = raw.get("payloads", {})
            logger.info("faiss_index_loaded", rows=len(self._row_to_id))

    async def upsert(
        self,
        chunk_ids: list[str],
        vectors: NDArray[np.float32],
        payloads: list[dict[str, Any]],
    ) -> None:
        if len(chunk_ids) != len(payloads) or vectors.shape[0] != len(chunk_ids):
            raise ValueError("upsert: parallel lists size mismatch")
        with self._lock:
            self._ensure_index(int(vectors.shape[1]))
            assert self._index is not None
            for cid, row, pay in zip(chunk_ids, vectors, payloads, strict=True):
                if cid in self._id_to_row:
                    self._payloads[cid] = {**pay, "stale_vector": True}
                    continue
                v = row.astype(np.float32).reshape(1, -1)
                faiss.normalize_L2(v)
                row_idx = self._index.ntotal
                self._index.add(v)
                self._id_to_row[cid] = row_idx
                self._row_to_id.append(cid)
                self._payloads[cid] = pay
            self._persist_unlocked()

    def _persist_unlocked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(self._path))
        self._meta_path.parent.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(
            json.dumps(
                {"row_to_id": self._row_to_id, "payloads": self._payloads},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    async def search(self, query_vector: NDArray[np.float32], top_k: int) -> list[tuple[str, float]]:
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            q = query_vector.astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(q)
            scores, indices = self._index.search(q, min(top_k, self._index.ntotal))
            out: list[tuple[str, float]] = []
            for score, idx in zip(scores[0], indices[0], strict=True):
                if idx < 0 or idx >= len(self._row_to_id):
                    continue
                cid = self._row_to_id[idx]
                out.append((cid, float(score)))
            return out

    def get_payload(self, chunk_id: str) -> dict[str, Any] | None:
        return self._payloads.get(chunk_id)
