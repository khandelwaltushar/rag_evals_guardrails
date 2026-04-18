"""Semantic chunking: merge adjacent sentences using embedding similarity."""

from __future__ import annotations

import re
import uuid

import numpy as np
from numpy.typing import NDArray

from core.config import Settings
from core.models import ChunkRecord, DocumentInput
from ingestion.embedders.openai_embedder import OpenAIEmbedder


class SemanticChunker:
    """Embeds sentence windows; merges while cosine similarity to running centroid stays high."""

    def __init__(self, settings: Settings, embedder: OpenAIEmbedder) -> None:
        self._settings = settings
        self._embedder = embedder
        self._threshold = settings.semantic_similarity_threshold
        self._max_tokens = settings.chunk_size

    def _sentences(self, text: str) -> list[str]:
        s = re.split(r"(?<=[.!?])\s+", text.strip())
        return [x.strip() for x in s if x.strip()]

    async def chunk(self, doc: DocumentInput) -> list[ChunkRecord]:
        sents = self._sentences(doc.text)
        if not sents:
            return []

        embeddings = await self._embedder.embed_texts(sents)
        groups: list[list[int]] = []
        current: list[int] = [0]

        for i in range(1, len(sents)):
            # approximate token check using char length ratio
            merged = " ".join(sents[j] for j in current + [i])
            if len(merged) > self._max_tokens * 4:  # rough char proxy
                groups.append(current)
                current = [i]
                continue

            sub_emb = embeddings[current + [i]]
            centroid = np.mean(sub_emb, axis=0, keepdims=True)
            centroid = centroid / (np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-9)
            sims = (sub_emb @ centroid.T).flatten()
            if float(np.min(sims)) >= self._threshold:
                current.append(i)
            else:
                groups.append(current)
                current = [i]
        groups.append(current)

        out: list[ChunkRecord] = []
        for gi, idxs in enumerate(groups):
            text = " ".join(sents[j] for j in idxs)
            cid = f"{doc.id}::sem::{uuid.uuid4().hex[:12]}::{gi}"
            meta = {
                **doc.metadata,
                "chunk_index": gi,
                "strategy": "semantic",
                "sentence_span": [min(idxs), max(idxs)],
            }
            out.append(ChunkRecord(chunk_id=cid, doc_id=doc.id, text=text, metadata=meta))
        return out
