"""Hybrid retrieval: dense + BM25 with score fusion."""

from __future__ import annotations

from core.config import Settings
from core.models import RetrievedChunk
from core.logging_config import get_logger
from ingestion.embedders.openai_embedder import OpenAIEmbedder
from ingestion.stores.bm25_index import BM25SparseIndex
from ingestion.stores.faiss_store import FaissVectorStore

logger = get_logger(__name__)


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        embedder: OpenAIEmbedder,
        dense: FaissVectorStore,
        sparse: BM25SparseIndex,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._dense = dense
        self._sparse = sparse

    async def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self._settings.top_k
        dense_k = max(k * 3, k)
        sparse_k = max(k * 3, k)

        qv = await self._embedder.embed_query(query)
        dense_hits = await self._dense.search(qv, dense_k)
        sparse_hits = await self._sparse.search(query, sparse_k)

        dense_scores = {cid: s for cid, s in dense_hits}
        sparse_scores = {cid: s for cid, s in sparse_hits}
        dn = _minmax(dense_scores)
        sn = _minmax(sparse_scores)
        all_ids = set(dn) | set(sn)
        alpha = self._settings.hybrid_alpha
        fused: dict[str, float] = {}
        for cid in all_ids:
            d = dn.get(cid, 0.0)
            s = sn.get(cid, 0.0)
            fused[cid] = alpha * d + (1.0 - alpha) * s

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        out: list[RetrievedChunk] = []
        for cid, score in ranked:
            pay = self._dense.get_payload(cid)
            if not pay:
                continue
            text = pay.get("text", "")
            doc_id = pay.get("doc_id", "")
            meta = pay.get("metadata", {})
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=str(doc_id),
                    text=text,
                    score=score,
                    metadata=meta,
                )
            )
            if len(out) >= k:
                break
        logger.debug("hybrid_retrieval", top_ids=[x.chunk_id for x in out])
        return out
