"""Orchestrates chunking → embedding → FAISS + BM25."""

from __future__ import annotations

from core.config import ChunkingStrategy, Settings
from core.interfaces import Chunker
from core.logging_config import get_logger
from core.models import ChunkRecord, DocumentInput
from ingestion.chunkers.recursive import RecursiveChunker
from ingestion.chunkers.semantic import SemanticChunker
from ingestion.embedders.openai_embedder import OpenAIEmbedder
from ingestion.stores.bm25_index import BM25SparseIndex
from ingestion.stores.faiss_store import FaissVectorStore

logger = get_logger(__name__)


def _select_chunker(settings: Settings, embedder: OpenAIEmbedder) -> Chunker:
    if settings.chunking_strategy == ChunkingStrategy.SEMANTIC:
        return SemanticChunker(settings, embedder)
    return RecursiveChunker(settings)


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: OpenAIEmbedder,
        vector_store: FaissVectorStore,
        bm25: BM25SparseIndex,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25 = bm25
        self._chunker = _select_chunker(settings, embedder)

    async def ingest(self, documents: list[DocumentInput]) -> tuple[int, int, list[str]]:
        errors: list[str] = []
        all_chunks: list[ChunkRecord] = []

        for doc in documents:
            try:
                chunks = await self._chunker.chunk(doc)
                all_chunks.extend(chunks)
                logger.info("document_chunked", doc_id=doc.id, chunks=len(chunks))
            except Exception as e:
                errors.append(f"{doc.id}: {e!s}")
                logger.exception("chunk_failed", doc_id=doc.id)

        if not all_chunks:
            return len(documents), 0, errors

        texts = [c.text for c in all_chunks]
        embeddings = await self._embedder.embed_texts(texts)

        chunk_ids = [c.chunk_id for c in all_chunks]
        payloads = [
            {
                "doc_id": c.doc_id,
                "text": c.text,
                "metadata": c.metadata,
            }
            for c in all_chunks
        ]

        await self._vector_store.upsert(chunk_ids, embeddings, payloads)
        await self._bm25.index_documents(all_chunks)

        logger.info(
            "ingestion_complete",
            chunks=len(all_chunks),
            dim=embeddings.shape[1] if embeddings.size else 0,
        )
        return len(documents), len(all_chunks), errors
