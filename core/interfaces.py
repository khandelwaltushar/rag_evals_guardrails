"""Protocol-based interfaces for dependency injection and testing."""

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from core.models import ChunkRecord, DocumentInput, RetrievedChunk


@runtime_checkable
class Chunker(Protocol):
    """Splits documents into chunks."""

    async def chunk(self, doc: DocumentInput) -> list[ChunkRecord]:
        ...


@runtime_checkable
class Embedder(Protocol):
    """Text → dense vectors. Implementations must dedupe/cache externally."""

    embedding_dim: int

    async def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """Returns shape (n, dim) float32, L2-normalized rows for cosine/IP."""

    async def embed_query(self, text: str) -> NDArray[np.float32]:
        ...


@runtime_checkable
class VectorStore(Protocol):
    """FAISS-backed or similar vector index."""

    async def upsert(
        self,
        chunk_ids: list[str],
        vectors: NDArray[np.float32],
        payloads: list[dict[str, Any]],
    ) -> None:
        ...

    async def search(
        self, query_vector: NDArray[np.float32], top_k: int
    ) -> list[tuple[str, float]]:
        """Returns list of (chunk_id, score)."""


@runtime_checkable
class SparseIndex(Protocol):
    """BM25 or other lexical index."""

    async def index_documents(self, chunks: list[ChunkRecord]) -> None:
        ...

    async def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        ...


@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        ...


@runtime_checkable
class LLMClient(Protocol):
    """Pluggable chat completion."""

    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> tuple[str, dict[str, int]]:
        """Returns (text, token_usage dict: prompt_tokens, completion_tokens)."""


@runtime_checkable
class IngestionService(Protocol):
    async def ingest(self, documents: list[DocumentInput]) -> tuple[int, int, list[str]]:
        """Returns (docs_count, chunks_count, errors)."""


@runtime_checkable
class QueryService(Protocol):
    async def query(self, query: str, **kwargs: Any) -> Any:
        ...
