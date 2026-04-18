"""Application container — dependency injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.cache import EmbeddingCache
from core.config import EmbedderProviderName, Settings, get_settings
from core.redis_client import create_redis
from evaluation.judge import LLMJudge
from evaluation.pipeline import EvaluationPipeline
from guardrails.pipeline import GuardrailPipeline
from ingestion.embedders.local_embedder import LocalEmbedder
from ingestion.embedders.openai_embedder import OpenAIEmbedder
from ingestion.pipeline import IngestionPipeline
from ingestion.stores.bm25_index import BM25SparseIndex
from ingestion.stores.faiss_store import FaissVectorStore
from llm.factory import get_llm_client
from retrieval.hybrid import HybridRetriever
from retrieval.query_service import QueryPipelineService
from retrieval.reranker import LLMReranker


@dataclass
class AppState:
    settings: Settings
    redis: Any
    embedder: Any
    vector_store: FaissVectorStore
    bm25: BM25SparseIndex
    ingestion: IngestionPipeline
    hybrid: HybridRetriever
    query_service: QueryPipelineService


def _build_embedder(settings: Settings, cache: EmbeddingCache) -> Any:
    if settings.embedder_provider == EmbedderProviderName.LOCAL:
        return LocalEmbedder(settings, cache)
    return OpenAIEmbedder(settings, cache)


async def build_app_state() -> AppState:
    settings = get_settings()
    redis = await create_redis(settings.redis_url)
    emb_cache = EmbeddingCache(redis)
    embedder = _build_embedder(settings, emb_cache)
    vector_store = FaissVectorStore(settings)
    vector_store.load()
    bm25 = BM25SparseIndex(settings)
    bm25.load()
    ingestion = IngestionPipeline(settings, embedder, vector_store, bm25)
    hybrid = HybridRetriever(settings, embedder, vector_store, bm25)
    llm = get_llm_client(settings)
    judge = LLMJudge(settings, llm)
    evaluator = EvaluationPipeline(settings, embedder, judge)
    guardrails = GuardrailPipeline(settings)
    reranker = LLMReranker(settings, llm) if settings.use_reranker else None
    query_service = QueryPipelineService(
        settings,
        llm,
        embedder,
        hybrid,
        evaluator,
        guardrails,
        reranker,
    )
    return AppState(
        settings=settings,
        redis=redis,
        embedder=embedder,
        vector_store=vector_store,
        bm25=bm25,
        ingestion=ingestion,
        hybrid=hybrid,
        query_service=query_service,
    )
