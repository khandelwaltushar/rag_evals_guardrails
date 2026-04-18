"""End-to-end query pipeline: rewrite → retrieve → rerank → generate → eval → guardrails."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from core.config import Settings
from core.logging_config import get_logger
from core.models import QueryResponse
from core.tokens import TokenLedger
from core.tracing import TraceContext
from evaluation.pipeline import EvaluationPipeline
from guardrails.pipeline import GuardrailPipeline
from ingestion.embedders.openai_embedder import OpenAIEmbedder
from retrieval.hybrid import HybridRetriever
from retrieval.query_rewrite import rewrite_query
from retrieval.reranker import LLMReranker

logger = get_logger(__name__)


class QueryPipelineService:
    def __init__(
        self,
        settings: Settings,
        llm: Any,
        embedder: OpenAIEmbedder,
        hybrid: HybridRetriever,
        evaluator: EvaluationPipeline,
        guardrails: GuardrailPipeline,
        reranker: LLMReranker | None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._embedder = embedder
        self._hybrid = hybrid
        self._evaluator = evaluator
        self._guardrails = guardrails
        self._reranker = reranker

    async def run(self, query: str, *, skip_evaluation: bool = False, top_k: int | None = None) -> QueryResponse:
        trace = TraceContext()
        ledger = TokenLedger()
        k = top_k or self._settings.top_k

        rw, u0 = await rewrite_query(self._settings, self._llm, query)
        ledger.merge(u0)
        trace.add_step("query_rewrite", {"rewritten": rw})

        retrieved = await self._hybrid.retrieve(rw, top_k=k * 2 if self._settings.use_reranker else k)
        trace.retrieved_chunk_ids = [c.chunk_id for c in retrieved]
        trace.add_step("retrieval", {"count": len(retrieved), "chunk_ids": trace.retrieved_chunk_ids[:20]})

        if self._settings.use_reranker and self._reranker and retrieved:
            retrieved = await self._reranker.rerank(rw, retrieved, top_n=k)
            trace.add_step("rerank", {"count": len(retrieved)})

        context_block = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in retrieved[: k + 2])
        system = (
            "You are a careful assistant. Answer using ONLY the provided context. "
            "If the context does not contain the answer, say you cannot find it in the sources."
        )
        user = f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer concisely:"
        trace.set_prompt_preview(system + "\n\n" + user)

        answer, u1 = await self._llm.complete(
            system=system,
            user=user,
            temperature=0.2,
            max_tokens=1024,
        )
        ledger.merge(u1)
        trace.add_step("generation", {"answer_chars": len(answer)})

        metrics = None
        if not skip_evaluation:
            metrics, u2 = await self._evaluator.run(query, answer, retrieved)
            ledger.merge(u2)
            trace.add_step(
                "evaluation",
                {
                    "faithfulness": metrics.faithfulness_score,
                    "relevance": metrics.relevance_score,
                },
            )

        final, confidence, action = await self._guardrails.apply(
            query, answer, retrieved, metrics, ledger=ledger
        )
        trace.add_step("guardrails", {"action": action, "confidence": confidence})

        trace_dict = trace.to_pipeline_trace(self._settings.rag_debug).model_dump()
        if not self._settings.rag_debug:
            trace_dict.pop("prompt_preview", None)

        return QueryResponse(
            answer=final,
            retrieved_docs=retrieved[:k],
            confidence_score=confidence,
            evaluation_metrics=metrics,
            guardrail_action=action,
            trace=trace_dict,
            token_usage=ledger.to_dict(),
        )

    async def run_stream(self, query: str, *, top_k: int | None = None) -> AsyncIterator[tuple[str, Any]]:
        """Streaming variant: yields (event_type, payload) tuples.

        Events: 'retrieved' (doc list), 'token' (text delta), 'done' (final guardrail+confidence meta).
        Evaluation is skipped in streaming mode; call /query for metrics.
        """
        if not hasattr(self._llm, "stream_complete"):
            raise RuntimeError("Current LLM provider does not support streaming")

        k = top_k or self._settings.top_k
        rw, _ = await rewrite_query(self._settings, self._llm, query)
        retrieved = await self._hybrid.retrieve(rw, top_k=k * 2 if self._settings.use_reranker else k)
        if self._settings.use_reranker and self._reranker and retrieved:
            retrieved = await self._reranker.rerank(rw, retrieved, top_n=k)
        retrieved = retrieved[:k]

        yield ("retrieved", [c.model_dump() for c in retrieved])

        context_block = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in retrieved)
        system = (
            "You are a careful assistant. Answer using ONLY the provided context. "
            "If the context does not contain the answer, say you cannot find it in the sources."
        )
        user = f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer concisely:"

        buf: list[str] = []
        async for delta in self._llm.stream_complete(
            system=system, user=user, temperature=0.2, max_tokens=1024
        ):
            buf.append(delta)
            yield ("token", delta)

        answer = "".join(buf)
        final, confidence, action = await self._guardrails.apply(query, answer, retrieved, None)
        replaced = final != answer

        yield (
            "done",
            {
                "confidence_score": confidence,
                "guardrail_action": action,
                "guardrail_replaced_answer": replaced,
                "final_answer_if_replaced": final if replaced else None,
            },
        )
