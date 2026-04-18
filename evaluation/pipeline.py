"""Combines judge + embedding metrics into EvaluationMetrics."""

from __future__ import annotations

import numpy as np

from core.config import Settings
from core.models import EvaluationMetrics, RetrievedChunk
from evaluation.embedding_metrics import answer_context_similarity
from evaluation.judge import LLMJudge
from ingestion.embedders.openai_embedder import OpenAIEmbedder


class EvaluationPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: OpenAIEmbedder,
        judge: LLMJudge | None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._judge = judge

    async def run(
        self,
        query: str,
        answer: str,
        retrieved: list[RetrievedChunk],
    ) -> tuple[EvaluationMetrics, dict[str, int]]:
        context = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in retrieved[:8])
        ans_vec = await self._embedder.embed_texts([answer])
        ctx_texts = [c.text for c in retrieved[:8]]
        ctx_arr = await self._embedder.embed_texts(ctx_texts) if ctx_texts else None
        ctx_stack = [ctx_arr[i] for i in range(ctx_arr.shape[0])] if ctx_arr is not None and ctx_arr.size else []

        ac_sim = await answer_context_similarity(ans_vec[0], ctx_stack)
        q_vec = await self._embedder.embed_query(query)
        aq_sim = float(
            np.dot(ans_vec[0].flatten(), q_vec.flatten())
            / (
                np.linalg.norm(ans_vec[0])
                * np.linalg.norm(q_vec)
                + 1e-12
            )
        )

        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        judge_raw: dict = {}

        if self._judge:
            jr, u = await self._judge.evaluate(query, answer, context)
            usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            usage["completion_tokens"] += u.get("completion_tokens", 0)
            faith = jr.faithfulness
            rel = jr.relevance
            ret_q = jr.retrieval_quality
            judge_raw = jr.to_json_dict()
        else:
            faith = ac_sim
            rel = aq_sim
            ret_q = None
            judge_raw = {"mode": "embedding_only"}

        metrics = EvaluationMetrics(
            faithfulness_score=faith,
            relevance_score=rel,
            retrieval_hit_at_k=ret_q,
            embedding_answer_context_sim=ac_sim,
            embedding_answer_query_sim=aq_sim,
            judge_raw=judge_raw,
        )
        return metrics, usage
