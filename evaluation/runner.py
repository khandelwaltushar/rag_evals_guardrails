"""Offline eval runner — ingests the corpus, runs each query, and reports metrics.

Usage:
    PYTHONPATH=. python -m evaluation.runner \
        --corpus data/sample_documents.json \
        --eval-set data/eval_set.json \
        --out data/eval_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from api.deps import build_app_state
from core.models import DocumentInput


@dataclass
class ItemResult:
    id: str
    query: str
    category: str
    gold_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    answer: str
    confidence: float
    faithfulness: float | None
    relevance: float | None
    guardrail_action: str
    latency_ms: float
    token_usage: dict[str, int]
    hit: bool
    out_of_scope_correct: bool | None


@dataclass
class Summary:
    provider: str
    embedder: str
    chat_model: str
    embedding_model: str
    n_items: int
    n_in_scope: int
    n_out_of_scope: int
    retrieval_recall_at_k: float
    avg_faithfulness: float | None
    avg_relevance: float | None
    avg_confidence: float
    oos_refusal_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    items: list[dict[str, Any]] = field(default_factory=list)


def _recall_hit(gold: list[str], retrieved: list[str]) -> bool:
    if not gold:
        return False
    gold_set = set(gold)
    return any(r in gold_set for r in retrieved)


def _is_refusal(answer: str, action: str) -> bool:
    if action in {"insufficient_info", "clarify"}:
        return True
    low = answer.lower()
    refusal_markers = (
        "cannot find",
        "can't find",
        "not in the provided context",
        "not found in",
        "insufficient information",
        "i do not have",
        "i don't have",
        "no information",
    )
    return any(m in low for m in refusal_markers)


async def _ingest_corpus(state, corpus_path: Path) -> None:
    data = json.loads(corpus_path.read_text())
    docs = [DocumentInput(**d) for d in data["documents"]]
    n_docs, n_chunks, errs = await state.ingestion.ingest(docs)
    if errs:
        raise RuntimeError(f"Ingestion errors: {errs}")
    print(f"  ingested: {n_docs} docs, {n_chunks} chunks")


async def _run_item(state, item: dict[str, Any], top_k: int) -> ItemResult:
    t0 = time.perf_counter()
    resp = await state.query_service.run(item["query"], top_k=top_k, skip_evaluation=False)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    retrieved_doc_ids = [c.doc_id for c in resp.retrieved_docs]
    gold = item.get("gold_doc_ids") or []
    is_oos = not gold

    em = resp.evaluation_metrics
    hit = _recall_hit(gold, retrieved_doc_ids) if not is_oos else False
    oos_correct = _is_refusal(resp.answer, resp.guardrail_action) if is_oos else None

    return ItemResult(
        id=item["id"],
        query=item["query"],
        category=item.get("category", "unknown"),
        gold_doc_ids=gold,
        retrieved_doc_ids=retrieved_doc_ids,
        answer=resp.answer,
        confidence=resp.confidence_score,
        faithfulness=em.faithfulness_score if em else None,
        relevance=em.relevance_score if em else None,
        guardrail_action=resp.guardrail_action,
        latency_ms=dt_ms,
        token_usage=resp.token_usage,
        hit=hit,
        out_of_scope_correct=oos_correct,
    )


def _summarize(items: list[ItemResult], state) -> Summary:
    in_scope = [i for i in items if i.gold_doc_ids]
    oos = [i for i in items if not i.gold_doc_ids]

    recall = (sum(1 for i in in_scope if i.hit) / len(in_scope)) if in_scope else 0.0
    oos_refusal = (
        sum(1 for i in oos if i.out_of_scope_correct) / len(oos) if oos else 0.0
    )

    faiths = [i.faithfulness for i in in_scope if i.faithfulness is not None]
    rels = [i.relevance for i in in_scope if i.relevance is not None]
    latencies = sorted(i.latency_ms for i in items)
    confs = [i.confidence for i in items]

    def _pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
        return xs[k]

    prompt_toks = [i.token_usage.get("prompt_tokens", 0) for i in items]
    compl_toks = [i.token_usage.get("completion_tokens", 0) for i in items]

    s = state.settings
    return Summary(
        provider=s.llm_provider.value,
        embedder=s.embedder_provider.value,
        chat_model=s.chat_model,
        embedding_model=s.embedding_model,
        n_items=len(items),
        n_in_scope=len(in_scope),
        n_out_of_scope=len(oos),
        retrieval_recall_at_k=recall,
        avg_faithfulness=statistics.mean(faiths) if faiths else None,
        avg_relevance=statistics.mean(rels) if rels else None,
        avg_confidence=statistics.mean(confs) if confs else 0.0,
        oos_refusal_rate=oos_refusal,
        latency_p50_ms=_pct(latencies, 50),
        latency_p95_ms=_pct(latencies, 95),
        avg_prompt_tokens=statistics.mean(prompt_toks) if prompt_toks else 0.0,
        avg_completion_tokens=statistics.mean(compl_toks) if compl_toks else 0.0,
        items=[i.__dict__ for i in items],
    )


def _print_table(s: Summary) -> None:
    def fmt(v, pct: bool = False, nd: int = 3) -> str:
        if v is None:
            return "n/a"
        if pct:
            return f"{v * 100:.1f}%"
        return f"{v:.{nd}f}"

    rows = [
        ("provider", f"{s.provider} / {s.chat_model}"),
        ("embedder", f"{s.embedder} / {s.embedding_model}"),
        ("items", f"{s.n_items}  (in-scope: {s.n_in_scope}, oos: {s.n_out_of_scope})"),
        ("retrieval recall@k", fmt(s.retrieval_recall_at_k, pct=True)),
        ("avg faithfulness (judge)", fmt(s.avg_faithfulness)),
        ("avg relevance (judge)", fmt(s.avg_relevance)),
        ("avg confidence", fmt(s.avg_confidence)),
        ("out-of-scope refusal rate", fmt(s.oos_refusal_rate, pct=True)),
        ("latency p50 (ms)", f"{s.latency_p50_ms:.0f}"),
        ("latency p95 (ms)", f"{s.latency_p95_ms:.0f}"),
        ("avg prompt tokens", f"{s.avg_prompt_tokens:.0f}"),
        ("avg completion tokens", f"{s.avg_completion_tokens:.0f}"),
    ]
    w = max(len(k) for k, _ in rows)
    print("\n=== Eval summary ===")
    for k, v in rows:
        print(f"  {k.ljust(w)}  {v}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/sample_documents.json"))
    ap.add_argument("--eval-set", type=Path, default=Path("data/eval_set.json"))
    ap.add_argument("--out", type=Path, default=Path("data/eval_results.json"))
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    eval_set = json.loads(args.eval_set.read_text())
    items = eval_set["items"]

    print("Loading app state (may take ~10s for local models)...")
    state = await build_app_state()
    print(f"  provider={state.settings.llm_provider.value} embedder={state.settings.embedder_provider.value}")

    print("Ingesting corpus...")
    await _ingest_corpus(state, args.corpus)

    print(f"Running {len(items)} eval items (top_k={args.top_k})...")
    results: list[ItemResult] = []
    for idx, item in enumerate(items, 1):
        r = await _run_item(state, item, top_k=args.top_k)
        marker = "OK " if (r.hit or (not r.gold_doc_ids and r.out_of_scope_correct)) else "MISS"
        print(f"  [{idx:02d}/{len(items)}] {marker} {r.id} ({r.latency_ms:.0f}ms) {r.query[:60]}")
        results.append(r)

    summary = _summarize(results, state)
    _print_table(summary)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary.__dict__, indent=2, default=str))
    print(f"\n  wrote detailed results to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
