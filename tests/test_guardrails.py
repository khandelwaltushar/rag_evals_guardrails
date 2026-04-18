import pytest

from core.config import Settings
from core.models import EvaluationMetrics, RetrievedChunk
from guardrails.pipeline import GuardrailPipeline


@pytest.mark.asyncio
async def test_guardrail_insufficient_when_no_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    s = Settings(openai_api_key="x", min_confidence_threshold=0.1)
    g = GuardrailPipeline(s)

    async def no_self(*_a, **_k):
        return 1.0, {"prompt_tokens": 0, "completion_tokens": 0}

    monkeypatch.setattr("guardrails.pipeline.self_evaluate", no_self)

    ans, conf, action = await g.apply(
        "q?",
        "draft",
        [],
        EvaluationMetrics(
            faithfulness_score=0.9,
            relevance_score=0.9,
            embedding_answer_context_sim=0.9,
        ),
    )
    assert action == "insufficient_info"
    assert "enough information" in ans.lower()
    assert conf <= 0.2
