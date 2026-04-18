"""Integration-style test with mocked pipeline (no real OpenAI)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.deps import AppState
from core.config import Settings
from core.models import QueryResponse, RetrievedChunk


@pytest.fixture
def mock_state(monkeypatch: pytest.MonkeyPatch) -> AppState:
    settings = Settings(openai_api_key="test")
    qs = AsyncMock()
    qs.run = AsyncMock(
        return_value=QueryResponse(
            answer="mocked",
            retrieved_docs=[
                RetrievedChunk(
                    chunk_id="1",
                    doc_id="d",
                    text="ctx",
                    score=0.9,
                )
            ],
            confidence_score=0.8,
            guardrail_action="none",
        )
    )
    st = MagicMock(spec=AppState)
    st.settings = settings
    st.redis = None
    st.ingestion = AsyncMock()
    st.ingestion.ingest = AsyncMock(return_value=(1, 3, []))
    st.query_service = qs
    return st


def test_health() -> None:
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_ingest_and_query_mocked(mock_state: AppState, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_build() -> AppState:
        return mock_state

    monkeypatch.setattr("api.main.build_app_state", fake_build)

    with TestClient(app) as client:
        r = client.post(
            "/ingest",
            json={
                "documents": [
                    {"id": "a", "text": "hello world", "metadata": {}},
                ]
            },
        )
        assert r.status_code == 200
        assert r.json()["chunks_created"] == 3

        q = client.post("/query", json={"query": "what?"})
        assert q.status_code == 200
        body = q.json()
        assert body["answer"] == "mocked"
        assert body["confidence_score"] == 0.8
