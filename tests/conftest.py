import os

import pytest


@pytest.fixture(autouse=True)
def env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "sk-test-fake-key-for-unit-tests"))
    # Stable defaults so tests ignore a developer .env with LLM_PROVIDER=anthropic
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("CHAT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4o-mini")
