FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN pip install -r requirements.txt

COPY api ./api
COPY core ./core
COPY evaluation ./evaluation
COPY guardrails ./guardrails
COPY ingestion ./ingestion
COPY llm ./llm
COPY retrieval ./retrieval
COPY data ./data

RUN useradd --create-home --uid 1001 app && chown -R app /app
USER app

ENV PYTHONPATH=/app \
    FAISS_INDEX_PATH=/app/data/faiss_index \
    BM25_CORPUS_PATH=/app/data/bm25_corpus.json

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
