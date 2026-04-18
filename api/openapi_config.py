"""OpenAPI / Swagger UI metadata — single place for API documentation."""

OPENAPI_TAGS = [
    {
        "name": "health",
        "description": "Process and dependency checks for orchestration (K8s, load balancers).",
    },
    {
        "name": "ingestion",
        "description": "Chunk, embed, and index documents into FAISS (dense) and BM25 (sparse).",
    },
    {
        "name": "query",
        "description": "Rewrite query, hybrid retrieval, optional rerank, LLM answer, evaluation, guardrails.",
    },
]

SWAGGER_UI_PARAMETERS: dict[str, str | bool | int] = {
    "displayRequestDuration": True,
    "persistAuthorization": True,
    "defaultModelsExpandDepth": 2,
    "defaultModelExpandDepth": 2,
    "docExpansion": "list",
    "filter": True,
    "syntaxHighlight.theme": "monokai",
}

API_DESCRIPTION = """
## RAG service

Production-oriented pipeline: **ingestion** (recursive or semantic chunking → OpenAI embeddings → FAISS + BM25),
**query** (rewrite → hybrid retrieval → optional rerank → generation → LLM-as-judge + embedding metrics → guardrails).

### Auth

Set `OPENAI_API_KEY` in the server environment (or `.env`). Optional: `ANTHROPIC_API_KEY` when `LLM_PROVIDER=anthropic`.

### Docs

- **Swagger UI** — this page (`/docs`)
- **ReDoc** — `/redoc`
- **OpenAPI JSON** — `/openapi.json`
"""
