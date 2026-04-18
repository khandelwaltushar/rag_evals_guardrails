"""FastAPI entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

import json

import openai
from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from api.deps import AppState, build_app_state
from api.errors import openai_api_error_handler
from api.openapi_config import API_DESCRIPTION, OPENAPI_TAGS, SWAGGER_UI_PARAMETERS
from core.logging_config import configure_logging, get_logger
from core.models import IngestRequest, IngestResponse, QueryRequest, QueryResponse

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    st = await build_app_state()
    configure_logging(debug=st.settings.rag_debug)
    app.state.app_state = st
    logger.info("app_started", rag_debug=st.settings.rag_debug)
    yield
    r = getattr(app.state.app_state, "redis", None)
    if r is not None:
        try:
            await r.close()
        except Exception:
            pass


app = FastAPI(
    title="RAG Evals & Guardrails API",
    description=API_DESCRIPTION.strip(),
    version="0.1.0",
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
    swagger_ui_parameters=SWAGGER_UI_PARAMETERS,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={"name": "RAG service"},
    license_info={"name": "See project LICENSE"},
)

app.add_exception_handler(openai.APIError, openai_api_error_handler)


def get_state(request: Request) -> AppState:
    return request.app.state.app_state


@app.get(
    "/health",
    tags=["health"],
    summary="Liveness probe",
    response_description="Service is accepting traffic.",
)
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/",
    include_in_schema=False,
    response_class=HTMLResponse,
)
async def root() -> str:
    """Redirect browsers to Swagger UI."""
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>RAG API</title></head><body>"
        "<p>Open <a href='/docs'>Swagger UI</a> or <a href='/redoc'>ReDoc</a>.</p>"
        "</body></html>"
    )


@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["ingestion"],
    summary="Ingest documents",
    response_description="Counts of documents and chunks indexed; errors list partial failures.",
)
async def ingest(req: IngestRequest, state: AppState = Depends(get_state)) -> IngestResponse:
    docs, chunks, errors = await state.ingestion.ingest(req.documents)
    status = "ok" if not errors else "partial"
    if docs == 0 and errors:
        status = "error"
    return IngestResponse(
        status=status,
        documents_ingested=docs,
        chunks_created=chunks,
        errors=errors,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["query"],
    summary="Run RAG query",
    response_description="Answer, retrieved chunks, confidence, optional evaluation metrics and trace.",
)
async def query(req: QueryRequest, state: AppState = Depends(get_state)) -> QueryResponse:
    return await state.query_service.run(
        req.query,
        skip_evaluation=req.skip_evaluation,
        top_k=req.top_k,
    )


@app.post(
    "/query_stream",
    tags=["query"],
    summary="Run RAG query with streamed tokens (SSE)",
    response_description=(
        "text/event-stream with events: 'retrieved' (docs), 'token' (text delta), "
        "'done' (confidence + guardrail action). Evaluation is skipped in streaming mode."
    ),
)
async def query_stream(req: QueryRequest, state: AppState = Depends(get_state)) -> StreamingResponse:
    async def _gen():
        try:
            async for event, payload in state.query_service.run_stream(req.query, top_k=req.top_k):
                yield f"event: {event}\ndata: {json.dumps(payload)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")
