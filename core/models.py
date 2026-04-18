"""Pydantic models for API and internal pipelines."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentInput(BaseModel):
    """Single document for ingestion."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "policy-2024",
                "text": "All customer data must be encrypted at rest and in transit.",
                "metadata": {"source": "internal"},
            }
        }
    )

    id: str = Field(description="Stable document identifier.")
    text: str = Field(description="Full text to chunk and index.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Opaque metadata stored with chunks.")


class IngestRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "documents": [
                    {
                        "id": "policy-2024",
                        "text": "All customer data must be encrypted at rest and in transit.",
                        "metadata": {"source": "internal"},
                    }
                ]
            }
        }
    )

    documents: list[DocumentInput] = Field(description="One or more documents to ingest.")


class IngestResponse(BaseModel):
    status: str
    documents_ingested: int
    chunks_created: int
    errors: list[str] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationMetrics(BaseModel):
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    retrieval_hit_at_k: float | None = Field(
        default=None,
        description="Heuristic retrieval quality 0-1 when labels available",
    )
    embedding_answer_context_sim: float | None = None
    embedding_answer_query_sim: float | None = None
    judge_raw: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What encryption is required for customer data?",
                "top_k": 8,
                "skip_evaluation": False,
            }
        }
    )

    query: str = Field(description="User question.")
    top_k: int | None = Field(default=None, description="Override default retrieval depth from config.")
    skip_evaluation: bool = Field(
        default=False,
        description="If true, skip LLM judge + embedding metrics (faster, lower cost).",
    )


class QueryResponse(BaseModel):
    answer: str
    retrieved_docs: list[RetrievedChunk]
    confidence_score: float
    evaluation_metrics: EvaluationMetrics | None = None
    guardrail_action: str = "none"
    trace: dict[str, Any] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)


class PipelineTrace(BaseModel):
    trace_id: str
    steps: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    prompt_preview: str | None = None
    debug: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
