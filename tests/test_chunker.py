import pytest

from core.config import Settings
from core.models import DocumentInput
from ingestion.chunkers.recursive import RecursiveChunker


@pytest.mark.asyncio
async def test_recursive_chunker_produces_chunks() -> None:
    s = Settings(openai_api_key="x")
    c = RecursiveChunker(s)
    doc = DocumentInput(
        id="d1",
        text="First sentence here. Second sentence here.\n\nThird paragraph with more text " * 20,
    )
    chunks = await c.chunk(doc)
    assert len(chunks) >= 1
    assert all(ch.doc_id == "d1" for ch in chunks)
    assert chunks[0].metadata.get("strategy") == "recursive"
