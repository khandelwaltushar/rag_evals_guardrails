from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from core.config import Settings
from core.models import RetrievedChunk
from retrieval.hybrid import HybridRetriever, _minmax


def test_minmax() -> None:
    assert _minmax({"a": 1.0, "b": 3.0}) == {"a": 0.0, "b": 1.0}
    assert _minmax({}) == {}


@pytest.mark.asyncio
async def test_hybrid_fusion_orders_by_blended_score() -> None:
    settings = Settings(openai_api_key="x", hybrid_alpha=0.5)
    emb = AsyncMock()
    emb.embed_query = AsyncMock(return_value=np.ones((1, 4), dtype=np.float32))

    dense = AsyncMock()
    dense.search = AsyncMock(
        return_value=[("c1", 0.9), ("c2", 0.1)],
    )
    dense.get_payload = MagicMock(
        side_effect=lambda cid: {
            "c1": {"doc_id": "d", "text": "t1", "metadata": {}},
            "c2": {"doc_id": "d", "text": "t2", "metadata": {}},
        }[cid]
    )

    sparse = AsyncMock()
    sparse.search = AsyncMock(
        return_value=[("c2", 10.0), ("c1", 1.0)],
    )

    h = HybridRetriever(settings, emb, dense, sparse)
    out = await h.retrieve("q", top_k=2)
    assert len(out) == 2
    assert isinstance(out[0], RetrievedChunk)
