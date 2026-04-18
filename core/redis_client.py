"""Lazy Redis connection for optional caching."""

from __future__ import annotations

from typing import Any

from core.logging_config import get_logger

logger = get_logger(__name__)


async def create_redis(url: str | None) -> Any | None:
    if not url:
        return None
    try:
        import redis.asyncio as redis

        client = redis.from_url(url, decode_responses=True)
        await client.ping()
        return client
    except Exception as e:
        logger.warning("redis_unavailable", url=url, error=str(e))
        return None
