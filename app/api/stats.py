"""Stats endpoint: aggregate request log.

GET /api/stats
    -> { total_queries, cache_hits, cache_hit_rate, refusals,
         total_cost, avg_total_seconds, avg_llm_seconds_uncached,
         avg_retrieve_seconds, total_prompt_tokens, total_completion_tokens }
"""

from __future__ import annotations

from fastapi import APIRouter

from app.store.db import connect
from app.store.repository import get_stats

router = APIRouter()


@router.get("/stats")
def stats() -> dict:
    with connect() as conn:
        return get_stats(conn)
