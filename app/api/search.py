"""Debug retrieval endpoint.

Exposes hybrid_search() over HTTP so reviewers can probe retrieval behavior
without going through the LLM synthesis layer. Returns raw FTS / vector ranks
alongside the fused score so it's also a tuning tool.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.retrieve.hybrid import hybrid_search
from app.store.db import connect

router = APIRouter()


class SearchHit(BaseModel):
    document_id: str
    source: str
    type: str
    title: str | None
    chunk_text: str
    fused_score: float
    fts_rank: int | None
    vector_rank: int | None
    document_created_at: str
    document_updated_at: str
    document_metadata: dict


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Free-text query"),
    k: int = Query(8, ge=1, le=50, description="Top-k after fusion"),
    fts_limit: int = Query(20, ge=1, le=100, description="Per-lane BM25 top-k"),
    vector_limit: int = Query(20, ge=1, le=100, description="Per-lane vector top-k"),
) -> SearchResponse:
    with connect() as conn:
        fused = hybrid_search(conn, q, top_k=k, fts_limit=fts_limit, vector_limit=vector_limit)

    return SearchResponse(
        query=q,
        hits=[
            SearchHit(
                document_id=h.document_id,
                source=h.source,
                type=h.type,
                title=h.title,
                chunk_text=h.chunk_text,
                fused_score=h.fused_score,
                fts_rank=h.fts_rank,
                vector_rank=h.vector_rank,
                document_created_at=h.document_created_at,
                document_updated_at=h.document_updated_at,
                document_metadata=h.document_metadata,
            )
            for h in fused
        ],
    )
