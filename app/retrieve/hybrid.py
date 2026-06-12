"""Hybrid retrieval: BM25 (FTS5) + cosine (vector) merged via Reciprocal Rank Fusion.

Why hybrid? Pure dense retrieval misses literal identifiers (ENG-142, customer
names, dollar amounts). Pure BM25 misses paraphrase ("blocked task" vs "stuck
ticket"). RRF combines both without per-lane weight tuning.

Why RRF and not weighted sum? FTS5 BM25 and cosine scores live on
incomparable scales. RRF only uses rank order, so it's robust to that mismatch
and to per-query score distribution shifts.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from app.config import settings
from app.embed.embedder import encode_one
from app.store.repository import ChunkHit, fts_search, vector_search

RRF_K = 60


@dataclass(slots=True)
class FusedHit:
    document_id: str
    source: str
    type: str
    title: str | None
    chunk_text: str
    fused_score: float
    fts_rank: int | None
    vector_rank: int | None
    document_metadata: dict
    document_created_at: str
    document_updated_at: str


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    top_k: int | None = None,
    fts_limit: int = 20,
    vector_limit: int = 20,
) -> list[FusedHit]:
    """Run both lanes, fuse via RRF, dedupe by document_id."""
    top_k = top_k or settings.retrieval_top_k

    fts_hits = fts_search(conn, query, limit=fts_limit)
    vec_hits = vector_search(conn, encode_one(query), limit=vector_limit)

    return _fuse(fts_hits, vec_hits, top_k=top_k)


def _fuse(
    fts_hits: list[ChunkHit],
    vec_hits: list[ChunkHit],
    *,
    top_k: int,
) -> list[FusedHit]:
    """RRF: score(doc) = sum over lanes of 1 / (k + rank). Dedupe by document_id."""
    agg: dict[str, dict] = {}

    for hit in fts_hits:
        agg.setdefault(hit.document_id, _empty(hit))
        slot = agg[hit.document_id]
        if slot["fts_rank"] is None or hit.rank < slot["fts_rank"]:
            slot["fts_rank"] = hit.rank
            if not slot["chunk_text"]:
                slot["chunk_text"] = hit.chunk_text

    for hit in vec_hits:
        agg.setdefault(hit.document_id, _empty(hit))
        slot = agg[hit.document_id]
        if slot["vector_rank"] is None or hit.rank < slot["vector_rank"]:
            slot["vector_rank"] = hit.rank
            slot["chunk_text"] = hit.chunk_text

    fused: list[FusedHit] = []
    for doc_id, slot in agg.items():
        score = 0.0
        if slot["fts_rank"] is not None:
            score += 1.0 / (RRF_K + slot["fts_rank"])
        if slot["vector_rank"] is not None:
            score += 1.0 / (RRF_K + slot["vector_rank"])
        fused.append(
            FusedHit(
                document_id=doc_id,
                source=slot["source"],
                type=slot["type"],
                title=slot["title"],
                chunk_text=slot["chunk_text"],
                fused_score=score,
                fts_rank=slot["fts_rank"],
                vector_rank=slot["vector_rank"],
                document_metadata=slot["document_metadata"],
                document_created_at=slot["document_created_at"],
                document_updated_at=slot["document_updated_at"],
            )
        )

    fused.sort(key=lambda h: h.fused_score, reverse=True)
    return fused[:top_k]


def _empty(hit: ChunkHit) -> dict:
    """Seed a per-document aggregation slot from any hit's parent metadata."""
    return {
        "source": hit.source,
        "type": hit.type,
        "title": hit.title,
        "chunk_text": "",
        "fts_rank": None,
        "vector_rank": None,
        "document_metadata": hit.document_metadata,
        "document_created_at": hit.document_created_at,
        "document_updated_at": hit.document_updated_at,
    }
