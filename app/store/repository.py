"""Storage primitives over the SQLite documents/chunks tables.

This module is the only place we hand-write SQL outside scripts/. Higher layers
(retrieve/, answer/) consume these typed helpers and never touch sqlite3 directly.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from app.embed.embedder import from_blob


@dataclass(slots=True)
class ChunkHit:
    """A single retrieval result: a chunk plus its parent document context."""
    chunk_id: int
    document_id: str
    source: str
    type: str
    title: str | None
    chunk_text: str
    score: float
    rank: int
    document_metadata: dict[str, Any]
    document_created_at: str
    document_updated_at: str


def insert_chunk(
    conn: sqlite3.Connection,
    document_id: str,
    chunk_index: int,
    text: str,
    embedding: np.ndarray | None,
) -> int:
    """Insert (or replace) one chunk + its embedding. Returns chunk row id."""
    blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
    conn.execute(
        """
        INSERT INTO chunks (document_id, chunk_index, text, embedding)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(document_id, chunk_index) DO UPDATE SET
            text      = excluded.text,
            embedding = excluded.embedding
        """,
        (document_id, chunk_index, text, blob),
    )
    row = conn.execute(
        "SELECT id FROM chunks WHERE document_id = ? AND chunk_index = ?",
        (document_id, chunk_index),
    ).fetchone()
    return int(row["id"])


def clear_chunks(conn: sqlite3.Connection) -> None:
    """Wipe all chunks (for clean re-ingest)."""
    conn.execute("DELETE FROM chunks")


def fts_search(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[ChunkHit]:
    """BM25 over documents_fts. Returns one hit per matching document (chunk 0).

    FTS5 indexes title+body of the parent document, not chunks. We surface
    chunk 0 as a representative, which works because the answer pipeline
    reconstitutes context from the parent document's full body anyway.
    """
    safe = _escape_fts(query)
    if not safe:
        return []
    rows = conn.execute(
        """
        SELECT
            d.id           AS document_id,
            d.source       AS source,
            d.type         AS type,
            d.title        AS title,
            d.metadata     AS metadata,
            d.created_at   AS created_at,
            d.updated_at   AS updated_at,
            c.id           AS chunk_id,
            c.text         AS chunk_text,
            bm25(documents_fts) AS score
        FROM documents_fts
        JOIN documents d  ON d.rowid = documents_fts.rowid
        LEFT JOIN chunks c ON c.document_id = d.id AND c.chunk_index = 0
        WHERE documents_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (safe, limit),
    ).fetchall()
    return [
        ChunkHit(
            chunk_id=int(r["chunk_id"]) if r["chunk_id"] is not None else -1,
            document_id=r["document_id"],
            source=r["source"],
            type=r["type"],
            title=r["title"],
            chunk_text=r["chunk_text"] or "",
            score=float(r["score"]),
            rank=rank,
            document_metadata=json.loads(r["metadata"] or "{}"),
            document_created_at=r["created_at"],
            document_updated_at=r["updated_at"],
        )
        for rank, r in enumerate(rows, start=1)
    ]


def vector_search(
    conn: sqlite3.Connection,
    query_vec: np.ndarray,
    limit: int = 20,
) -> list[ChunkHit]:
    """Cosine top-k via brute-force dot product over all stored chunk embeddings.

    Embeddings are L2-normalized at ingest, so cosine = dot. With ~100 docs the
    full matmul is microseconds; revisit ANN once the corpus is 10x+.
    """
    rows = conn.execute(
        """
        SELECT
            c.id           AS chunk_id,
            c.document_id  AS document_id,
            c.text         AS chunk_text,
            c.embedding    AS embedding,
            d.source       AS source,
            d.type         AS type,
            d.title        AS title,
            d.metadata     AS metadata,
            d.created_at   AS created_at,
            d.updated_at   AS updated_at
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.embedding IS NOT NULL
        """,
    ).fetchall()
    if not rows:
        return []

    matrix = np.stack([from_blob(r["embedding"]) for r in rows]).astype(np.float32)
    q = query_vec.astype(np.float32)
    scores = matrix @ q
    order = np.argsort(-scores)[:limit]

    hits: list[ChunkHit] = []
    for rank, idx in enumerate(order, start=1):
        r = rows[int(idx)]
        hits.append(
            ChunkHit(
                chunk_id=int(r["chunk_id"]),
                document_id=r["document_id"],
                source=r["source"],
                type=r["type"],
                title=r["title"],
                chunk_text=r["chunk_text"] or "",
                score=float(scores[int(idx)]),
                rank=rank,
                document_metadata=json.loads(r["metadata"] or "{}"),
                document_created_at=r["created_at"],
                document_updated_at=r["updated_at"],
            )
        )
    return hits


def get_document(conn: sqlite3.Connection, document_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["participants"] = json.loads(d["participants"] or "[]")
    d["metadata"] = json.loads(d["metadata"] or "{}")
    return d


def list_sources(conn: sqlite3.Connection) -> list[dict]:
    """Counts per source, for the /api/sources endpoint."""
    rows = conn.execute(
        "SELECT source, COUNT(*) AS n FROM documents GROUP BY source ORDER BY source"
    ).fetchall()
    return [{"source": r["source"], "documents": int(r["n"])} for r in rows]


# --- answer cache + request log + stats -------------------------------------


def cache_get(conn: sqlite3.Connection, question_hash: str, *, ttl: timedelta) -> str | None:
    """Return serialized AnswerResult JSON if within TTL, else None (and evict)."""
    row = conn.execute(
        "SELECT answer_json, created_at FROM answer_cache WHERE question_hash = ?",
        (question_hash,),
    ).fetchone()
    if not row:
        return None
    created = datetime.fromisoformat(row["created_at"])
    if datetime.now(timezone.utc) - created > ttl:
        conn.execute("DELETE FROM answer_cache WHERE question_hash = ?", (question_hash,))
        return None
    return row["answer_json"]


def cache_put(conn: sqlite3.Connection, question_hash: str, question: str, answer_json: str) -> None:
    conn.execute(
        """
        INSERT INTO answer_cache (question_hash, question, answer_json, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(question_hash) DO UPDATE SET
            answer_json = excluded.answer_json,
            created_at  = excluded.created_at
        """,
        (question_hash, question, answer_json, datetime.now(timezone.utc).isoformat()),
    )


def log_request(
    conn: sqlite3.Connection,
    question: str,
    *,
    cache_hit: bool,
    refused: bool,
    top_fused_score: float | None,
    cost: float,
    prompt_tokens: int,
    completion_tokens: int,
    retrieve_seconds: float,
    llm_seconds: float,
    total_seconds: float,
) -> None:
    conn.execute(
        """
        INSERT INTO request_log
            (ts, question, cache_hit, refused, top_fused_score, cost,
             prompt_tokens, completion_tokens, retrieve_seconds, llm_seconds, total_seconds)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            question,
            int(cache_hit),
            int(refused),
            top_fused_score,
            cost,
            prompt_tokens,
            completion_tokens,
            retrieve_seconds,
            llm_seconds,
            total_seconds,
        ),
    )


def get_stats(conn: sqlite3.Connection) -> dict:
    """Aggregate metrics for /api/stats."""
    row = conn.execute(
        """
        SELECT
            COUNT(*)                                                       AS total_queries,
            COALESCE(SUM(cache_hit), 0)                                    AS cache_hits,
            COALESCE(SUM(refused), 0)                                      AS refusals,
            COALESCE(SUM(cost), 0)                                         AS total_cost,
            COALESCE(AVG(total_seconds), 0)                                AS avg_total_seconds,
            COALESCE(AVG(CASE WHEN cache_hit = 0 THEN llm_seconds END), 0) AS avg_llm_seconds_uncached,
            COALESCE(AVG(retrieve_seconds), 0)                             AS avg_retrieve_seconds,
            COALESCE(SUM(prompt_tokens), 0)                                AS total_prompt_tokens,
            COALESCE(SUM(completion_tokens), 0)                            AS total_completion_tokens
        FROM request_log
        """
    ).fetchone()
    total = int(row["total_queries"] or 0)
    hits = int(row["cache_hits"] or 0)
    return {
        "total_queries":             total,
        "cache_hits":                hits,
        "cache_hit_rate":            (hits / total) if total else 0.0,
        "refusals":                  int(row["refusals"] or 0),
        "total_cost":                round(float(row["total_cost"] or 0.0), 6),
        "avg_total_seconds":         round(float(row["avg_total_seconds"] or 0.0), 3),
        "avg_llm_seconds_uncached":  round(float(row["avg_llm_seconds_uncached"] or 0.0), 3),
        "avg_retrieve_seconds":      round(float(row["avg_retrieve_seconds"] or 0.0), 4),
        "total_prompt_tokens":       int(row["total_prompt_tokens"] or 0),
        "total_completion_tokens":   int(row["total_completion_tokens"] or 0),
    }


_FTS_BAD = set('"():*^+-')


def _escape_fts(query: str) -> str:
    """Sanitize free-text query for FTS5 MATCH.

    FTS5 has its own query syntax (NEAR, column filters, quoted phrases).
    We coerce arbitrary user input into a safe OR of bare terms, dropping
    operator characters that would otherwise raise SQLite errors.
    """
    tokens = []
    for raw in query.split():
        cleaned = "".join(ch for ch in raw if ch not in _FTS_BAD).strip()
        if cleaned:
            tokens.append(f'"{cleaned}"')
    return " OR ".join(tokens)
