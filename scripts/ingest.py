"""Wipe the documents + chunks tables and re-ingest all fixtures into SQLite.

Usage:
    python scripts/ingest.py

After upserting documents from each connector, runs a single batched pass that
chunks every body and embeds all chunks in one call to the embedder. Single-
batch is meaningfully faster than per-document calls (model overhead dominates
small batches).
"""

from __future__ import annotations

import importlib
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.embed.embedder import encode  # noqa: E402
from app.models import Document  # noqa: E402
from app.retrieve.chunker import chunk_document  # noqa: E402
from app.store.db import connect, init_db  # noqa: E402
from app.store.repository import clear_chunks, insert_chunk  # noqa: E402

CONNECTORS: list[tuple[str, str]] = [
    ("gmail",  "gmail.json"),
    ("gcal",   "gcal.json"),
    ("linear", "linear.json"),
    ("slack",  "slack.json"),
]


_UPSERT_SQL = """
INSERT INTO documents (id, source, type, title, body, author, participants, created_at, updated_at, metadata)
VALUES (:id, :source, :type, :title, :body, :author, :participants, :created_at, :updated_at, :metadata)
ON CONFLICT(id) DO UPDATE SET
    source       = excluded.source,
    type         = excluded.type,
    title        = excluded.title,
    body         = excluded.body,
    author       = excluded.author,
    participants = excluded.participants,
    created_at   = excluded.created_at,
    updated_at   = excluded.updated_at,
    metadata     = excluded.metadata
"""


def upsert_document(conn, doc: Document) -> None:
    conn.execute(_UPSERT_SQL, doc.to_row())


def main() -> int:
    init_db()
    fixtures_dir = settings.fixtures_dir
    print(f"Ingesting from: {fixtures_dir}")
    print(f"Store:          {settings.db_path}\n")

    started = time.perf_counter()
    counts: Counter[str] = Counter()
    pending_docs: list[tuple[str, str | None, str]] = []

    with connect() as conn:
        conn.execute("BEGIN")
        conn.execute("DELETE FROM documents")
        conn.execute("DELETE FROM answer_cache")
        clear_chunks(conn)
        print("  cache cleared")

        for source, filename in CONNECTORS:
            path = fixtures_dir / filename
            if not path.exists():
                print(f"  [skip] {source:<6} -- fixture not found at {path}")
                continue
            module = importlib.import_module(f"app.ingest.{source}")
            for doc in module.load(path):
                upsert_document(conn, doc)
                pending_docs.append((doc.id, doc.title, doc.body))
                counts[source] += 1
            print(f"  [ok]   {source:<6} -- {counts[source]:>3} documents from {filename}")

        print("\nChunking + embedding…")
        embed_started = time.perf_counter()
        all_chunks: list[tuple[str, int, str]] = []
        for doc_id, title, body in pending_docs:
            for ch in chunk_document(title, body):
                all_chunks.append((doc_id, ch.index, ch.text))

        if all_chunks:
            texts = [t for _, _, t in all_chunks]
            vectors = encode(texts)
            for (doc_id, idx, text), vec in zip(all_chunks, vectors, strict=True):
                insert_chunk(conn, doc_id, idx, text, vec)

        conn.execute("COMMIT")

        total_row    = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        chunk_row    = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        embedded_row = conn.execute("SELECT COUNT(*) AS n FROM chunks WHERE embedding IS NOT NULL").fetchone()
        fts_row      = conn.execute("SELECT COUNT(*) AS n FROM documents_fts").fetchone()

    elapsed = time.perf_counter() - started
    embed_elapsed = time.perf_counter() - embed_started
    print(f"  chunks: {chunk_row['n']} (embedded: {embedded_row['n']})  in {embed_elapsed:.2f}s")
    print(f"\nTotal documents in store: {total_row['n']}")
    print(f"FTS5 rows:                {fts_row['n']}")
    print(f"Elapsed (total):          {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
