"""Wipe the documents table and re-ingest all fixtures into SQLite.

Usage:
    python scripts/ingest.py

Reads JSON fixtures from data/fixtures/{gmail,gcal,linear,slack}.json,
normalizes them through the per-source connectors, and writes Documents
to the SQLite store. FTS5 stays in sync via the schema's triggers.

This is intentionally destructive (DELETE FROM documents at the start)
so reviewers get a clean, reproducible state on every run.
"""

from __future__ import annotations

import importlib
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.models import Document  # noqa: E402
from app.store.db import connect, init_db  # noqa: E402

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


def upsert(conn, doc: Document) -> None:
    conn.execute(_UPSERT_SQL, doc.to_row())


def main() -> int:
    init_db()
    fixtures_dir = settings.fixtures_dir
    print(f"Ingesting from: {fixtures_dir}")
    print(f"Store:          {settings.db_path}\n")

    started = time.perf_counter()
    counts: Counter[str] = Counter()

    with connect() as conn:
        conn.execute("BEGIN")
        conn.execute("DELETE FROM documents")

        for source, filename in CONNECTORS:
            path = fixtures_dir / filename
            if not path.exists():
                print(f"  [skip] {source:<6} -- fixture not found at {path}")
                continue
            module = importlib.import_module(f"app.ingest.{source}")
            for doc in module.load(path):
                upsert(conn, doc)
                counts[source] += 1
            print(f"  [ok]   {source:<6} -- {counts[source]:>3} documents from {filename}")

        conn.execute("COMMIT")

        total_row = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        fts_row = conn.execute("SELECT COUNT(*) AS n FROM documents_fts").fetchone()

    elapsed = time.perf_counter() - started
    print(f"\nTotal documents in store: {total_row['n']}")
    print(f"FTS5 rows:                {fts_row['n']}")
    print(f"Elapsed:                  {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
