"""Shared fixtures.

require_ingested_db: skip any test that needs the populated SQLite store
when the DB doesn't exist yet or is empty. Tests should depend on this
fixture instead of crashing with cryptic sqlite errors during local runs.
"""

from __future__ import annotations

import pytest

from app.store.db import connect


@pytest.fixture(scope="session")
def require_ingested_db() -> None:
    try:
        with connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
    except Exception as exc:
        pytest.skip(f"app.db not initialized ({exc}). Run scripts/ingest.py first.")
        return
    if not row or int(row["n"]) == 0:
        pytest.skip("documents table empty. Run scripts/ingest.py first.")
