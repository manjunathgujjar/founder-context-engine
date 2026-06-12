"""Health + status endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.store.db import connect

router = APIRouter()


@router.get("/health")
def health() -> dict:
    db_exists = settings.db_path.exists()
    document_count = 0
    if db_exists:
        with connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
            document_count = row["n"]
    return {
        "status": "ok",
        "db_path": str(settings.db_path),
        "db_exists": db_exists,
        "documents": document_count,
    }
