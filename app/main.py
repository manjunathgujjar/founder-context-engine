"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api import ask, health, search, stats
from app.store.db import init_db

WEB_DIR = Path(__file__).parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Founder AI Assistant",
    description="Local context engine for a startup founder. Grounded answers over a SQLite store.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(ask.router, prefix="/api", tags=["ask"])
app.include_router(stats.router, prefix="/api", tags=["stats"])


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
