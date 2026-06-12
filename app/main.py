"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health
from app.store.db import init_db


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
