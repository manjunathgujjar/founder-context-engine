"""Connector protocol.

Each connector module under app/ingest/ exports:
    SOURCE: Source                            -- one of 'gmail' | 'gcal' | 'linear' | 'slack'
    load(fixture_path: Path) -> Iterable[Document]

The ingest orchestrator (scripts/ingest.py) imports each module by source name
and calls load() to obtain normalized Documents for the SQLite store.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from app.models import Document, Source


class Connector(Protocol):
    SOURCE: Source

    @staticmethod
    def load(fixture_path: Path) -> Iterable[Document]: ...
