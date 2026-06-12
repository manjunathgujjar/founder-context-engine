"""Normalized cross-source document schema.

Every connector (gmail, gcal, linear, slack) emits this same shape so
storage, retrieval, and synthesis don't need to know about the source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

Source = Literal["gmail", "gcal", "linear", "slack"]


@dataclass(slots=True)
class Document:
    id: str
    source: Source
    type: str
    title: str | None
    body: str
    author: str | None
    participants: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now())
    updated_at: datetime = field(default_factory=lambda: datetime.now())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "type": self.type,
            "title": self.title,
            "body": self.body,
            "author": self.author,
            "participants": json.dumps(self.participants, ensure_ascii=False),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": json.dumps(self.metadata, ensure_ascii=False),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Document:
        return cls(
            id=row["id"],
            source=row["source"],
            type=row["type"],
            title=row["title"],
            body=row["body"],
            author=row["author"],
            participants=json.loads(row["participants"] or "[]"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row["metadata"] or "{}"),
        )
