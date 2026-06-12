"""Linear fixture connector: linear.json → Document stream.

Each Linear issue becomes one Document. Comments are inlined into the body so
a single retrieval hit returns the full conversational context.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from app.models import Document

SOURCE = "linear"


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _format_comment(c: dict) -> str:
    user = (c.get("user") or {}).get("name") or "unknown"
    when = c.get("createdAt", "")
    return f"  [{when}] {user}: {c.get('body', '')}"


def load(fixture_path: Path) -> Iterable[Document]:
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    for iss in data["issues"]:
        labels = [l["name"] for l in (iss.get("labels") or {}).get("nodes", [])]
        comments_nodes = (iss.get("comments") or {}).get("nodes", [])
        assignee = iss.get("assignee") or {}
        creator = iss.get("creator") or {}
        state = iss.get("state") or {}

        header_lines = [
            f"Status: {state.get('name', '?')} ({state.get('type', '?')})",
            f"Priority: {iss.get('priorityLabel', '?')}",
        ]
        if assignee:             header_lines.append(f"Assignee: {assignee.get('name', '?')}")
        if iss.get("dueDate"):   header_lines.append(f"Due: {iss['dueDate']}")
        if iss.get("blockedBy"): header_lines.append(f"Blocked by: {', '.join(iss['blockedBy'])}")
        if iss.get("blocks"):    header_lines.append(f"Blocks: {', '.join(iss['blocks'])}")
        if labels:               header_lines.append(f"Labels: {', '.join(labels)}")
        if iss.get("project"):   header_lines.append(f"Project: {iss['project'].get('name','?')}")

        body_parts = ["\n".join(header_lines), iss.get("description") or ""]
        if comments_nodes:
            body_parts.append("Comments:\n" + "\n".join(_format_comment(c) for c in comments_nodes))
        body = "\n\n".join(p for p in body_parts if p).strip()

        people: set[str] = set()
        if assignee.get("email"): people.add(assignee["email"])
        if creator.get("email"):  people.add(creator["email"])
        for c in comments_nodes:
            email = (c.get("user") or {}).get("email")
            if email:
                people.add(email)

        created_at = _parse_iso(iss["createdAt"])
        updated_at = _parse_iso(iss.get("updatedAt")) or created_at

        yield Document(
            id=f"linear:{iss['identifier']}",
            source="linear",
            type="issue",
            title=f"[{iss['identifier']}] {iss['title']}",
            body=body,
            author=creator.get("email"),
            participants=sorted(people),
            created_at=created_at,
            updated_at=updated_at,
            metadata={
                "identifier": iss["identifier"],
                "state": state.get("name"),
                "state_type": state.get("type"),
                "priority": iss.get("priority"),
                "priority_label": iss.get("priorityLabel"),
                "due_date": iss.get("dueDate"),
                "started_at": iss.get("startedAt"),
                "completed_at": iss.get("completedAt"),
                "labels": labels,
                "blocked_by": iss.get("blockedBy", []),
                "blocks": iss.get("blocks", []),
                "assignee_name": assignee.get("name"),
                "assignee_email": assignee.get("email"),
                "creator_name": creator.get("name"),
                "creator_email": creator.get("email"),
                "project": (iss.get("project") or {}).get("name"),
                "url": iss.get("url"),
                "comment_count": len(comments_nodes),
            },
        )
