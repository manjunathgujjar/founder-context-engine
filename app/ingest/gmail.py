"""Gmail fixture connector: gmail.json → Document stream."""

from __future__ import annotations

import email.utils
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from app.models import Document

SOURCE = "gmail"


def _header(headers: list[dict], name: str) -> str:
    """Case-insensitive header lookup. Returns '' if absent."""
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def _extract_emails(value: str) -> list[str]:
    """Parse a comma-separated address header into a list of email addresses."""
    if not value:
        return []
    return [addr for _, addr in email.utils.getaddresses([value]) if addr]


def _extract_body(payload: dict) -> str:
    """Pull plaintext body from payload.body.data or recursively from text/plain parts.

    Fixture stores plaintext directly in body.data. Real Gmail returns base64url —
    a future connector would .decode() here.
    """
    body = payload.get("body", {})
    if body.get("data"):
        return body["data"]
    for part in payload.get("parts", []):
        text = _extract_body(part)
        if text:
            return text
    return ""


def load(fixture_path: Path) -> Iterable[Document]:
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    for msg in data["messages"]:
        headers = msg["payload"]["headers"]
        from_value = _header(headers, "From")
        subject = _header(headers, "Subject") or None

        from_addrs = _extract_emails(from_value)
        to_addrs = _extract_emails(_header(headers, "To"))
        cc_addrs = _extract_emails(_header(headers, "Cc"))
        participants = sorted(set(from_addrs + to_addrs + cc_addrs))
        author = from_addrs[0] if from_addrs else None

        created = datetime.fromtimestamp(int(msg["internalDate"]) / 1000, tz=timezone.utc)
        body = _extract_body(msg["payload"])
        is_auto = bool(_header(headers, "Auto-Submitted") or _header(headers, "X-Autoreply"))

        yield Document(
            id=f"gmail:{msg['id']}",
            source="gmail",
            type="email",
            title=subject,
            body=body,
            author=author,
            participants=participants,
            created_at=created,
            updated_at=created,
            metadata={
                "thread_id": msg["threadId"],
                "label_ids": msg.get("labelIds", []),
                "message_id_header": _header(headers, "Message-ID"),
                "in_reply_to": _header(headers, "In-Reply-To") or None,
                "snippet": msg.get("snippet"),
                "from_display": from_value,
                "to": _header(headers, "To"),
                "cc": _header(headers, "Cc"),
                "is_auto_reply": is_auto,
            },
        )
