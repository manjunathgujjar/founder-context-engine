"""Slack fixture connector: slack.json → Document stream.

Each Slack message (root or reply) becomes one Document. User IDs are resolved
against the workspace user list; channel context is denormalized into title +
body so retrieval hits land with full context.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from app.models import Document

SOURCE = "slack"


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _channel_label(ch: dict, users_by_id: dict) -> str:
    if ch.get("is_im"):
        members = [users_by_id.get(uid, {}).get("real_name") or uid for uid in ch.get("members", [])]
        return f"DM ({', '.join(members)})"
    return f"#{ch.get('name', ch['id'])}"


def load(fixture_path: Path) -> Iterable[Document]:
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    users_by_id = {u["id"]: u for u in data.get("users", [])}
    channels_by_id = {c["id"]: c for c in data.get("channels", [])}

    for msg in data["messages"]:
        ch = channels_by_id.get(msg["channel"], {"id": msg["channel"]})
        user = users_by_id.get(
            msg["user"],
            {"real_name": msg["user"], "email": None, "name": msg["user"]},
        )

        ch_label = _channel_label(ch, users_by_id)
        author_email = user.get("email")
        author_name = user.get("real_name") or user.get("name") or msg["user"]
        created = _parse_iso(msg["created_at"])

        title = f"{author_name} in {ch_label}"

        body_lines = [f"Channel: {ch_label}", f"From: {author_name} <{author_email or ''}>"]
        if msg.get("thread_ts"):
            body_lines.append(f"Thread reply (parent ts={msg['thread_ts']})")
        else:
            body_lines.append("Channel post")
        body_lines.append("")
        body_lines.append(msg.get("text", ""))
        if msg.get("reactions"):
            reacts = ", ".join(f":{r['name']}: x{r.get('count', 0)}" for r in msg["reactions"])
            body_lines.append(f"\nReactions: {reacts}")
        body = "\n".join(body_lines)

        member_emails = sorted(
            {users_by_id[m]["email"] for m in ch.get("members", []) if users_by_id.get(m, {}).get("email")}
        )

        yield Document(
            id=f"slack:{ch['id']}/{msg['ts']}",
            source="slack",
            type="message",
            title=title,
            body=body,
            author=author_email,
            participants=member_emails,
            created_at=created,
            updated_at=created,
            metadata={
                "channel_id": ch["id"],
                "channel_name": ch.get("name"),
                "channel_label": ch_label,
                "is_im": ch.get("is_im", False),
                "is_private": ch.get("is_private", False),
                "ts": msg["ts"],
                "thread_ts": msg.get("thread_ts"),
                "reply_count": msg.get("reply_count", 0),
                "user_id": msg["user"],
                "user_name": author_name,
                "user_title": user.get("title"),
                "reactions": msg.get("reactions", []),
            },
        )
