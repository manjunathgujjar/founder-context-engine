"""Google Calendar fixture connector: gcal.json → Document stream."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from app.models import Document

SOURCE = "gcal"


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _attendee_roster(attendees: list[dict]) -> str:
    lines = []
    for a in attendees:
        name = a.get("displayName") or a["email"]
        status = a.get("responseStatus", "needsAction")
        tags = []
        if a.get("organizer"): tags.append("organizer")
        if a.get("optional"):  tags.append("optional")
        if a.get("self"):      tags.append("self")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        lines.append(f"  - {name} <{a['email']}> — {status}{tag_str}")
    return "\n".join(lines)


def load(fixture_path: Path) -> Iterable[Document]:
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    for ev in data["events"]:
        summary = ev.get("summary", "") or ""
        description = ev.get("description", "") or ""
        location = ev.get("location") or ""
        attendees = ev.get("attendees", [])
        organizer = ev.get("organizer", {}) or {}

        body_parts = []
        if description: body_parts.append(description)
        if location:    body_parts.append(f"Location: {location}")
        if attendees:   body_parts.append("Attendees:\n" + _attendee_roster(attendees))
        body = "\n\n".join(body_parts)

        start_dt = _parse_iso(ev["start"]["dateTime"])
        end_dt = _parse_iso(ev["end"]["dateTime"])
        updated_dt = _parse_iso(ev.get("updated")) or _parse_iso(ev.get("created")) or start_dt

        conference_uri = None
        entry_points = (ev.get("conferenceData") or {}).get("entryPoints") or []
        if entry_points:
            conference_uri = entry_points[0].get("uri")

        yield Document(
            id=f"gcal:{ev['id']}",
            source="gcal",
            type="event",
            title=summary,
            body=body,
            author=organizer.get("email"),
            participants=sorted({a["email"] for a in attendees if a.get("email")}),
            created_at=start_dt,
            updated_at=updated_dt,
            metadata={
                "status": ev.get("status"),
                "start": ev["start"]["dateTime"],
                "end": ev["end"]["dateTime"],
                "time_zone": ev["start"].get("timeZone"),
                "location": location or None,
                "html_link": ev.get("htmlLink"),
                "ical_uid": ev.get("iCalUID"),
                "recurring_event_id": ev.get("recurringEventId"),
                "organizer_email": organizer.get("email"),
                "organizer_name": organizer.get("displayName"),
                "conference_uri": conference_uri,
                "attendees": [
                    {
                        "email": a["email"],
                        "displayName": a.get("displayName"),
                        "responseStatus": a.get("responseStatus"),
                        "optional": a.get("optional", False),
                        "self": a.get("self", False),
                    }
                    for a in attendees
                ],
                "duration_minutes": int((end_dt - start_dt).total_seconds() // 60) if start_dt and end_dt else None,
            },
        )
