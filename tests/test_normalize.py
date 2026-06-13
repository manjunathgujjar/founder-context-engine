"""Connector smoke tests: each source's load() yields well-formed Documents.

We don't reassert exhaustive field-by-field shape — Document is a dataclass and
ingest is type-checked. We do assert the invariants that downstream retrieval
+ grounding rely on: source prefix on `id`, non-empty body, datetime types on
the time fields, and a couple of fixture-specific facts (Naomi's auto-reply,
ENG-142's inlined comments) that prove the connector unwrapped the source
JSON the way the rest of the pipeline expects.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from app.config import settings
from app.ingest import gcal, gmail, linear, slack
from app.models import Document

CONNECTORS = [
    ("gmail", gmail, "gmail.json"),
    ("gcal", gcal, "gcal.json"),
    ("linear", linear, "linear.json"),
    ("slack", slack, "slack.json"),
]


def _load(connector, fname: str) -> list[Document]:
    path = settings.fixtures_dir / fname
    assert path.exists(), f"Fixture missing: {path}"
    return list(connector.load(path))


@pytest.mark.parametrize(("source", "connector", "fname"), CONNECTORS)
def test_connector_yields_well_formed_documents(source, connector, fname):
    docs = _load(connector, fname)
    assert docs, f"{source}: no documents loaded"
    for d in docs:
        assert d.id.startswith(f"{source}:"), f"{source}: id missing prefix → {d.id}"
        assert d.source == source
        assert d.body and d.body.strip(), f"{source}: empty body in {d.id}"
        assert isinstance(d.created_at, datetime), f"{source}: bad created_at in {d.id}"
        assert isinstance(d.updated_at, datetime), f"{source}: bad updated_at in {d.id}"
        assert isinstance(d.participants, list)
        assert isinstance(d.metadata, dict)


def test_gmail_marks_auto_reply():
    """Naomi Patel's OOO bounce must round-trip as is_auto_reply=True."""
    docs = _load(gmail, "gmail.json")
    auto = [d for d in docs if d.metadata.get("is_auto_reply")]
    assert auto, "Expected at least one Gmail message flagged is_auto_reply"
    assert any("Naomi" in (d.metadata.get("from_display") or "") for d in auto), \
        "Expected the Stripe OOO bounce (from Naomi Patel) among auto-replies"


def test_linear_eng142_inlines_blocked_comments():
    """ENG-142's body must include the inlined comments (Blocked / Alex Volkov / Stripe)."""
    docs = _load(linear, "linear.json")
    eng142 = next((d for d in docs if d.id == "linear:ENG-142"), None)
    assert eng142 is not None, "Expected linear:ENG-142 in fixture"
    body = eng142.body
    assert "Blocked" in body, "Expected ENG-142 body to contain 'Blocked'"
    assert "Alex Volkov" in body, "Expected ENG-142 body to inline Alex Volkov's comment"
    assert "Stripe" in body, "Expected ENG-142 body to mention Stripe (the blocker)"
