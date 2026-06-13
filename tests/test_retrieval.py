"""Hybrid retrieval smoke tests.

We assert that known queries surface the documents we expect in the top K,
not exact ordering — RRF is robust but small embedding shifts can swap rank 1
and 2. For off-topic queries we assert the on-topic anchor is absent from the
top K — RRF scores are quasi-uniform when both lanes hit, so raw score
comparison isn't a meaningful signal; presence/absence of the known doc is.
"""

from __future__ import annotations

from app.retrieve.hybrid import hybrid_search
from app.store.db import connect


def _doc_ids(hits) -> list[str]:
    return [h.document_id for h in hits]


def test_blocked_tasks_surfaces_eng142(require_ingested_db):
    with connect() as conn:
        hits = hybrid_search(conn, "which tasks are blocked and why", top_k=8)
    ids = _doc_ids(hits)
    assert "linear:ENG-142" in ids, f"ENG-142 missing from top-8: {ids}"


def test_rachel_query_surfaces_investor_thread(require_ingested_db):
    with connect() as conn:
        hits = hybrid_search(conn, "what did Rachel ask me for", top_k=8)
    ids = _doc_ids(hits)
    assert "gmail:18bf1b3d4e5f6071" in ids, f"Rachel's reply missing from top-8: {ids}"


def test_duplicate_events_cross_source(require_ingested_db):
    with connect() as conn:
        hits = hybrid_search(conn, "duplicate events keep coming up across customers", top_k=8)
    sources = {h.source for h in hits}
    assert len(sources) >= 2, f"expected cross-source recall, got sources={sources}"


def test_off_topic_does_not_surface_eng142(require_ingested_db):
    """Wifi/office query has no overlap with the blocked-Stripe-webhook ticket;
    ENG-142 must not appear in the off-topic top-8."""
    with connect() as conn:
        off = hybrid_search(conn, "what is the wifi password at the office", top_k=8)
    ids = _doc_ids(off)
    assert "linear:ENG-142" not in ids, f"off-topic query unexpectedly found ENG-142: {ids}"
