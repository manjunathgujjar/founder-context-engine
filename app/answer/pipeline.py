"""End-to-end answer pipeline.

   question
       ↓
   hybrid_search ──→ list[FusedHit]
       ↓
   prompts.build_messages
       ↓
   llm.call
       ↓
   prompts.parse_citations + verify against CONTEXT IDs
       ↓
   AnswerResult { answer, citations[], unverified_citations[], context[], usage, latency }

`citations[]` are inline IDs the model emitted that map to documents we actually
sent. `unverified_citations[]` are IDs the model emitted that we did NOT send
in CONTEXT — the anti-hallucination guardrail. UI should warn / strip these.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.answer import llm, prompts
from app.retrieve.hybrid import FusedHit, hybrid_search
from app.store.db import connect


@dataclass(slots=True)
class CitationContext:
    """Lightweight, UI-renderable record for one document that was sent as context."""
    document_id: str
    source: str
    type: str
    title: str | None
    snippet: str
    created_at: str
    updated_at: str
    fused_score: float
    fts_rank: int | None
    vector_rank: int | None


@dataclass(slots=True)
class AnswerResult:
    question: str
    answer: str
    citations: list[str]
    unverified_citations: list[str]
    context: list[CitationContext]
    today_iso: str
    model: str
    usage: dict[str, int]
    timings: dict[str, float] = field(default_factory=dict)


def answer(question: str, *, top_k: int | None = None) -> AnswerResult:
    """Single-call pipeline: retrieve, synthesize, parse + verify citations."""
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    with connect() as conn:
        hits = hybrid_search(conn, question, top_k=top_k)
    timings["retrieve_seconds"] = time.perf_counter() - t0

    today_iso = prompts.resolve_today_iso()
    messages = prompts.build_messages(question, hits, today_iso=today_iso)

    t1 = time.perf_counter()
    llm_resp = llm.call(messages)
    timings["llm_seconds"] = time.perf_counter() - t1
    timings["llm_round_trip_seconds"] = llm_resp.latency_seconds

    context_ids = {h.document_id for h in hits}
    parsed = prompts.parse_citations(llm_resp.text)
    verified, unverified = prompts.verify_citations(parsed, context_ids)

    return AnswerResult(
        question=question,
        answer=llm_resp.text,
        citations=verified,
        unverified_citations=unverified,
        context=[_to_context(h) for h in hits],
        today_iso=today_iso,
        model=llm_resp.model,
        usage=llm_resp.usage,
        timings=timings,
    )


def _to_context(h: FusedHit) -> CitationContext:
    return CitationContext(
        document_id=h.document_id,
        source=h.source,
        type=h.type,
        title=h.title,
        snippet=_snippet(h.chunk_text),
        created_at=h.document_created_at,
        updated_at=h.document_updated_at,
        fused_score=h.fused_score,
        fts_rank=h.fts_rank,
        vector_rank=h.vector_rank,
    )


_SNIPPET_CHARS = 320


def _snippet(text: str) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= _SNIPPET_CHARS:
        return text
    return text[: _SNIPPET_CHARS - 1].rsplit(" ", 1)[0] + "…"
