"""End-to-end answer pipeline.

   question
       ↓
   sha256 → answer_cache lookup (1hr TTL) → HIT returns cached result
       ↓ miss
   hybrid_search ──→ list[FusedHit]
       ↓
   top_fused_score < CONFIDENCE_FLOOR  ──→ structured refusal (no LLM call)
       ↓ pass
   prompts.build_messages
       ↓
   llm.call  (off-loaded to thread executor; doesn't block the event loop)
       ↓
   prompts.parse_citations + verify against CONTEXT IDs
       ↓
   cache_put + log_request
       ↓
   AnswerResult { answer, citations[], unverified_citations[], context[],
                  refused, cached, today_iso, model, usage, timings }
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta

from app.answer import llm, prompts
from app.retrieve.hybrid import FusedHit, hybrid_search
from app.store.db import connect
from app.store.repository import cache_get, cache_put, log_request

CONFIDENCE_FLOOR = 0.005
CACHE_TTL = timedelta(hours=1)

REFUSAL_TEXT = (
    "I don't see anything in the stored context relevant to that question. "
    "The retrieved documents had no meaningful match against my stored Gmail, "
    "Calendar, Linear, or Slack data — no grounded citations are possible."
)


@dataclass(slots=True)
class CitationContext:
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
    usage: dict
    timings: dict = field(default_factory=dict)
    refused: bool = False
    cached: bool = False


def _question_hash(question: str) -> str:
    norm = " ".join(question.strip().lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


async def answer(question: str, *, top_k: int | None = None) -> AnswerResult:
    """End-to-end. Async so the LLM call doesn't block the event loop."""
    started = time.perf_counter()
    timings: dict[str, float] = {}
    q_hash = _question_hash(question)

    with connect() as conn:
        cached_json = cache_get(conn, q_hash, ttl=CACHE_TTL)
    if cached_json is not None:
        result = _deserialize(cached_json)
        result.cached = True
        result.timings = {**(result.timings or {}), "total_seconds": round(time.perf_counter() - started, 4)}
        with connect() as conn:
            log_request(
                conn, question,
                cache_hit=True, refused=result.refused,
                top_fused_score=None, cost=0.0,
                prompt_tokens=0, completion_tokens=0,
                retrieve_seconds=0.0, llm_seconds=0.0,
                total_seconds=result.timings["total_seconds"],
            )
        return result

    t0 = time.perf_counter()
    with connect() as conn:
        hits = hybrid_search(conn, question, top_k=top_k)
    timings["retrieve_seconds"] = round(time.perf_counter() - t0, 4)

    top_score = hits[0].fused_score if hits else 0.0
    today_iso = prompts.resolve_today_iso()

    if top_score < CONFIDENCE_FLOOR:
        timings["llm_seconds"] = 0.0
        timings["total_seconds"] = round(time.perf_counter() - started, 4)
        result = AnswerResult(
            question=question,
            answer=REFUSAL_TEXT,
            citations=[],
            unverified_citations=[],
            context=[_to_context(h) for h in hits],
            today_iso=today_iso,
            model="(refused: below confidence floor)",
            usage={},
            timings=timings,
            refused=True,
        )
        with connect() as conn:
            cache_put(conn, q_hash, question, _serialize(result))
            log_request(
                conn, question,
                cache_hit=False, refused=True,
                top_fused_score=top_score, cost=0.0,
                prompt_tokens=0, completion_tokens=0,
                retrieve_seconds=timings["retrieve_seconds"],
                llm_seconds=0.0,
                total_seconds=timings["total_seconds"],
            )
        return result

    messages = prompts.build_messages(question, hits, today_iso=today_iso)
    t1 = time.perf_counter()
    loop = asyncio.get_running_loop()
    llm_resp = await loop.run_in_executor(None, llm.call, messages)
    timings["llm_seconds"] = round(time.perf_counter() - t1, 4)

    context_ids = {h.document_id for h in hits}
    parsed = prompts.parse_citations(llm_resp.text)
    verified, unverified = prompts.verify_citations(parsed, context_ids)
    timings["total_seconds"] = round(time.perf_counter() - started, 4)

    result = AnswerResult(
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

    with connect() as conn:
        cache_put(conn, q_hash, question, _serialize(result))
        log_request(
            conn, question,
            cache_hit=False, refused=False,
            top_fused_score=top_score,
            cost=float(llm_resp.usage.get("cost", 0.0) or 0.0),
            prompt_tokens=int(llm_resp.usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(llm_resp.usage.get("completion_tokens", 0) or 0),
            retrieve_seconds=timings["retrieve_seconds"],
            llm_seconds=timings["llm_seconds"],
            total_seconds=timings["total_seconds"],
        )
    return result


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


def _serialize(r: AnswerResult) -> str:
    return json.dumps(asdict(r), default=str)


def _deserialize(s: str) -> AnswerResult:
    d = json.loads(s)
    d["context"] = [CitationContext(**c) for c in d.get("context", [])]
    return AnswerResult(**d)
