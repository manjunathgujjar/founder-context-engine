"""System prompt, context-block template, and citation parser.

The system prompt and context format are the contract that the rest of the
answer pipeline orchestrates around. Any tweak here should be reviewed against
the "Phase E Lock-in" section of the plan document, then this file.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from app.config import settings
from app.retrieve.hybrid import FusedHit


SYSTEM_PROMPT_TEMPLATE = """You are the founder assistant for Maya Chen, CEO of Pierside — a B2B
usage-based billing reconciliation SaaS. Your job is to help Maya
understand what is happening across her work life by answering questions
grounded in her own Gmail, Google Calendar, Linear, and Slack data.

GROUNDING RULES
1. Answer only from the CONTEXT block in the user message. Do not use
   outside knowledge to fill gaps about Pierside, its customers, investors,
   employees, decisions, deadlines, dollar amounts, or commitments.
2. Never invent. No fictional names, dates, amounts, ticket IDs, attendee
   lists, meeting times, or outcomes. If a detail is not explicitly in
   CONTEXT, do not state it.
3. Refuse honestly when CONTEXT is thin. Say "I don't see that in the
   stored context — the closest thing I found is X." Do not pad with
   plausible-sounding generalities.
4. No extrapolation. "Annoyed" is not "about to churn"; "in progress" is
   not "almost done".
5. Reason about time relative to TODAY (below). Treat "this week" as the
   calendar week containing TODAY, "next Wednesday" as the Wednesday
   after TODAY, etc. Respect timezones on CONTEXT events.

CITATION RULES
- Every factual claim must be cited inline with one or more document IDs
  in square brackets, taken verbatim from the `id=` field of CONTEXT items.
- Format: [linear:ENG-142], [gmail:18bf1b3d4e5f6071],
  [slack:C07FOUNDRS/1781086500.000100], [gcal:5kfp8q1k0aoi52rpmpvkvulqu9]
- Multiple sources for one claim: [linear:ENG-142][slack:C07ENG/1780910040.000100]
- Do not invent IDs. Do not add a separate "Sources:" section.

STYLE
- Concise, direct. Bullets for lists. No preambles like "Based on the
  provided context...".
- Speak to Maya in second person.
- Surface overdue/blocked/recurring without being prompted when clearly
  relevant.
- Cross-link sources when useful.

TODAY is {today_iso}."""


_DOCUMENT_BLOCK = """--- DOCUMENT id={document_id} source={source} type={type} ---
Title: {title}
Created: {created_at}    Updated: {updated_at}

{chunk_text}"""


_USER_TEMPLATE = """CONTEXT
=======

{context_blocks}

QUESTION
========
{question}

Answer Maya's question above, following the grounding and citation rules
in your instructions. If CONTEXT is empty or unrelated, say so plainly."""


# Anchored to known source prefixes. Character class covers every observed
# ID shape: gmail hex, linear ENG-NNN, slack channel/ts, gcal lowercase alnum.
CITATION_REGEX = re.compile(r"\[((?:gmail|gcal|linear|slack):[A-Za-z0-9._/+\-]+)\]")


def resolve_today_iso() -> str:
    """ISO date for prompt substitution. Honors settings.pinned_today for demos."""
    if settings.pinned_today:
        return settings.pinned_today
    return datetime.now(timezone.utc).date().isoformat()


def render_context_block(hit: FusedHit) -> str:
    return _DOCUMENT_BLOCK.format(
        document_id=hit.document_id,
        source=hit.source,
        type=hit.type,
        title=hit.title or "(untitled)",
        created_at=hit.document_created_at,
        updated_at=hit.document_updated_at,
        chunk_text=hit.chunk_text.strip(),
    )


def build_messages(question: str, hits: list[FusedHit], today_iso: str | None = None) -> list[dict]:
    """Build the OpenAI-style messages array for the synthesis call."""
    today = today_iso or resolve_today_iso()
    system = SYSTEM_PROMPT_TEMPLATE.format(today_iso=today)

    if hits:
        context_blocks = "\n\n".join(render_context_block(h) for h in hits)
    else:
        context_blocks = "(no documents retrieved)"

    user = _USER_TEMPLATE.format(context_blocks=context_blocks, question=question)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_citations(text: str) -> list[str]:
    """Extract unique citation IDs from an answer, preserving first-appearance order."""
    seen: dict[str, None] = {}
    for match in CITATION_REGEX.findall(text):
        seen.setdefault(match, None)
    return list(seen.keys())


def verify_citations(citations: list[str], context_ids: set[str]) -> tuple[list[str], list[str]]:
    """Split parsed citations into (verified, unverified) against the CONTEXT-supplied IDs."""
    verified, unverified = [], []
    for cid in citations:
        (verified if cid in context_ids else unverified).append(cid)
    return verified, unverified
