"""Golden-set eval harness for the Pierside Founder AI Assistant.

Loads evals/golden.json, runs each question through the live answer pipeline
(real retrieval + LLM call — uses OPENROUTER_API_KEY from .env), then scores
each item against four optional checks. Exits 1 on any failure so it can be
wired into CI.

    python evals/run.py [path/to/golden.json]
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

# Make `app` importable when run as `python evals/run.py` from the repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.answer import pipeline  # noqa: E402


# Phrases the grounding-rules system prompt instructs the model to emit when
# CONTEXT doesn't actually answer the question. Either a pre-LLM floor refusal
# (result.refused=True) or an LLM-emitted refusal containing one of these
# phrases counts — both are valid "I won't hallucinate" outcomes from the UX.
_REFUSAL_PHRASES = (
    "don't see that in the stored context",
    "don't see anything in the stored context",
    "no relevant context",
    "i don't see",
)


def _looks_like_refusal(result) -> bool:
    if result.refused:
        return True
    answer = (result.answer or "").lower()
    return any(p in answer for p in _REFUSAL_PHRASES)


def _check_refusal(item: dict, result) -> str | None:
    expected = item.get("refusal_expected")
    if expected is None:
        return None
    actual = _looks_like_refusal(result)
    if actual != bool(expected):
        return (
            f"refusal mismatch: refused_flag={result.refused}, "
            f"looks_like_refusal={actual}, expected={expected}. "
            f"answer={result.answer[:200]!r}"
        )
    return None


def _check_must_cite(item: dict, result) -> str | None:
    required = item.get("must_cite") or []
    cited = set(result.citations or [])
    missing = [c for c in required if c not in cited]
    if missing:
        return f"missing required citation(s): {missing}. got={sorted(cited)}"
    return None


def _check_must_cite_any_of(item: dict, result) -> str | None:
    options = item.get("must_cite_any_of") or []
    if not options:
        return None
    cited = set(result.citations or [])
    if not any(c in cited for c in options):
        return f"none of {options} cited. got={sorted(cited)}"
    return None


def _check_must_mention(item: dict, result) -> str | None:
    needles = item.get("must_mention") or []
    haystack = (result.answer or "").lower()
    missing = [n for n in needles if n.lower() not in haystack]
    if missing:
        return f"missing required mention(s): {missing}"
    return None


CHECKS = [_check_refusal, _check_must_cite, _check_must_cite_any_of, _check_must_mention]


async def score_one(item: dict) -> tuple[bool, list[str], float]:
    q = item["question"]
    t0 = time.perf_counter()
    result = await pipeline.answer(q)
    elapsed = time.perf_counter() - t0
    failures: list[str] = []
    for check in CHECKS:
        msg = check(item, result)
        if msg:
            failures.append(msg)
    return (not failures, failures, elapsed)


async def main(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["items"]
    print(f"Running {len(items)} eval items from {path}\n")

    n_pass = 0
    for item in items:
        ok, failures, elapsed = await score_one(item)
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {item['id']:<28} ({elapsed:5.2f}s)  {item['question']}")
        for f in failures:
            print(f"         - {f}")
        if ok:
            n_pass += 1

    n_fail = len(items) - n_pass
    print(f"\nSummary: {n_pass}/{len(items)} passed, {n_fail} failed.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "evals/golden.json"
    path = Path(arg)
    if not path.is_absolute():
        path = ROOT / path
    sys.exit(asyncio.run(main(path)))
