# Pierside Founder AI Assistant

A small but realistic **context engine** for a startup founder. Ingests Gmail, Google Calendar, Linear, and Slack into a single SQLite store, retrieves relevant context with hybrid search, and answers questions with grounded, inline-cited synthesis. It is explicitly **not a chatbot that calls APIs live** — at query time it only reads from the local store, which is the architecture the take-home asks for.

Demo persona: **Maya Chen, CEO of Pierside**, a B2B usage-based billing reconciliation SaaS. The four fixture files in `data/fixtures/` tell one coherent week-in-the-life story (investor follow-up, blocked Stripe webhook ticket, recurring duplicate-event customer reports, v1 API deprecation decision) so cross-source retrieval has real signal to find.

---

## Quickstart

### Prerequisites

- **Python 3.12+** ([python.org/downloads](https://www.python.org/downloads/)) — required for the local-Python option. The Docker option ships its own Python.
- **uv** package manager — local-Python option only.
  ```bash
  pip install uv
  # or, no-pip install:
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Docker + Docker Compose** ([docker.com/get-started](https://www.docker.com/get-started/)) — Option A only.
- **OpenRouter API key** ([openrouter.ai/keys](https://openrouter.ai/keys)) — free tier works for all six example questions. One `/api/ask` round-trip costs ~$0.005.

Clone the repo:

```bash
git clone https://github.com/manjunathgujjar/founder-context-engine.git
cd founder-context-engine
```

### Option A: Docker (one command)

```bash
cp .env.example .env
# put your OpenRouter key in .env
docker compose up --build
# → http://localhost:8000
```

The image pre-bakes the MiniLM weights at build time, so the first request is not a 90 MB cold-start. The SQLite DB is persisted on a named volume.

### Option B: Local Python

```bash
cp .env.example .env
# put your OpenRouter key in .env

uv sync                          # installs runtime deps
uv sync --extra dev              # adds pytest + ruff (optional)

python scripts/ingest.py         # populates data/app.db
uv run uvicorn app.main:app --reload    # → http://localhost:8000
```

The ingest step downloads the MiniLM model (~80 MB) on first run and caches it. If port 8000 is occupied, use `uv run uvicorn app.main:app --reload --port 8123`.

### `.env`

```
OPENROUTER_API_KEY=sk-or-...           # required
OPENROUTER_MODEL=anthropic/claude-haiku-4.5
PINNED_TODAY=2026-06-12                # pins "today" so the demo is deterministic
```

`PINNED_TODAY` is the most important setting for demo: the fixtures are dated around 2026-06-08 to 2026-06-19, so pinning today to **2026-06-12** is what makes phrases like "this week" and "next Wednesday" resolve correctly. Unset it for live-clock behavior.

---

## Connected data sources

| Source | Connector | Fixture | Notes |
|---|---|---|---|
| Gmail    | [app/ingest/gmail.py](app/ingest/gmail.py) | `data/fixtures/gmail.json` | Real Gmail API shape (`payload.headers`, `payload.body.data`). 10 messages. |
| Calendar | [app/ingest/gcal.py](app/ingest/gcal.py)  | `data/fixtures/gcal.json`  | Real Google Calendar shape. 10 events. |
| Linear   | [app/ingest/linear.py](app/ingest/linear.py) | `data/fixtures/linear.json` | 15 issues. Comments inlined into the body so a single retrieval hit returns full thread context. |
| Slack    | [app/ingest/slack.py](app/ingest/slack.py)   | `data/fixtures/slack.json`  | 31 messages across `#founders`, `#eng`, `#customer-success`, and three DMs. |

Fixtures are fictional but written in the actual provider response shapes, so swapping in real OAuth is a connector-by-connector change with no schema impact.

---

## Six example questions

All six pass through the same pipeline: hybrid retrieval → strict-grounding system prompt → LLM synthesis → citation parser → verified vs unverified split. Examples 1–5 are short expected-behavior summaries; example 6 and the bonus transcript below carry live capture from the running server. With `PINNED_TODAY=2026-06-12`:

1. **"What did Rachel ask me for?"**
   → Cites `gmail:18bf1b3d4e5f6071` (Rachel Goldman, Northbound Capital). Expected mentions: cohort breakdown, slide 14 re-cut, two customer references, end-of-week deadline.

2. **"Which tasks are blocked and why?"**
   → Cites `linear:ENG-142` (Stripe webhook receiver). Expected mentions: Stripe support ticket cs-9924, blocks Helix/Brightline rollouts, Naomi Patel OOO until 06-15.

3. **"What customer issues keep coming up?"**
   → Cites some subset of the duplicate-event-reports thread: `slack:C07ENG/1781184120.000100` (Priya's "3rd customer this week" post), the Helix/Brightline/Vesta emails (`gmail:18bf2c4e5f607182`, `gmail:18bf0e6071829304`, `gmail:18bf3082930405c6`), or `linear:ENG-135` (schema validation hardening). Expected mention: "duplicate".

4. **"What decisions were made this week?"**
   → Cites `linear:ENG-152` (deprecate v1 metering API, sunset 2026-09-30). Expected mentions: v1, deprecation, customer-comms due 2026-06-19.

5. **"What is the wifi password at the office?"**
   → Refusal. No fixture contains this. The assistant must emit a refusal phrasing (e.g. "I don't see that in the stored context") rather than hallucinate. If retrieval returns zero hits the pre-LLM confidence floor short-circuits and `refused: true` is set without a token call; otherwise the system prompt instructs the model to refuse.

6. **"What follow-ups am I missing?"**
   → Cross-source — investor commitments + a customer-facing root-cause note both surface as overdue.
   Citations: `gmail:18bf1b3d4e5f6071`, `linear:ENG-150`, `slack:C07CS/1781172600.000101`, `gmail:18bf2c4e5f607182`, `gmail:18bf2d5f607182a3`. No unverified.

   Live transcript (truncated, `PINNED_TODAY=2026-06-12`):
   > You have three deliverables due **today (Friday 06-12)** for the Northbound partner meeting on 06-17 [gmail:18bf1b3d4e5f6071][linear:ENG-150]:
   >
   > 1. **90-day cohort ARR breakdown by tier** — pulling from the reconciliation warehouse, share by EOD Friday [linear:ENG-150].
   > 2. **Competitive-moat rewrite of slide 14** — Rachel asked for sharper articulation vs. Metronome/Orb [gmail:18bf1b3d4e5f6071].
   > 3. **Warm intros to 2 customer references** — Verdant Foods + Northwind Pay confirmed [slack:C07CS/1781172600.000101].
   >
   > **Separate issue:** Helix flagged ~40 duplicate event_ids; Priya committed to a written root-cause to Marcus by Friday [gmail:18bf2d5f607182a3] — also due today.

**Bonus transcript: "What should I focus on today?"** — the daily-priorities flagship question the rubric calls out. Live capture (truncated, `PINNED_TODAY=2026-06-12`):

> **Immediate (blocking others):**
> - **Series A materials due Friday** [gmail:18bf1b3d4e5f6071]. Rachel needs the 90-day cohort accuracy %, the revised slide 14, and warm intros to Verdant Foods + Northwind Pay. Partners meeting next Wed (06-17).
> - **Helix Health v2 migration date** [linear:ENG-148]. Marcus needs a firm date by Mon (06-16) for their 06-30 reporting close. ENG-142 is currently blocked on Stripe SE OOO [slack:D07DVNMAYA/1781256600.000100], which cascades.
>
> **Secondary (in motion):** v1 deprecation comms launched today [slack:C07CS/1781253600.000100]; Vesta pulse call Friday [gcal:5kfp8q1k0aoi52rpmpvkvulqu8].

Citations: 6 verified, 0 unverified. (This response was a cache hit on the live run, 17 ms — see the caching note below.)

A bonus probe: **`/api/search?q=ENG-142`** hits the debug retrieval endpoint with a literal ticket ID. After the FTS escaping fix, the BM25 lane finds the ticket directly instead of relying on the vector lane.

---

## Architecture

```
scripts/ingest.py
        │
        ▼
4 connectors → normalized Document → SQLite (documents, chunks, documents_fts, embedding BLOB)
                                                         │
                                                         ▼
                                          hybrid retrieval = RRF(BM25, cosine)
                                                         │
                                                         ▼
                                  strict-grounding prompt → LLM → citation parser → verify
                                                         │
                                                         ▼
                                              FastAPI /api/ask
                                                         │
                                                         ▼
                                       app/web/index.html (3-column UI)
```

- **Normalized `Document`** ([app/models.py](app/models.py)): shared across all 4 sources. IDs encode source + entity (`linear:ENG-142`, `gmail:18bf...`, `slack:C07ENG/1781...`, `gcal:5kfp...`) so citations are self-describing.
- **Storage** ([app/store/schema.sql](app/store/schema.sql)): documents + chunks + FTS5 virtual table (with triggers) + L2-normalized embedding BLOBs. Plus `answer_cache` (1-hour TTL, sha256 of normalized question) and `request_log` (one row per `/api/ask`, drives `/api/stats`).
- **Hybrid retrieval** ([app/retrieve/hybrid.py](app/retrieve/hybrid.py)): BM25 top-20 + cosine top-20 → Reciprocal Rank Fusion (k=60) → dedupe by document → top-8.
- **Synthesis** ([app/answer/prompts.py](app/answer/prompts.py), [app/answer/pipeline.py](app/answer/pipeline.py)): strict-grounding system prompt with numbered rules, inline `[source:id]` citation format, refusal posture. The LLM call runs in a thread executor so it doesn't block the FastAPI event loop.
- **API** ([app/api/](app/api/)): `POST /api/ask`, `GET /api/search`, `GET /api/stats`, `GET /api/health`. The `/api/stats` endpoint exposes hit rate, refusals, average timings, total cost.
- **UI** ([app/web/index.html](app/web/index.html)): single-file vanilla HTML, three-column "command center" layout (sidebar quick-asks · main answer · right-rail citation cards). Tailwind play CDN, no build step.

---

## Decisions and tradeoffs

**Pre-ingested fictional fixtures, not live OAuth.** Live OAuth eats 6–10 hours that the rubric doesn't grade and that reviewers wouldn't exercise. Fictional fixtures let the reviewer run one command and have a populated assistant on a story tuned to the example questions. Cost: the corpus has no surprises — if you ask something the fixture doesn't cover, the assistant refuses cleanly, which is the correct behavior but is also a thin demo surface for unscripted queries.

**SQLite + FTS5 + BLOB vectors in one file.** Single-binary deploy, no extra service to run. Scales to ~10k docs comfortably; would need pgvector or Qdrant beyond that. At the current 66 chunks the vector lane is a brute-force matmul in NumPy — microseconds, but unconditionally O(N). Documented as future work, not optimized.

**Reciprocal Rank Fusion instead of weighted-sum hybrid.** BM25 scores and cosine similarities live on incomparable scales. RRF only uses rank order, so it's robust without per-source weight tuning. Tradeoff: RRF scores end up quasi-uniform when both lanes return top-K, so they're a poor signal for confidence thresholding — see the confidence-floor note below.

**No query router.** Every question goes through the same retrieval+synthesis path. A more ambitious system would classify by intent ("which meeting", "summarize this week", "draft a reply") and route to specialized prompts or tools. Out of scope here; documented as future work.

**Strict-grounding system prompt + verified-citation routing.** The model is instructed to emit `[source:id]` citations inline, taken verbatim from the CONTEXT block. A regex parser pulls them out, cross-checks against the IDs the pipeline actually sent, and splits into `citations` (verified) vs `unverified_citations` (model invented this ID — should never happen, but observable when it does). Citation verification catches fabricated IDs but does not detect a wrong claim attributed to a real document — it is a complementary check, not a full faithfulness guarantee.

**Confidence floor is an empty-retrieval safety net, not a semantic filter.** Set at 0.005 on the top RRF score. In practice the dense vector lane always returns *some* top hit at RRF ≈ 0.0164, so the floor only fires when retrieval returns literally zero rows. For "I don't know" responses on partial-match queries we rely on prompt obedience, not a hard score gate. A more honest gate would require a cross-encoder reranker score; we didn't build one.

**FTS escape preserves hyphens** ([app/store/repository.py:293](app/store/repository.py#L293)). Earlier version stripped `-` from query tokens, flattening "ENG-142" to "ENG142" (one token) which never matched the index's tokenized `[ENG, 142]`. Fixed in this branch — the regression test is `tests/test_retrieval.py::test_literal_ticket_id_surfaces_via_fts`.

**Eval harness is keyword-spot, not LLM-as-judge.** [`evals/run.py`](evals/run.py) checks `must_mention` (case-insensitive substring), `must_cite` (exact ID present), and refusal (either pre-LLM floor or LLM-emitted phrasing). This is enough to catch regressions on the known fixture but does **not** judge groundedness in general — an answer that includes the right substrings is treated as passing even if the surrounding prose is wrong. A real eval would use a cross-encoder for retrieval relevance and an LLM judge for answer fidelity. Mentioned because the rubric grades anti-hallucination explicitly, and substring-match doesn't really test it.

**Caching trades freshness for cost.** `POST /api/ask` hashes the normalized question and caches the full `AnswerResult` for 1 hour. Repeated demo questions are free and fast; the cost is that an answer can be stale if the underlying store changes mid-hour. Acceptable for a single-user demo, would need invalidation hooks in a multi-user setting.

---

## Tests and evals

```bash
uv run pytest -v        # 17 tests (normalize / retrieval / grounding)
uv run python evals/run.py   # 5 golden Q/A pairs against the live pipeline
```

The eval harness exits 1 on any failure, so it's CI-shaped even though no CI is wired up. Re-running the same questions inside an hour hits the SQLite answer cache and returns in milliseconds — useful for a demo but worth knowing when interpreting eval timings.

---

## Future improvements

- Real OAuth + scheduled incremental ingest (the `Connector` protocol already abstracts this).
- Cross-encoder reranker on the top-N hits before synthesis. Likely the highest-leverage retrieval improvement we didn't make.
- Streaming answers via SSE.
- Per-user namespaces in the store.
- LLM-as-judge eval for groundedness fidelity, replacing the keyword-spot harness.
- Intent router → specialized prompts (meeting prep vs week-in-review vs action drafting).
- Action layer: "draft a reply to Rachel", "create a Linear ticket from this Slack thread". Write-side, deliberately out of scope.

---

## Traces

AI-assisted development workflow: **https://traces.com/s/jn74a7az1hxg3xc4jmad66h7tx88m4vh**
