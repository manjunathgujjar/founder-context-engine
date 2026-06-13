"""Grounding tests: citation parser correctness + pipeline refusal & verification.

The async tests exercise the answer pipeline end-to-end without spending OpenRouter
tokens. We patch llm.call with side_effect=AssertionError on the confidence-floor
test (it MUST NOT be called) and with a fake return on the unverified-citation
test (so we can assert the parser routes a made-up ID to unverified_citations).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.answer import llm, pipeline, prompts


# --- parse_citations ---------------------------------------------------------


def test_parse_citations_finds_all_four_sources():
    text = (
        "Rachel asked for cohort data [gmail:18bf1b3d4e5f6071] and slide 14. "
        "ENG-142 is blocked [linear:ENG-142]. See Slack [slack:C07ENG/1781184120.000100] "
        "and the Wednesday investor meeting [gcal:5kfp8q1k0aoi52rpmpvkvulqu9]."
    )
    cids = prompts.parse_citations(text)
    assert "gmail:18bf1b3d4e5f6071" in cids
    assert "linear:ENG-142" in cids
    assert "slack:C07ENG/1781184120.000100" in cids
    assert "gcal:5kfp8q1k0aoi52rpmpvkvulqu9" in cids
    assert len(cids) == 4


def test_parse_citations_dedupes_preserving_order():
    text = "[linear:ENG-142] then [gmail:abc123] then [linear:ENG-142] again."
    cids = prompts.parse_citations(text)
    assert cids == ["linear:ENG-142", "gmail:abc123"]


def test_parse_citations_ignores_non_prefix_brackets():
    text = "Maya said [hi] and ENG-142 [Maya] [foo:bar]; only [linear:ENG-142] counts."
    cids = prompts.parse_citations(text)
    assert cids == ["linear:ENG-142"]


def test_verify_citations_splits_known_vs_unknown():
    parsed = ["linear:ENG-142", "linear:ENG-FAKE999", "gmail:abc"]
    context_ids = {"linear:ENG-142", "gmail:abc"}
    verified, unverified = prompts.verify_citations(parsed, context_ids)
    assert verified == ["linear:ENG-142", "gmail:abc"]
    assert unverified == ["linear:ENG-FAKE999"]


# --- pipeline.answer (async) -------------------------------------------------


async def test_pipeline_refuses_when_retrieval_empty_without_llm_call(require_ingested_db):
    """When retrieval returns no hits, the confidence floor trips and the LLM is not called.

    We mock hybrid_search rather than relying on a "nonsense" natural-language query,
    because the dense vector lane always returns *some* top-K above the floor —
    cosine similarity isn't sparse. The floor's purpose is the empty-retrieval safety
    net; this test exercises that contract directly.
    """
    with (
        patch.object(llm, "call", side_effect=AssertionError("LLM should not be called")),
        patch("app.answer.pipeline.hybrid_search", return_value=[]),
    ):
        result = await pipeline.answer("xyzzy plover frobnicate quux", top_k=8)
    assert result.refused is True
    assert result.citations == []
    assert "stored context" in result.answer.lower()


async def test_pipeline_marks_fake_id_as_unverified(require_ingested_db):
    """A model that emits a fake citation should land it in unverified_citations."""
    fake_response = llm.LLMResponse(
        text="ENG-142 is blocked [linear:ENG-142]. Also see [linear:ENG-FAKE999].",
        model="test-fake-model",
        usage={"prompt_tokens": 0, "completion_tokens": 0},
        latency_seconds=0.0,
    )
    with patch.object(llm, "call", return_value=fake_response):
        result = await pipeline.answer("which tasks are blocked", top_k=8)

    if result.refused:
        pytest.skip("refusal floor tripped — query did not retrieve enough context")

    assert "linear:ENG-FAKE999" in result.unverified_citations
    assert "linear:ENG-FAKE999" not in result.citations
