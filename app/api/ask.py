"""Q&A endpoint.

POST /api/ask
    { "question": "...", "top_k": 8 }
returns
    { answer, citations[], unverified_citations[], context[], today_iso,
      model, usage, timings }
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.answer import pipeline
from app.answer.llm import LLMError

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class CitationContextOut(BaseModel):
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


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]
    unverified_citations: list[str]
    context: list[CitationContextOut]
    today_iso: str
    model: str
    usage: dict
    timings: dict


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        result = pipeline.answer(req.question, top_k=req.top_k)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AskResponse(
        question=result.question,
        answer=result.answer,
        citations=result.citations,
        unverified_citations=result.unverified_citations,
        context=[CitationContextOut(**asdict(c)) for c in result.context],
        today_iso=result.today_iso,
        model=result.model,
        usage=result.usage,
        timings=result.timings,
    )
