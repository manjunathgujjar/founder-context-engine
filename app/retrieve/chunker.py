"""Document chunking for embedding.

Most of our documents (emails, Slack messages, calendar events, even most
Linear issues) are short enough to embed as a single chunk. The few that
are longer get a sliding-window split with overlap.

Tokens are approximated by whitespace splits — sufficient for chunk sizing
and avoids pulling in a tokenizer here. The embedder applies its own
tokenization at encode time.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CHUNK_TOKENS = 300
DEFAULT_OVERLAP_TOKENS = 50


@dataclass(slots=True, frozen=True)
class Chunk:
    index: int
    text: str


def chunk_text(
    text: str,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Split text into Chunks. Empty text → empty list. Short text → single chunk."""
    text = (text or "").strip()
    if not text:
        return []

    tokens = text.split()
    if len(tokens) <= chunk_tokens:
        return [Chunk(0, text)]

    step = chunk_tokens - overlap_tokens
    if step <= 0:
        raise ValueError("chunk_tokens must be larger than overlap_tokens")

    chunks: list[Chunk] = []
    i = 0
    idx = 0
    while i < len(tokens):
        window = tokens[i : i + chunk_tokens]
        chunks.append(Chunk(idx, " ".join(window)))
        idx += 1
        if i + chunk_tokens >= len(tokens):
            break
        i += step
    return chunks


def chunk_document(title: str | None, body: str, **kwargs) -> list[Chunk]:
    """Prepend the title to each chunk so retrieval hits include the title signal.

    For multi-chunk documents the title appears in every chunk — small redundancy
    cost in exchange for each chunk being self-contained when surfaced as a citation.
    """
    body = (body or "").strip()
    if not body and not title:
        return []
    prefix = f"{title}\n\n" if title else ""
    raw_chunks = chunk_text(body, **kwargs)
    if not raw_chunks:
        return [Chunk(0, prefix.rstrip())] if prefix else []
    return [Chunk(c.index, prefix + c.text) for c in raw_chunks]
