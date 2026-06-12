"""Local sentence-transformers wrapper.

Lazy-loaded so importing this module is cheap; the ~80 MB MiniLM model only
materializes on first encode() call. Vectors are L2-normalized so cosine
similarity reduces to a dot product at query time.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from app.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _model() -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.embedding_model)


def embedding_dim() -> int:
    return int(_model().get_sentence_embedding_dimension())


def encode(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed a list of texts. Returns float32 (N, D), L2-normalized."""
    if not texts:
        return np.zeros((0, embedding_dim()), dtype=np.float32)
    vectors = _model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32, copy=False)


def encode_one(text: str) -> np.ndarray:
    return encode([text])[0]


def to_blob(vec: np.ndarray) -> bytes:
    """Serialize a float32 vector for SQLite BLOB storage."""
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    """Inverse of to_blob."""
    return np.frombuffer(blob, dtype=np.float32)
