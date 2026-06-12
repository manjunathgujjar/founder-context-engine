"""OpenRouter LLM client (httpx).

OpenRouter is OpenAI-compatible. We POST to /chat/completions with low
temperature for grounded behavior, with bounded retries on transient errors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import settings


class LLMError(RuntimeError):
    """Surfaces in the answer pipeline when synthesis fails."""


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    usage: dict[str, int]
    latency_seconds: float


_TIMEOUT_SECONDS = 30.0
_MAX_ATTEMPTS = 3
_BASE_BACKOFF = 1.0
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def call(
    messages: list[dict],
    *,
    temperature: float = 0.1,
    max_tokens: int | None = None,
) -> LLMResponse:
    """Synchronous chat completion via OpenRouter. Raises LLMError on permanent failures."""
    if not settings.openrouter_api_key:
        raise LLMError("OPENROUTER_API_KEY is not set. Add it to .env (see .env.example).")

    url = f"{settings.openrouter_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/pierside/founder-ai-assistant",
        "X-Title": "Pierside Founder AI Assistant",
    }
    body: dict[str, Any] = {
        "model": settings.openrouter_model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    last_error: str | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
                resp = client.post(url, headers=headers, json=body)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_error = f"network/timeout: {exc}"
            if attempt == _MAX_ATTEMPTS:
                break
            time.sleep(_BASE_BACKOFF * (2 ** (attempt - 1)))
            continue

        if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_ATTEMPTS:
            last_error = f"http {resp.status_code}: {resp.text[:200]}"
            time.sleep(_BASE_BACKOFF * (2 ** (attempt - 1)))
            continue

        if resp.status_code >= 400:
            raise LLMError(f"OpenRouter returned {resp.status_code}: {resp.text[:400]}")

        latency = time.perf_counter() - started
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected response shape: {data}") from exc

        return LLMResponse(
            text=text,
            model=data.get("model", settings.openrouter_model),
            usage=data.get("usage", {}) or {},
            latency_seconds=latency,
        )

    raise LLMError(f"OpenRouter call failed after {_MAX_ATTEMPTS} attempts. Last: {last_error}")
