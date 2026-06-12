"""Runtime configuration loaded from .env / environment."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="anthropic/claude-haiku-4.5", alias="OPENROUTER_MODEL")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )

    db_path: Path = Field(default=PROJECT_ROOT / "data" / "app.db", alias="DB_PATH")
    fixtures_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "fixtures", alias="FIXTURES_DIR"
    )

    retrieval_top_k: int = Field(default=8, alias="RETRIEVAL_TOP_K")

    # Pin TODAY to a fixed ISO date (e.g. "2026-06-12") for deterministic demo
    # answers. Leave unset to use real current date at request time.
    pinned_today: str | None = Field(default=None, alias="PINNED_TODAY")


settings = Settings()
