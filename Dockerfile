# Multi-stage build for the Pierside Founder AI Assistant.
#
# Stage 1 (builder): install Python deps + pre-download the MiniLM weights
#   so the first request doesn't pay a ~90MB cold-start.
# Stage 2 (runtime): copy site-packages + huggingface cache + app source.
#
# Build:  docker build -t pierside .
# Run:    docker run -p 8000:8000 --env-file .env -v pierside_data:/app/data pierside

FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip \
 && pip install \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.32" \
        "sentence-transformers>=3.0" \
        "numpy>=2.0" \
        "httpx>=0.27" \
        "pydantic>=2.9" \
        "pydantic-settings>=2.6" \
        "python-dotenv>=1.0"

# Pre-cache MiniLM into /root/.cache/huggingface so runtime is offline-capable.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"


FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/fixtures/ ./data/fixtures/

RUN mkdir -p /app/data

EXPOSE 8000

# Ingest is idempotent: first boot populates app.db; subsequent boots are a no-op.
CMD ["sh", "-c", "python scripts/ingest.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
