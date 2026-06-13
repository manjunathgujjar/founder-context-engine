-- Founder AI Assistant context engine schema.
-- Single SQLite file, FTS5 for keyword search, BLOB column for dense embeddings.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    type         TEXT NOT NULL,
    title        TEXT,
    body         TEXT NOT NULL,
    author       TEXT,
    participants TEXT NOT NULL DEFAULT '[]',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_documents_source     ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents(updated_at);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

-- FTS5 virtual table over documents (title + body). Kept in sync by triggers below.
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title,
    body,
    content='documents',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, title, body) VALUES (new.rowid, new.title, new.body);
END;

CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, body) VALUES ('delete', old.rowid, old.title, old.body);
END;

CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, body) VALUES ('delete', old.rowid, old.title, old.body);
    INSERT INTO documents_fts(rowid, title, body) VALUES (new.rowid, new.title, new.body);
END;

-- Answer cache: sha256(normalized question) -> serialized AnswerResult JSON, 1hr TTL.
CREATE TABLE IF NOT EXISTS answer_cache (
    question_hash TEXT PRIMARY KEY,
    question      TEXT NOT NULL,
    answer_json   TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_answer_cache_created_at ON answer_cache(created_at);

-- Request log: one row per /api/ask call (hit or miss). Drives /api/stats.
CREATE TABLE IF NOT EXISTS request_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT NOT NULL,
    question          TEXT NOT NULL,
    cache_hit         INTEGER NOT NULL,
    refused           INTEGER NOT NULL,
    top_fused_score   REAL,
    cost              REAL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    retrieve_seconds  REAL,
    llm_seconds       REAL,
    total_seconds     REAL
);
CREATE INDEX IF NOT EXISTS idx_request_log_ts ON request_log(ts);
