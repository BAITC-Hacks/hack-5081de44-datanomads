-- Pulse 109 Data Foundation, migration 001.
-- PostgreSQL is the source of truth for normalized tickets and import audit.
-- Raw source files are intentionally not represented by this schema.

CREATE TABLE IF NOT EXISTS regions (
    id TEXT PRIMARY KEY,
    name_ru TEXT NOT NULL,
    name_kk TEXT NOT NULL,
    name_en TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS topics (
    id TEXT PRIMARY KEY,
    parent_id TEXT REFERENCES topics(id),
    name_ru TEXT NOT NULL,
    name_kk TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS services (
    id TEXT PRIMARY KEY,
    name_ru TEXT NOT NULL,
    name_kk TEXT,
    active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS topic_source_mappings (
    source_system TEXT NOT NULL,
    raw_direction TEXT NOT NULL,
    topic_id TEXT NOT NULL REFERENCES topics(id),
    canonical_subtopic TEXT,
    confidence NUMERIC(5, 4),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source_system, raw_direction)
);

CREATE TABLE IF NOT EXISTS tickets (
    id BIGSERIAL PRIMARY KEY,
    external_ticket_id TEXT NOT NULL,
    source_system TEXT NOT NULL,
    region_id TEXT NOT NULL REFERENCES regions(id),
    created_at TIMESTAMPTZ NOT NULL,
    original_text TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'UNKNOWN',
    topic_raw TEXT NOT NULL DEFAULT 'UNKNOWN',
    topic_id TEXT NOT NULL REFERENCES topics(id),
    service_raw TEXT,
    service_id TEXT REFERENCES services(id),
    priority TEXT,
    status TEXT NOT NULL DEFAULT 'UNKNOWN',
    district TEXT,
    address TEXT,
    coordinates JSONB,
    object TEXT,
    channel TEXT,
    closed_at TIMESTAMPTZ,
    deadline_at TIMESTAMPTZ,
    resolution_text TEXT,
    official_response TEXT,
    attachments JSONB NOT NULL DEFAULT '[]'::jsonb,
    assignment_history JSONB NOT NULL DEFAULT '[]'::jsonb,
    pulse_prediction JSONB,
    operator_confirmed_decision JSONB,
    model_versions JSONB NOT NULL DEFAULT '{}'::jsonb,
    needs_review BOOLEAN NOT NULL DEFAULT FALSE,
    embedding_ref TEXT,
    duplicate_feedback TEXT,
    repeat_feedback TEXT,
    text_redaction_count INTEGER NOT NULL DEFAULT 0,
    schema_version TEXT NOT NULL DEFAULT 'unified-ticket.v1',
    created_in_pulse_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_in_pulse_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_system, external_ticket_id),
    CHECK (text_redaction_count >= 0),
    CHECK (language IN ('RU', 'KZ', 'OTHER', 'UNKNOWN'))
);

CREATE INDEX IF NOT EXISTS idx_tickets_region_created ON tickets (region_id, created_at);
CREATE INDEX IF NOT EXISTS idx_tickets_topic_created ON tickets (topic_id, created_at);
CREATE INDEX IF NOT EXISTS idx_tickets_source_created ON tickets (source_system, created_at);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets (status);

CREATE TABLE IF NOT EXISTS ticket_predictions (
    id BIGSERIAL PRIMARY KEY,
    ticket_id BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    model_version TEXT NOT NULL,
    topic_id TEXT REFERENCES topics(id),
    service_id TEXT REFERENCES services(id),
    priority TEXT,
    confidence NUMERIC(6, 5),
    alternatives JSONB NOT NULL DEFAULT '[]'::jsonb,
    prediction JSONB NOT NULL DEFAULT '{}'::jsonb,
    needs_review BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ticket_predictions_ticket ON ticket_predictions (ticket_id, created_at DESC);

CREATE TABLE IF NOT EXISTS operator_decisions (
    id BIGSERIAL PRIMARY KEY,
    ticket_id BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    user_id TEXT,
    confirmed_topic_id TEXT REFERENCES topics(id),
    confirmed_service_id TEXT REFERENCES services(id),
    confirmed_priority TEXT,
    decision TEXT NOT NULL CHECK (decision IN ('CONFIRMED', 'CORRECTED')),
    feedback JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS similarity_feedback (
    id BIGSERIAL PRIMARY KEY,
    ticket_id BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    related_ticket_id BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    relation TEXT NOT NULL CHECK (relation IN ('DUPLICATE', 'REPEAT', 'SIMILAR', 'UNRELATED')),
    user_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (ticket_id, related_ticket_id, relation)
);

CREATE TABLE IF NOT EXISTS data_import_runs (
    id BIGSERIAL PRIMARY KEY,
    source_system TEXT NOT NULL,
    source_uri TEXT,
    dataset_version TEXT,
    status TEXT NOT NULL CHECK (status IN ('RUNNING', 'COMPLETED', 'FAILED')),
    total_rows INTEGER NOT NULL DEFAULT 0,
    valid_rows INTEGER NOT NULL DEFAULT 0,
    quarantined_rows INTEGER NOT NULL DEFAULT 0,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    error_code TEXT,
    error_detail TEXT
);

CREATE TABLE IF NOT EXISTS quarantine_rows (
    id BIGSERIAL PRIMARY KEY,
    import_run_id BIGINT REFERENCES data_import_runs(id) ON DELETE SET NULL,
    source_system TEXT NOT NULL,
    row_number INTEGER NOT NULL,
    reason TEXT NOT NULL CHECK (reason IN ('BAD_CSV_STRUCTURE', 'INVALID_DATE', 'MISSING_REQUIRED_FIELD', 'UNKNOWN_SCHEMA', 'PII_REVIEW', 'INVALID_VALUE')),
    field TEXT,
    detail TEXT NOT NULL,
    row_snapshot JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_quarantine_source_reason ON quarantine_rows (source_system, reason);

CREATE TABLE IF NOT EXISTS dataset_versions (
    id BIGSERIAL PRIMARY KEY,
    dataset_version TEXT NOT NULL UNIQUE,
    schema_version TEXT NOT NULL,
    manifest_uri TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    is_synthetic BOOLEAN NOT NULL DEFAULT FALSE,
    record_count INTEGER NOT NULL DEFAULT 0,
    quarantine_record_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_versions (
    id BIGSERIAL PRIMARY KEY,
    model_version TEXT NOT NULL UNIQUE,
    model_family TEXT NOT NULL,
    dataset_version TEXT REFERENCES dataset_versions(dataset_version),
    status TEXT NOT NULL CHECK (status IN ('CANDIDATE', 'SHADOW', 'PRODUCTION', 'REJECTED')),
    manifest_uri TEXT,
    artifact_checksum TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    promoted_at TIMESTAMPTZ
);

