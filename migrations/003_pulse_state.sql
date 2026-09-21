-- Pulse 109 P0 state: operator feedback, situation alerts and controlled
-- learning lifecycle.  PostgreSQL remains the source of truth even when the
-- local Core API uses deterministic in-memory fixtures for an offline demo.

CREATE TABLE IF NOT EXISTS response_templates (
    id BIGSERIAL PRIMARY KEY,
    template_key TEXT NOT NULL,
    language TEXT NOT NULL CHECK (language IN ('RU', 'KZ')),
    topic_id TEXT REFERENCES topics(id),
    service_id TEXT REFERENCES services(id),
    body TEXT NOT NULL,
    approved BOOLEAN NOT NULL DEFAULT FALSE,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (template_key, language, version)
);

CREATE TABLE IF NOT EXISTS alerts (
    id BIGSERIAL PRIMARY KEY,
    incident_key TEXT NOT NULL UNIQUE,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN',
    region_id TEXT REFERENCES regions(id),
    topic_id TEXT REFERENCES topics(id),
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    current_count INTEGER NOT NULL DEFAULT 0,
    baseline NUMERIC,
    deviation NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by TEXT
);

CREATE TABLE IF NOT EXISTS alert_ticket_links (
    alert_id BIGINT NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
    ticket_id BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    PRIMARY KEY (alert_id, ticket_id)
);

CREATE TABLE IF NOT EXISTS relation_feedback (
    id BIGSERIAL PRIMARY KEY,
    ticket_id BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    related_ticket_id BIGINT REFERENCES tickets(id) ON DELETE CASCADE,
    relation TEXT NOT NULL CHECK (relation IN ('DUPLICATE', 'REPEAT', 'SIMILAR', 'UNRELATED')),
    decision TEXT NOT NULL CHECK (decision IN ('CONFIRMED', 'REJECTED')),
    user_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS learning_cycles (
    id BIGSERIAL PRIMARY KEY,
    cycle_id TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK (state IN ('COLLECT', 'TRAINING', 'EVALUATE', 'DECISION', 'PROMOTED', 'REJECTED', 'INSUFFICIENT_FEEDBACK')),
    collect_started_at TIMESTAMPTZ,
    collect_ends_at TIMESTAMPTZ,
    evaluation_started_at TIMESTAMPTZ,
    evaluation_ends_at TIMESTAMPTZ,
    production_model_version TEXT,
    candidate_dataset_version TEXT,
    candidate_model_version TEXT,
    min_feedback_count INTEGER NOT NULL DEFAULT 1,
    promotion_policy_version TEXT NOT NULL DEFAULT 'policy-v1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS learning_feedback (
    id BIGSERIAL PRIMARY KEY,
    cycle_id BIGINT NOT NULL REFERENCES learning_cycles(id) ON DELETE CASCADE,
    ticket_id BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    production_model_version TEXT,
    production_prediction JSONB NOT NULL DEFAULT '{}'::jsonb,
    operator_confirmed_decision JSONB NOT NULL DEFAULT '{}'::jsonb,
    accepted_or_corrected TEXT NOT NULL CHECK (accepted_or_corrected IN ('ACCEPTED', 'CORRECTED')),
    validation_status TEXT NOT NULL DEFAULT 'VALID',
    feedback_created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_evaluations (
    id BIGSERIAL PRIMARY KEY,
    evaluation_id TEXT NOT NULL UNIQUE,
    model_version TEXT NOT NULL,
    evaluation_version TEXT NOT NULL,
    split_version TEXT NOT NULL,
    metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    shadow_metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    critical_regressions JSONB NOT NULL DEFAULT '[]'::jsonb,
    sample_size INTEGER NOT NULL DEFAULT 0,
    decision TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    evaluator TEXT
);

CREATE TABLE IF NOT EXISTS background_jobs (
    id BIGSERIAL PRIMARY KEY,
    job_type TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    state TEXT NOT NULL CHECK (state IN ('QUEUED', 'RUNNING', 'COMPLETED', 'FAILED')),
    attempt INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    error TEXT
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    role TEXT NOT NULL CHECK (role IN ('OPERATOR', 'MANAGER', 'ML_REVIEWER', 'ADMIN')),
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    actor_id TEXT,
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT,
    request_id TEXT,
    reason TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_alerts_region_topic_period ON alerts (region_id, topic_id, period_start);
CREATE INDEX IF NOT EXISTS idx_learning_feedback_cycle ON learning_feedback (cycle_id, feedback_created_at);
CREATE INDEX IF NOT EXISTS idx_background_jobs_claim ON background_jobs (state, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log (entity_type, entity_id, created_at DESC);
