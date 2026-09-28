ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check;
ALTER TABLE users
    ADD CONSTRAINT users_role_check
    CHECK (role IN ('OPERATOR', 'MANAGER', 'ML_REVIEWER', 'ML_SERVICE', 'ADMIN'));

CREATE TABLE IF NOT EXISTS model_drift_triggers (
    evidence_id TEXT PRIMARY KEY,
    model_version TEXT NOT NULL REFERENCES model_versions(model_version),
    evidence_payload JSONB NOT NULL,
    state TEXT NOT NULL DEFAULT 'PENDING_REVIEW'
        CHECK (state IN ('PENDING_REVIEW', 'CYCLE_OPENED', 'DISMISSED')),
    learning_cycle_id BIGINT REFERENCES learning_cycles(id),
    created_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_by TEXT,
    reviewed_at TIMESTAMPTZ,
    CHECK ((state = 'CYCLE_OPENED') = (learning_cycle_id IS NOT NULL)),
    CHECK ((state = 'PENDING_REVIEW') = (reviewed_at IS NULL))
);

CREATE INDEX IF NOT EXISTS idx_model_drift_triggers_review
    ON model_drift_triggers (created_at DESC, evidence_id)
    WHERE state = 'PENDING_REVIEW';
