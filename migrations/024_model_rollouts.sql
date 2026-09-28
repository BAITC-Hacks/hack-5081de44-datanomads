ALTER TABLE learning_cycle_candidates
    DROP CONSTRAINT IF EXISTS learning_cycle_candidates_status_check;

ALTER TABLE learning_cycle_candidates
    ADD CONSTRAINT learning_cycle_candidates_status_check
    CHECK (status IN (
        'REGISTERED', 'EVALUATING', 'EVALUATED', 'EVALUATION_FAILED',
        'CANARY', 'MONITORING', 'PROMOTED', 'REJECTED'
    ));

ALTER TABLE learning_cycles
    DROP CONSTRAINT IF EXISTS learning_cycles_state_check;

ALTER TABLE learning_cycles
    ADD CONSTRAINT learning_cycles_state_check
    CHECK (state IN (
        'COLLECT', 'TRAINING', 'TRAINING_FAILED', 'EVALUATE', 'DECISION',
        'CANARY', 'MONITORING', 'PROMOTED', 'REJECTED', 'INSUFFICIENT_FEEDBACK',
        'DATASET_BUILD_FAILED'
    ));

CREATE TABLE IF NOT EXISTS model_rollouts (
    id BIGSERIAL PRIMARY KEY,
    rollout_id TEXT NOT NULL UNIQUE,
    rollout_version INTEGER NOT NULL CHECK (rollout_version > 0),
    learning_cycle_id BIGINT REFERENCES learning_cycles(id) ON DELETE RESTRICT,
    candidate_model_version TEXT NOT NULL REFERENCES model_versions(model_version) ON DELETE RESTRICT,
    candidate_artifact_checksum TEXT NOT NULL,
    previous_production_model_version TEXT NOT NULL REFERENCES model_versions(model_version) ON DELETE RESTRICT,
    previous_artifact_checksum TEXT,
    canary_traffic_percent SMALLINT NOT NULL CHECK (canary_traffic_percent BETWEEN 1 AND 25),
    policy_version TEXT NOT NULL,
    policy_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL CHECK (status IN ('CANARY', 'MONITORING', 'FULL_PRODUCTION', 'ROLLED_BACK', 'CANCELLED')),
    created_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    monitoring_started_at TIMESTAMPTZ,
    full_production_at TIMESTAMPTZ,
    rollback_triggered_at TIMESTAMPTZ,
    rolled_back_at TIMESTAMPTZ,
    rollback_reason TEXT,
    baseline_correction_rate DOUBLE PRECISION,
    UNIQUE (candidate_model_version, rollout_version),
    CHECK (candidate_model_version <> previous_production_model_version),
    CHECK ((status <> 'FULL_PRODUCTION') OR full_production_at IS NOT NULL),
    CHECK ((status <> 'ROLLED_BACK') OR rolled_back_at IS NOT NULL)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_model_rollouts_single_active
    ON model_rollouts ((1))
    WHERE status IN ('CANARY', 'MONITORING');

CREATE INDEX IF NOT EXISTS idx_model_rollouts_created
    ON model_rollouts (created_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS model_rollout_inferences (
    id BIGSERIAL PRIMARY KEY,
    rollout_id BIGINT NOT NULL REFERENCES model_rollouts(id) ON DELETE RESTRICT,
    ticket_id BIGINT REFERENCES tickets(id) ON DELETE SET NULL,
    traffic_group TEXT NOT NULL CHECK (traffic_group IN ('CANARY', 'CONTROL', 'FULL_PRODUCTION')),
    model_version TEXT NOT NULL REFERENCES model_versions(model_version) ON DELETE RESTRICT,
    inference_status TEXT NOT NULL DEFAULT 'ASSIGNED'
        CHECK (inference_status IN ('ASSIGNED', 'COMPLETED', 'FAILED')),
    candidate_topic_id TEXT,
    production_topic_id TEXT,
    error_code TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    CHECK ((inference_status = 'FAILED') OR error_code IS NULL),
    CHECK ((inference_status = 'ASSIGNED') OR completed_at IS NOT NULL),
    CHECK ((traffic_group <> 'CANARY' OR inference_status <> 'COMPLETED') OR candidate_topic_id IS NOT NULL),
    UNIQUE (rollout_id, ticket_id)
);

CREATE INDEX IF NOT EXISTS idx_model_rollout_inferences_rollout_created
    ON model_rollout_inferences (rollout_id, created_at, id);

CREATE INDEX IF NOT EXISTS idx_model_rollout_inferences_rollout_ticket
    ON model_rollout_inferences (rollout_id, ticket_id)
    WHERE ticket_id IS NOT NULL;
