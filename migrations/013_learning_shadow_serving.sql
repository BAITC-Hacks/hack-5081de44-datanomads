ALTER TABLE learning_cycles
    ADD COLUMN IF NOT EXISTS blind_ab_enabled BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE learning_cycles
    DROP CONSTRAINT IF EXISTS learning_cycles_state_check;

ALTER TABLE learning_cycles
    ADD CONSTRAINT learning_cycles_state_check
    CHECK (state IN (
        'COLLECT', 'TRAINING', 'TRAINING_FAILED', 'EVALUATE', 'DECISION',
        'PROMOTED', 'REJECTED', 'INSUFFICIENT_FEEDBACK', 'DATASET_BUILD_FAILED'
    ));

CREATE TABLE IF NOT EXISTS learning_cycle_shadow_predictions (
    learning_cycle_id BIGINT NOT NULL REFERENCES learning_cycles(id),
    ticket_id BIGINT NOT NULL REFERENCES tickets(id),
    predicted_at TIMESTAMPTZ NOT NULL,
    production_model_version TEXT NOT NULL,
    candidate_model_version TEXT NOT NULL,
    production_prediction JSONB NOT NULL,
    candidate_prediction JSONB,
    candidate_inference_status TEXT NOT NULL
        CHECK (candidate_inference_status IN ('COMPLETED', 'FAILED')),
    candidate_error_code TEXT,
    operator_decision_id BIGINT REFERENCES operator_decisions(id) ON DELETE SET NULL,
    PRIMARY KEY (learning_cycle_id, ticket_id),
    CHECK (
        (candidate_inference_status = 'COMPLETED'
            AND candidate_prediction IS NOT NULL
            AND candidate_error_code IS NULL)
        OR
        (candidate_inference_status = 'FAILED'
            AND candidate_prediction IS NULL
            AND candidate_error_code IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_learning_cycle_shadow_predictions_ticket
    ON learning_cycle_shadow_predictions (ticket_id);

CREATE INDEX IF NOT EXISTS idx_learning_cycles_evaluation_expiry
    ON learning_cycles (evaluation_ends_at, id)
    WHERE state = 'EVALUATE';
