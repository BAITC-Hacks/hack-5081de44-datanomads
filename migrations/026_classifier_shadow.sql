CREATE TABLE IF NOT EXISTS classifier_shadow_predictions (
    id BIGSERIAL PRIMARY KEY,
    cycle_id BIGINT NOT NULL REFERENCES learning_cycles(id) ON DELETE CASCADE,
    ticket_id BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    production_prediction_id BIGINT NOT NULL REFERENCES ticket_predictions(id) ON DELETE CASCADE,
    candidate_model_version TEXT NOT NULL REFERENCES model_versions(model_version),
    candidate_artifact_checksum TEXT NOT NULL,
    topic_id TEXT NOT NULL REFERENCES topics(id),
    confidence NUMERIC(6, 5) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    predicted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (cycle_id, ticket_id)
);

CREATE INDEX IF NOT EXISTS idx_classifier_shadow_cycle_time
    ON classifier_shadow_predictions (cycle_id, predicted_at);
