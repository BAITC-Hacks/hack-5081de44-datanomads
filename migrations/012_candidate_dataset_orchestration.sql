ALTER TABLE learning_cycles
    ADD COLUMN IF NOT EXISTS frozen_evaluation_dataset_version TEXT
        REFERENCES dataset_versions(dataset_version),
    ADD COLUMN IF NOT EXISTS candidate_dataset_checksum TEXT;

ALTER TABLE learning_cycles
    DROP CONSTRAINT IF EXISTS learning_cycles_state_check;

ALTER TABLE learning_cycles
    ADD CONSTRAINT learning_cycles_state_check
    CHECK (state IN (
        'COLLECT', 'TRAINING', 'EVALUATE', 'DECISION', 'PROMOTED', 'REJECTED',
        'INSUFFICIENT_FEEDBACK', 'DATASET_BUILD_FAILED'
    ));

CREATE TABLE IF NOT EXISTS learning_cycle_evaluation_tickets (
    learning_cycle_id BIGINT NOT NULL REFERENCES learning_cycles(id) ON DELETE CASCADE,
    dataset_version TEXT NOT NULL REFERENCES dataset_versions(dataset_version),
    ticket_id BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    PRIMARY KEY (learning_cycle_id, ticket_id)
);

CREATE INDEX IF NOT EXISTS idx_learning_cycle_evaluation_tickets_ticket
    ON learning_cycle_evaluation_tickets (ticket_id);
