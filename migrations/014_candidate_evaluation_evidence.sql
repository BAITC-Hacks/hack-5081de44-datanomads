ALTER TABLE model_evaluations
    ADD COLUMN IF NOT EXISTS learning_cycle_id BIGINT
        REFERENCES learning_cycles(id) ON DELETE CASCADE,
    ADD COLUMN IF NOT EXISTS evaluation_payload JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE UNIQUE INDEX IF NOT EXISTS idx_model_evaluations_learning_cycle
    ON model_evaluations (learning_cycle_id)
    WHERE learning_cycle_id IS NOT NULL;
