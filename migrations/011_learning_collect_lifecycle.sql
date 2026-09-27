-- Persist the demo-only early-close policy on the cycle that owns the window.
ALTER TABLE learning_cycles
    ADD COLUMN IF NOT EXISTS manual_close_enabled BOOLEAN NOT NULL DEFAULT FALSE;

-- Give pre-lifecycle rows a deterministic collection window without changing
-- their model state or inventing a production baseline.
UPDATE learning_cycles
SET collect_started_at = COALESCE(collect_started_at, created_at),
    collect_ends_at = COALESCE(collect_ends_at, created_at + INTERVAL '168 hours'),
    min_feedback_count = GREATEST(min_feedback_count, 1),
    promotion_policy_version = COALESCE(NULLIF(BTRIM(promotion_policy_version), ''), 'policy-v1');

ALTER TABLE learning_cycles
    ADD CONSTRAINT learning_cycles_min_feedback_count_positive
    CHECK (min_feedback_count > 0) NOT VALID;

ALTER TABLE learning_cycles
    VALIDATE CONSTRAINT learning_cycles_min_feedback_count_positive;

CREATE INDEX IF NOT EXISTS idx_learning_cycles_collect_expiry
    ON learning_cycles (collect_ends_at, id)
    WHERE state = 'COLLECT';

CREATE INDEX IF NOT EXISTS idx_learning_cycles_active_state
    ON learning_cycles (state, created_at DESC)
    WHERE state IN ('COLLECT', 'TRAINING', 'EVALUATE', 'DECISION');
