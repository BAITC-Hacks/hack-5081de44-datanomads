ALTER TABLE relation_feedback
    ADD COLUMN IF NOT EXISTS suggestion_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS suggestion_threshold DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS suggestion_rule_version TEXT,
    ADD COLUMN IF NOT EXISTS suggestion_model_version TEXT,
    ADD COLUMN IF NOT EXISTS suggestion_distance_metric TEXT;

ALTER TABLE relation_feedback
    ADD CONSTRAINT relation_feedback_suggestion_score_range
        CHECK (suggestion_score IS NULL OR suggestion_score BETWEEN 0 AND 1),
    ADD CONSTRAINT relation_feedback_suggestion_threshold_range
        CHECK (suggestion_threshold IS NULL OR suggestion_threshold BETWEEN 0 AND 1);
