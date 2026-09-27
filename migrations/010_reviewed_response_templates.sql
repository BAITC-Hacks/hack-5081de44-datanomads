ALTER TABLE response_templates
    ADD COLUMN IF NOT EXISTS created_by TEXT,
    ADD COLUMN IF NOT EXISTS updated_by TEXT,
    ADD COLUMN IF NOT EXISTS approved_by TEXT,
    ADD COLUMN IF NOT EXISTS approved_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;

UPDATE response_templates
SET updated_at = created_at
WHERE updated_at IS NULL;

ALTER TABLE response_templates
    ALTER COLUMN updated_at SET DEFAULT now(),
    ALTER COLUMN updated_at SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_response_templates_approved_lookup
    ON response_templates (topic_id, service_id, language, version DESC)
    WHERE approved;
