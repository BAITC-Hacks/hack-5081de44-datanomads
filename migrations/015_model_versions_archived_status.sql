ALTER TABLE model_versions
    DROP CONSTRAINT IF EXISTS model_versions_status_check;

ALTER TABLE model_versions
    ADD CONSTRAINT model_versions_status_check
    CHECK (status IN ('CANDIDATE', 'SHADOW', 'PRODUCTION', 'REJECTED', 'ARCHIVED'));
