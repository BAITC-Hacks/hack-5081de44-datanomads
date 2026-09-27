-- The rule-based ML service reports classifier-demo-2026-09-21-001.
-- Keep the old registry row for historic feedback, and replace only the
-- untouched seeded production pointer. Neither row represents a trained artifact.

DO $$
BEGIN
    IF (SELECT COUNT(*) FROM model_versions WHERE status = 'PRODUCTION') = 1
       AND EXISTS (
           SELECT 1 FROM model_versions
           WHERE model_version = 'classifier-deterministic-baseline-2026-09-21'
             AND status = 'PRODUCTION'
             AND artifact_checksum = 'builtin-baseline-no-artifact'
       )
       AND NOT EXISTS (
           SELECT 1 FROM model_versions
           WHERE model_version = 'classifier-demo-2026-09-21-001'
       ) THEN
        UPDATE model_versions
        SET status = 'ARCHIVED'
        WHERE model_version = 'classifier-deterministic-baseline-2026-09-21';

        INSERT INTO model_versions (
            model_version, model_family, dataset_version, status,
            manifest_uri, artifact_checksum, created_at
        )
        SELECT
            'classifier-demo-2026-09-21-001',
            'deterministic-keyword-demo',
            dataset_version,
            'PRODUCTION',
            'builtin://deterministic-keyword-demo',
            'builtin-baseline-no-artifact',
            created_at
        FROM model_versions
        WHERE model_version = 'classifier-deterministic-baseline-2026-09-21';
    END IF;
END $$;
