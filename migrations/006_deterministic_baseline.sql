-- Register the deterministic baseline explicitly.  It is a production
-- pointer for the current rule-based runtime, not a trained artifact and has
-- no fabricated evaluation metrics.

INSERT INTO dataset_versions (
    dataset_version, schema_version, manifest_uri, manifest_sha256,
    is_synthetic, record_count, quarantine_record_count
)
VALUES (
    'baseline-deterministic-2026-09-21', 'unified-ticket-v1',
    'builtin://baseline-deterministic', 'builtin-baseline-no-artifact',
    FALSE, 0, 0
)
ON CONFLICT (dataset_version) DO NOTHING;

INSERT INTO model_versions (
    model_version, model_family, dataset_version, status, manifest_uri,
    artifact_checksum
)
SELECT
    'classifier-deterministic-baseline-2026-09-21',
    'classifier-deterministic-baseline',
    'baseline-deterministic-2026-09-21',
    'PRODUCTION',
    'builtin://deterministic-classifier',
    'builtin-baseline-no-artifact'
WHERE NOT EXISTS (SELECT 1 FROM model_versions WHERE status = 'PRODUCTION')
ON CONFLICT (model_version) DO NOTHING;
