CREATE TABLE IF NOT EXISTS learning_cycle_candidates (
    learning_cycle_id BIGINT NOT NULL REFERENCES learning_cycles(id) ON DELETE CASCADE,
    model_version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'REGISTERED'
        CHECK (status IN (
            'REGISTERED', 'EVALUATING', 'EVALUATED', 'EVALUATION_FAILED',
            'PROMOTED', 'REJECTED'
        )),
    registered_by TEXT,
    registered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    evaluation_started_at TIMESTAMPTZ,
    evaluated_at TIMESTAMPTZ,
    PRIMARY KEY (learning_cycle_id, model_version)
);

INSERT INTO learning_cycle_candidates (
    learning_cycle_id, model_version, status, registered_at, evaluation_started_at
)
SELECT id,
       candidate_model_version,
       CASE
           WHEN state = 'PROMOTED' THEN 'PROMOTED'
           WHEN state = 'REJECTED' THEN 'REJECTED'
           WHEN state = 'EVALUATE' THEN 'EVALUATING'
           WHEN state = 'DECISION' AND EXISTS (
               SELECT 1
               FROM model_evaluations evaluation
               WHERE evaluation.learning_cycle_id = learning_cycles.id
                 AND evaluation.model_version = learning_cycles.candidate_model_version
           ) THEN 'EVALUATED'
           ELSE 'REGISTERED'
       END,
       created_at,
       evaluation_started_at
FROM learning_cycles
WHERE candidate_model_version IS NOT NULL
ON CONFLICT (learning_cycle_id, model_version) DO NOTHING;

DROP INDEX IF EXISTS idx_model_evaluations_learning_cycle;

CREATE UNIQUE INDEX IF NOT EXISTS idx_model_evaluations_cycle_model_version
    ON model_evaluations (learning_cycle_id, model_version)
    WHERE learning_cycle_id IS NOT NULL;

WITH ranked_production_models AS (
    SELECT id,
           ROW_NUMBER() OVER (ORDER BY created_at DESC, id DESC) AS position
    FROM model_versions
    WHERE status = 'PRODUCTION'
)
UPDATE model_versions AS model
SET status = 'ARCHIVED'
FROM ranked_production_models AS ranked
WHERE model.id = ranked.id
  AND ranked.position > 1;

CREATE UNIQUE INDEX IF NOT EXISTS idx_model_versions_single_production
    ON model_versions (status)
    WHERE status = 'PRODUCTION';

ALTER TABLE learning_cycle_shadow_predictions
    DROP CONSTRAINT IF EXISTS learning_cycle_shadow_predictions_pkey;

ALTER TABLE learning_cycle_shadow_predictions
    ADD CONSTRAINT learning_cycle_shadow_predictions_pkey
    PRIMARY KEY (learning_cycle_id, ticket_id, candidate_model_version);
