-- Manager-led signal monitoring remains separate from ACK/CLOSE and preserves
-- each completed observation window with its source ticket IDs.
CREATE TABLE IF NOT EXISTS alert_monitoring (
    id BIGSERIAL PRIMARY KEY,
    alert_id BIGINT NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
    state TEXT NOT NULL DEFAULT 'MONITORING'
        CHECK (state IN ('MONITORING', 'STABILIZED', 'PERSISTING', 'WORSENING', 'RECURRED', 'INSUFFICIENT_HISTORY')),
    monitoring_period_days INTEGER NOT NULL,
    observation_period_days INTEGER NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ends_at TIMESTAMPTZ NOT NULL,
    started_by TEXT NOT NULL,
    detector_version TEXT,
    detector_config JSONB,
    initial_count INTEGER,
    baseline NUMERIC,
    median_absolute_deviation NUMERIC,
    result JSONB NOT NULL DEFAULT '{}'::jsonb,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (monitoring_period_days > 0),
    CHECK (observation_period_days > 0),
    CHECK (monitoring_period_days % observation_period_days = 0),
    CHECK (monitoring_period_days / observation_period_days <= 12),
    CHECK (ends_at > started_at),
    CHECK ((state = 'MONITORING' AND completed_at IS NULL) OR
           (state <> 'MONITORING' AND completed_at IS NOT NULL))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_alert_monitoring_one_active
    ON alert_monitoring (alert_id)
    WHERE state = 'MONITORING';

CREATE INDEX IF NOT EXISTS idx_alert_monitoring_latest
    ON alert_monitoring (alert_id, started_at DESC, id DESC);
