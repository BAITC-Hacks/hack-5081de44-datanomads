-- Rolling forecast snapshots keep the model result, filter slice and
-- successive observed/forecast comparisons reproducible.
CREATE TABLE IF NOT EXISTS forecast_runs (
    id BIGSERIAL PRIMARY KEY,
    filter_state JSONB NOT NULL,
    horizon_days INTEGER NOT NULL CHECK (horizon_days IN (30, 60, 90)),
    model_version TEXT NOT NULL,
    issued_on DATE NOT NULL,
    issued_at TIMESTAMPTZ NOT NULL,
    previous_run_id BIGINT REFERENCES forecast_runs(id),
    response JSONB NOT NULL,
    comparison JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (filter_state, horizon_days, model_version, issued_on)
);

CREATE INDEX IF NOT EXISTS idx_forecast_runs_slice_latest
    ON forecast_runs (filter_state, horizon_days, issued_on DESC, id DESC);

CREATE TABLE IF NOT EXISTS forecast_manager_signals (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL UNIQUE REFERENCES forecast_runs(id) ON DELETE CASCADE,
    previous_run_id BIGINT NOT NULL REFERENCES forecast_runs(id),
    policy_version TEXT NOT NULL,
    previous_peak_date DATE NOT NULL,
    updated_peak_date DATE NOT NULL,
    previous_peak INTEGER NOT NULL CHECK (previous_peak >= 0),
    updated_peak INTEGER NOT NULL CHECK (updated_peak >= 0),
    delta BIGINT NOT NULL,
    threshold NUMERIC NOT NULL CHECK (threshold >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_forecast_manager_signals_latest
    ON forecast_manager_signals (created_at DESC, id DESC);
