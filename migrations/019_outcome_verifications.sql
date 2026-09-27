-- Pulse outcome verification is separate from the imported official ticket status.
-- UNKNOWN is represented by the absence of evidence, so this table stores only observed outcomes.
CREATE TABLE IF NOT EXISTS outcome_verifications (
    id BIGSERIAL PRIMARY KEY,
    ticket_id BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    state TEXT NOT NULL CHECK (state IN ('VERIFIED', 'PARTIAL', 'DISPUTED')),
    source_system TEXT NOT NULL CHECK (BTRIM(source_system) <> ''),
    channel TEXT NOT NULL CHECK (BTRIM(channel) <> ''),
    actor_user_id TEXT NOT NULL CHECK (BTRIM(actor_user_id) <> ''),
    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_outcome_verifications_ticket_created
    ON outcome_verifications (ticket_id, created_at DESC, id DESC);
