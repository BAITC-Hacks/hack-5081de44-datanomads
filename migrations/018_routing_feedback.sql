-- Service routing feedback is stored separately from operator decisions.
-- Demo feedback remains an explicit fixture and never changes production rules.
CREATE TABLE IF NOT EXISTS routing_feedback (
    id BIGSERIAL PRIMARY KEY,
    ticket_id BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    operator_decision_id BIGINT NOT NULL REFERENCES operator_decisions(id) ON DELETE RESTRICT,
    original_route_recommendation TEXT NOT NULL CHECK (BTRIM(original_route_recommendation) <> ''),
    operator_confirmed_route TEXT NOT NULL CHECK (BTRIM(operator_confirmed_route) <> ''),
    service_feedback TEXT NOT NULL CHECK (service_feedback IN ('ACCEPTED', 'CORRECTED')),
    corrected_target_service_id TEXT REFERENCES services(id),
    corrected_target_service TEXT,
    actor_user_id TEXT NOT NULL CHECK (BTRIM(actor_user_id) <> ''),
    source_system TEXT NOT NULL CHECK (source_system = 'DEMO_SIMULATION'),
    evaluation_status TEXT NOT NULL DEFAULT 'PENDING_OFFLINE_REVIEW'
        CHECK (evaluation_status = 'PENDING_OFFLINE_REVIEW'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (
        (service_feedback = 'ACCEPTED' AND corrected_target_service_id IS NULL AND corrected_target_service IS NULL)
        OR
        (service_feedback = 'CORRECTED' AND corrected_target_service_id IS NOT NULL AND corrected_target_service IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_routing_feedback_ticket_created
    ON routing_feedback (ticket_id, created_at DESC, id DESC);
