ALTER TABLE dataset_versions ADD COLUMN IF NOT EXISTS content_sha256 TEXT;

CREATE TABLE IF NOT EXISTS dataset_ticket_links (
    dataset_version TEXT NOT NULL REFERENCES dataset_versions(dataset_version),
    ticket_id BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    PRIMARY KEY (dataset_version, ticket_id)
);

CREATE INDEX IF NOT EXISTS idx_dataset_ticket_links_ticket ON dataset_ticket_links(ticket_id);
