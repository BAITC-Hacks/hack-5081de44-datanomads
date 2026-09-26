CREATE TABLE IF NOT EXISTS vector_index_state (
    singleton_id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (singleton_id = 1),
    embedder_version TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension BETWEEN 8 AND 1024),
    distance_metric TEXT NOT NULL CHECK (distance_metric IN ('Cosine', 'Dot', 'Euclid')),
    collection_name TEXT NOT NULL,
    generation BIGINT NOT NULL DEFAULT 1 CHECK (generation > 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
