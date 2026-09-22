-- PostgreSQL migration for the semantic-memory and post-run extraction layers.
-- Apply through the deployment migration runner before enabling QDRANT_URL.

DO $$ BEGIN
    CREATE TYPE memory_type AS ENUM (
        'PREFERENCE', 'FACT', 'DECISION', 'GOAL', 'PROJECT_CONTEXT', 'EPISODE'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS memory_type memory_type NOT NULL DEFAULT 'FACT';

DO $$ BEGIN
    CREATE TYPE memory_extraction_status AS ENUM ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE memory_candidate_decision AS ENUM (
        'PENDING', 'SAVED', 'UPDATED', 'MERGED', 'IGNORED', 'REJECTED'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS memory_extraction_runs (
    id UUID PRIMARY KEY,
    agent_run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status memory_extraction_status NOT NULL DEFAULT 'PENDING',
    model_name VARCHAR(255),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    candidates_extracted_at TIMESTAMPTZ,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    next_retry_at TIMESTAMPTZ,
    claim_token UUID,
    claim_expires_at TIMESTAMPTZ,
    error VARCHAR(2048),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_memory_extraction_run_agent_run UNIQUE (agent_run_id)
);

CREATE INDEX IF NOT EXISTS ix_memory_extraction_runs_user_id
    ON memory_extraction_runs (user_id);

ALTER TABLE memory_extraction_runs
    ADD COLUMN IF NOT EXISTS candidates_extracted_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS attempt_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS next_retry_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS claim_token UUID,
    ADD COLUMN IF NOT EXISTS claim_expires_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS ix_memory_extraction_runs_next_retry_at
    ON memory_extraction_runs (next_retry_at);

CREATE INDEX IF NOT EXISTS ix_memory_extraction_runs_claim_expires_at
    ON memory_extraction_runs (claim_expires_at);

CREATE TABLE IF NOT EXISTS memory_candidates (
    id UUID PRIMARY KEY,
    extraction_run_id UUID NOT NULL REFERENCES memory_extraction_runs(id) ON DELETE CASCADE,
    memory_type memory_type NOT NULL,
    scope memory_scope NOT NULL,
    agent_id UUID REFERENCES agents(id) ON DELETE SET NULL,
    key VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    reason TEXT NOT NULL,
    explicit_user_request BOOLEAN NOT NULL DEFAULT FALSE,
    decision memory_candidate_decision NOT NULL DEFAULT 'PENDING',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_memory_candidates_extraction_run_id
    ON memory_candidates (extraction_run_id);
