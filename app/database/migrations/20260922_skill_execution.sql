-- PostgreSQL migration for controlled, auditable executable-skill runs.
-- Package bytes remain in S3-compatible storage; this table stores only bounded
-- execution metadata and output previews.

DO $$ BEGIN
    CREATE TYPE skill_execution_status AS ENUM (
        'PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'TIMED_OUT',
        'CANCELLED', 'DENIED', 'SANDBOX_ERROR'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE skill_execution_subject_type AS ENUM (
        'persistent_agent', 'system_agent', 'temporary_subagent'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TYPE agent_run_event_type ADD VALUE IF NOT EXISTS 'skill_execution_requested';
ALTER TYPE agent_run_event_type ADD VALUE IF NOT EXISTS 'skill_execution_started';
ALTER TYPE agent_run_event_type ADD VALUE IF NOT EXISTS 'skill_execution_completed';
ALTER TYPE agent_run_event_type ADD VALUE IF NOT EXISTS 'skill_execution_failed';
ALTER TYPE agent_run_event_type ADD VALUE IF NOT EXISTS 'skill_execution_timed_out';

CREATE TABLE IF NOT EXISTS skill_executions (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    skill_id UUID NOT NULL REFERENCES skills(id) ON DELETE RESTRICT,
    skill_version INTEGER NOT NULL,
    requesting_subject_type skill_execution_subject_type NOT NULL,
    requesting_subject_id VARCHAR(255) NOT NULL,
    idempotency_key VARCHAR(128) NOT NULL UNIQUE,
    status skill_execution_status NOT NULL DEFAULT 'PENDING',
    arguments_sanitized JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,
    exit_code INTEGER,
    stdout_preview TEXT NOT NULL DEFAULT '',
    stderr_preview TEXT NOT NULL DEFAULT '',
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_skill_executions_version_positive CHECK (skill_version > 0)
);

CREATE INDEX IF NOT EXISTS ix_skill_executions_run_id ON skill_executions (run_id);
CREATE INDEX IF NOT EXISTS ix_skill_executions_user_id ON skill_executions (user_id);
CREATE INDEX IF NOT EXISTS ix_skill_executions_skill_id ON skill_executions (skill_id);
CREATE INDEX IF NOT EXISTS ix_skill_executions_run_created
    ON skill_executions (run_id, created_at);
