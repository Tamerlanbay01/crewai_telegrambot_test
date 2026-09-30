-- Declarative agent memory settings and normalized persistent user crews.
-- Existing agents have no policy rows and keep the pre-migration memory behavior.

CREATE TABLE IF NOT EXISTS agent_memory_policies (
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    scope VARCHAR(32) NOT NULL,
    enabled BOOLEAN NOT NULL,
    PRIMARY KEY (agent_id, scope),
    CONSTRAINT ck_agent_memory_policy_scope CHECK (scope IN ('USER_GLOBAL', 'AGENT_PRIVATE'))
);

CREATE TABLE IF NOT EXISTS crews (
    id UUID PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    purpose TEXT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    process VARCHAR(20) NOT NULL,
    config_version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_crews_status CHECK (status IN ('active', 'archived')),
    CONSTRAINT ck_crews_process CHECK (process = 'sequential'),
    CONSTRAINT ck_crews_version CHECK (config_version > 0)
);
CREATE INDEX IF NOT EXISTS ix_crews_user_id ON crews(user_id);

CREATE TABLE IF NOT EXISTS crew_agents (
    id UUID PRIMARY KEY,
    crew_id UUID NOT NULL REFERENCES crews(id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE RESTRICT,
    position INTEGER NOT NULL,
    CONSTRAINT uq_crew_agents_member UNIQUE (crew_id, agent_id),
    CONSTRAINT uq_crew_agents_position UNIQUE (crew_id, position),
    CONSTRAINT ck_crew_agents_position CHECK (position >= 0)
);
CREATE INDEX IF NOT EXISTS ix_crew_agents_crew_id ON crew_agents(crew_id);
CREATE INDEX IF NOT EXISTS ix_crew_agents_agent_id ON crew_agents(agent_id);

CREATE TABLE IF NOT EXISTS crew_tasks (
    id UUID PRIMARY KEY,
    crew_id UUID NOT NULL,
    agent_id UUID NOT NULL,
    position INTEGER NOT NULL,
    description TEXT NOT NULL,
    expected_output TEXT NOT NULL,
    CONSTRAINT fk_crew_tasks_member FOREIGN KEY (crew_id, agent_id)
        REFERENCES crew_agents(crew_id, agent_id) ON DELETE CASCADE,
    CONSTRAINT uq_crew_tasks_position UNIQUE (crew_id, position),
    CONSTRAINT ck_crew_tasks_position CHECK (position >= 0)
);
CREATE INDEX IF NOT EXISTS ix_crew_tasks_crew_id ON crew_tasks(crew_id);

CREATE TABLE IF NOT EXISTS crew_skills (
    crew_id UUID NOT NULL REFERENCES crews(id) ON DELETE CASCADE,
    skill_id UUID NOT NULL REFERENCES skills(id) ON DELETE RESTRICT,
    required BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY (crew_id, skill_id)
);
