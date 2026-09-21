"""Application contracts for user, agent, crew, and run memory."""

import json
from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MemoryScope(StrEnum):
    USER_GLOBAL = "USER_GLOBAL"
    AGENT_PRIVATE = "AGENT_PRIVATE"
    CREW_SHARED = "CREW_SHARED"
    RUN_EPHEMERAL = "RUN_EPHEMERAL"


class Memory(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: int
    scope: MemoryScope
    agent_id: UUID | None = None
    run_id: UUID | None = None
    key: str
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None


class MemoryCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: int
    scope: MemoryScope
    agent_id: UUID | None = None
    run_id: UUID | None = None
    key: str
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def validate_scope_and_json(self) -> "MemoryCreate":
        if self.scope == MemoryScope.USER_GLOBAL:
            if self.agent_id is not None or self.run_id is not None:
                raise ValueError("USER_GLOBAL memory cannot reference an agent or run")
        elif self.scope == MemoryScope.AGENT_PRIVATE:
            if self.agent_id is None or self.run_id is not None:
                raise ValueError("AGENT_PRIVATE memory requires agent_id and cannot reference a run")
        elif self.run_id is None:
            raise ValueError(f"{self.scope.value} memory requires run_id")
        try:
            json.dumps(self.metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("Memory metadata must contain JSON-compatible values") from exc
        if not self.key.strip():
            raise ValueError("Memory key cannot be empty")
        return self
