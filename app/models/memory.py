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


class MemoryType(StrEnum):
    PREFERENCE = "PREFERENCE"
    FACT = "FACT"
    DECISION = "DECISION"
    GOAL = "GOAL"
    PROJECT_CONTEXT = "PROJECT_CONTEXT"
    EPISODE = "EPISODE"


class MemoryExtractionStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class MemoryCandidateDecision(StrEnum):
    PENDING = "PENDING"
    SAVED = "SAVED"
    UPDATED = "UPDATED"
    MERGED = "MERGED"
    IGNORED = "IGNORED"
    REJECTED = "REJECTED"


class Memory(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: int
    scope: MemoryScope
    memory_type: MemoryType = MemoryType.FACT
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
    memory_type: MemoryType = MemoryType.FACT
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


class MemoryCandidate(BaseModel):
    """A proposed long-term memory; it has no persistence authority."""

    memory_type: MemoryType
    scope: MemoryScope
    agent_id: UUID | None = None
    key: str
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    explicit_user_request: bool = False


class MemoryExtractionResult(BaseModel):
    candidates: list[MemoryCandidate] = Field(default_factory=list)


class MemoryExtractionContext(BaseModel):
    """Sanitized completed-run material available to the internal Memory Agent."""

    agent_run_id: UUID
    user_id: int
    original_message: str
    final_response: str
    delegation_summaries: list[str] = Field(default_factory=list)
    existing_memory: list[dict[str, object]] = Field(default_factory=list)
    eligible_agent_ids: list[UUID] = Field(default_factory=list)
    explicit_memory_request: bool = False


class MemoryExtractionRun(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    agent_run_id: UUID
    user_id: int
    status: MemoryExtractionStatus
    model_name: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    candidates_extracted_at: datetime | None = None
    attempt_count: int = 0
    next_retry_at: datetime | None = None
    claim_token: UUID | None = None
    claim_expires_at: datetime | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class MemoryCandidateAudit(MemoryCandidate):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    extraction_run_id: UUID
    decision: MemoryCandidateDecision
    created_at: datetime
