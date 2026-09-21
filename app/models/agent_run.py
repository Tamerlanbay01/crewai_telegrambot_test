"""Pydantic models for persisted agent-runtime executions."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class AgentRunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentRunEventType(StrEnum):
    CREATED = "created"
    STATUS_CHANGED = "status_changed"
    RUNTIME = "runtime"
    RUNTIME_STARTED = "runtime_started"
    DELEGATION_REQUESTED = "delegation_requested"
    DELEGATION_COMPLETED = "delegation_completed"
    DELEGATION_FAILED = "delegation_failed"
    TEMPORARY_SUBAGENT_CREATED = "temporary_subagent_created"
    TOOL_REQUESTED = "tool_requested"
    PERMISSION_DENIED = "permission_denied"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_APPROVED = "approval_approved"
    APPROVAL_REJECTED = "approval_rejected"
    APPROVAL_EXPIRED = "approval_expired"
    APPROVAL_CANCELLED = "approval_cancelled"
    TOOL_EXECUTED = "tool_executed"
    BUDGET_EXCEEDED = "budget_exceeded"
    RUNTIME_COMPLETED = "runtime_completed"
    RUNTIME_FAILED = "runtime_failed"


class AgentRun(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: int
    starting_agent_id: UUID
    chat_id: UUID | None = None
    message_id: UUID | None = None
    prompt_version_id: UUID
    model_name: str
    status: AgentRunStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result_metadata: dict[str, object]
    usage: dict[str, object]
    checkpoint: dict[str, object]
    created_at: datetime
    updated_at: datetime


class AgentRunCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: int
    starting_agent_id: UUID
    prompt_version_id: UUID
    model_name: str
    chat_id: UUID | None = None
    message_id: UUID | None = None
    result_metadata: dict[str, object] = Field(default_factory=dict)
    usage: dict[str, object] = Field(default_factory=dict)
    checkpoint: dict[str, object] = Field(default_factory=dict)


class AgentRunEvent(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    run_id: UUID
    event_type: AgentRunEventType
    from_status: AgentRunStatus | None = None
    to_status: AgentRunStatus | None = None
    payload: dict[str, object]
    created_at: datetime


class AgentRunEventCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    event_type: AgentRunEventType
    from_status: AgentRunStatus | None = None
    to_status: AgentRunStatus | None = None
    payload: dict[str, object] = Field(default_factory=dict)
