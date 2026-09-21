"""Pydantic contracts for user-owned scheduled agent runs."""

from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models.agent_run import AgentRunStatus
from models.runtime import AgentRuntimeStatus


class ScheduleType(StrEnum):
    ONCE = "ONCE"
    INTERVAL = "INTERVAL"
    CRON = "CRON"


class ScheduleStatus(StrEnum):
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    ARCHIVED = "ARCHIVED"


class Schedule(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: int
    agent_id: UUID
    chat_id: UUID | None = None
    name: str
    prompt: str
    schedule_type: ScheduleType
    schedule_expression: str
    timezone: str
    status: ScheduleStatus
    next_run_at: datetime | None = None
    last_run_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    claim_token: UUID | None = Field(default=None, exclude=True, repr=False)
    claim_expires_at: datetime | None = Field(default=None, exclude=True, repr=False)

    @field_validator("next_run_at", "last_run_at", "created_at", "updated_at", "claim_expires_at")
    @classmethod
    def normalize_database_timestamps(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class ScheduleCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: int
    agent_id: UUID
    chat_id: UUID | None = None
    name: str
    prompt: str
    schedule_type: ScheduleType
    schedule_expression: str
    timezone: str
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    next_run_at: datetime


class ScheduleUpdate(BaseModel):
    name: str | None = None
    prompt: str | None = None
    agent_id: UUID | None = None
    chat_id: UUID | None = None
    schedule_type: ScheduleType | None = None
    schedule_expression: str | None = None
    timezone: str | None = None


class ScheduledExecutionResult(BaseModel):
    schedule: Schedule
    run_id: UUID
    run_status: AgentRunStatus
    runtime_status: AgentRuntimeStatus
    content: str | None = None
    error: str | None = None
    approval_id: UUID | None = None
