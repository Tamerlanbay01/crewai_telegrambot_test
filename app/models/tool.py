"""Serializable contracts crossing the CrewAI/backend tool boundary."""

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

from models.permission import ActionClass, PermissionSubjectType


class ToolExecutionStatus(StrEnum):
    EXECUTED = "executed"
    WAITING_APPROVAL = "waiting_approval"
    DENIED = "denied"
    FAILED = "failed"
    REJECTED = "rejected"


class ToolRequest(BaseModel):
    run_id: UUID
    user_id: int
    requesting_subject_type: PermissionSubjectType
    requesting_subject_id: str
    name: str
    action_class: ActionClass
    resource: str
    arguments: dict[str, object] = Field(default_factory=dict)


class ToolIntent(BaseModel):
    name: str
    action_class: ActionClass
    resource: str
    arguments: dict[str, object] = Field(default_factory=dict)


class ToolExecutionResult(BaseModel):
    status: ToolExecutionStatus
    output: object | None = None
    error: str | None = None
    approval_id: UUID | None = None
    flow_id: str | None = None


class ToolDefinition(BaseModel):
    name: str
    action_class: ActionClass
    resource_prefix: str
