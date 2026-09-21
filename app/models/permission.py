"""Serializable permission contracts owned by the backend."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ActionClass(StrEnum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"


class PermissionSubjectType(StrEnum):
    PERSISTENT_AGENT = "persistent_agent"
    SYSTEM_AGENT = "system_agent"
    TEMPORARY_SUBAGENT = "temporary_subagent"


class Permission(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: int
    subject_type: PermissionSubjectType
    subject_id: str
    action_class: ActionClass
    resource_scope: str
    allowed: bool
    created_at: datetime
    updated_at: datetime
    revoked_at: datetime | None = None


class PermissionCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: int
    subject_type: PermissionSubjectType
    subject_id: str
    action_class: ActionClass
    resource_scope: str
    allowed: bool = True
    revoked_at: datetime | None = None
