"""Serializable approval records for one concrete backend action."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from models.permission import ActionClass, PermissionSubjectType


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class Approval(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: int
    run_id: UUID
    flow_id: str | None = None
    requesting_subject_type: PermissionSubjectType
    requesting_subject_id: str
    action_name: str
    action_class: ActionClass
    resource: str
    arguments: dict[str, object]
    status: ApprovalStatus
    requested_at: datetime
    expires_at: datetime | None = None
    decided_at: datetime | None = None
    decision_metadata: dict[str, object]


class ApprovalCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: int
    run_id: UUID
    flow_id: str | None = None
    requesting_subject_type: PermissionSubjectType
    requesting_subject_id: str
    action_name: str
    action_class: ActionClass
    resource: str
    arguments: dict[str, object] = Field(default_factory=dict)
    expires_at: datetime | None = None
