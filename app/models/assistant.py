"""Application response for the primary assistant message flow."""

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel

from models.message import Message


class AssistantResponseStatus(StrEnum):
    COMPLETED = "completed"
    WAITING_APPROVAL = "waiting_approval"
    FAILED = "failed"


class AssistantResponse(BaseModel):
    status: AssistantResponseStatus
    run_id: UUID
    message: Message | None = None
    content: str | None = None
    approval_id: UUID | None = None
    approval_summary: str | None = None
    error: str | None = None
