from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
    )

    id: UUID
    chat_id: UUID
    role: str
    content: str
    created_at: datetime


class MessageCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    chat_id: UUID
    role: str
    content: str