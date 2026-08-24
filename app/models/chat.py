from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class Chat(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
    )

    id: UUID
    user_id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ChatCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: int
    title: str = "Новый чат"


class ChatUpdate(BaseModel):
    title: str | None = None