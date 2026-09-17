from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class AgentPromptVersion(BaseModel):
    model_config = ConfigDict(from_attributes=True,)

    id: UUID
    agent_id: UUID

    version: int

    role: str
    goal: str

    backstory: str | None = None
    custom_instructions: str | None = None

    created_at: datetime


class AgentPromptVersionCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4,)

    agent_id: UUID

    version: int

    role: str
    goal: str

    backstory: str | None = None
    custom_instructions: str | None = None