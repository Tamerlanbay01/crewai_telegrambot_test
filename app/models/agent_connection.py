"""Pydantic models for agent connections."""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class AgentConnection(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    parent_agent_id: UUID
    child_agent_id: UUID
    created_at: datetime


class AgentConnectionCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    parent_agent_id: UUID
    child_agent_id: UUID
