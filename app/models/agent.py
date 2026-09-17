"""Pydantic models for agents."""
from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

class AgentKind(StrEnum):
    PRIMARY = "primary"
    USER = "user"
    
class AgentStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"
    ARCHIVED = "archived"
    
class Agent(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    user_id: int
    kind: AgentKind
    status: AgentStatus
    name: str
    can_spawn_subagents: bool
    created_at: datetime
    updated_at: datetime
    
class AgentCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: int
    kind: AgentKind = AgentKind.USER
    name: str
    can_spawn_subagents: bool = False
    
class AgentUpdate(BaseModel):
    name: str | None = None
    status: AgentStatus | None = None
    can_spawn_subagents: bool | None = None
