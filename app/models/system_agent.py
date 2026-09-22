"""Pydantic models for reusable system-agent definitions."""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class SystemAgentTemplate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    key: str
    version: int
    role: str
    goal: str
    backstory: str | None = None
    enabled: bool
    allowed_skills: list[str]
    default_skills: list[str]
    created_at: datetime


class SystemAgentTemplateCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    key: str
    version: int = 1
    role: str
    goal: str
    backstory: str | None = None
    enabled: bool = True
    allowed_skills: list[str] = Field(default_factory=list)
    default_skills: list[str] = Field(default_factory=list)


class UserAgentOverride(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: int
    template_id: UUID
    custom_name: str | None = None
    custom_instructions: str | None = None
    enabled: bool
    memory_policy: str | None = None
    created_at: datetime
    updated_at: datetime


class UserAgentOverrideCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: int
    template_id: UUID
    custom_name: str | None = None
    custom_instructions: str | None = None
    enabled: bool = True
    memory_policy: str | None = None


class UserAgentOverrideUpdate(BaseModel):
    custom_name: str | None = None
    custom_instructions: str | None = None
    enabled: bool | None = None
    memory_policy: str | None = None


class RuntimeSystemAgent(BaseModel):
    """Resolved template plus a user's optional override; never persisted as Agent."""

    key: str
    version: int
    name: str
    role: str
    goal: str
    backstory: str | None = None
    custom_instructions: str | None = None
    allowed_skills: list[str]
    default_skills: list[str]
    memory_policy: str | None = None
    enabled: bool = True
