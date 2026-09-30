"""Persistent user crews and resolved runtime task references."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class CrewProcess(StrEnum):
    SEQUENTIAL = "sequential"


class CrewStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class UserCrew(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: int
    name: str
    purpose: str
    status: CrewStatus
    process: CrewProcess
    config_version: int
    created_at: datetime
    updated_at: datetime


class UserCrewCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: int
    name: str
    purpose: str
    process: CrewProcess = CrewProcess.SEQUENTIAL


class CrewMember(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    crew_id: UUID
    agent_id: UUID
    position: int


class CrewMemberCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    crew_id: UUID
    agent_id: UUID
    position: int = Field(ge=0)


class CrewTask(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    crew_id: UUID
    agent_id: UUID
    position: int
    description: str
    expected_output: str


class CrewTaskCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    crew_id: UUID
    agent_id: UUID
    position: int = Field(ge=0)
    description: str
    expected_output: str


class CrewSkill(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    crew_id: UUID
    skill_id: UUID
    required: bool


class CrewSkillCreate(BaseModel):
    crew_id: UUID
    skill_id: UUID
    required: bool = True


class RuntimeCrewTask(BaseModel):
    agent_id: UUID
    description: str
    expected_output: str
