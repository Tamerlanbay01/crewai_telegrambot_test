"""Declarative proposals from the AI design crew; they grant no authority."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from models.agent import AgentKind
from models.crew import CrewProcess, RuntimeCrewTask
from models.memory import MemoryScope
from models.permission import ActionClass
from models.runtime import RuntimeAgentDefinition, RuntimeBudgets


class RequestedPermission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_class: ActionClass
    resource_scope: str = Field(min_length=1, max_length=255)

    @field_validator("resource_scope")
    @classmethod
    def strip_scope(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("Permission resource scope cannot be blank")
        return clean


class AvailableSkill(BaseModel):
    id: UUID
    key: str
    name: str
    description: str
    required_permissions: list[str] = Field(default_factory=list)


class AvailableAgent(BaseModel):
    id: UUID
    name: str
    role: str = ""
    goal: str = ""
    skill_ids: list[UUID] = Field(default_factory=list)


class AgentFactoryInput(BaseModel):
    user_request: str = Field(min_length=1, max_length=4000)
    available_skills: list[AvailableSkill] = Field(default_factory=list)
    available_agents: list[AvailableAgent] = Field(default_factory=list)
    parent_permissions: list[RequestedPermission] = Field(default_factory=list)
    parent_can_spawn_subagents: bool
    limits: RuntimeBudgets = Field(default_factory=RuntimeBudgets)


class AgentBlueprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[AgentKind.USER] = AgentKind.USER
    name: str = Field(min_length=1, max_length=255)
    role: str = Field(min_length=1, max_length=2000)
    goal: str = Field(min_length=1, max_length=4000)
    backstory: str = Field(default="", max_length=4000)
    custom_instructions: str | None = Field(default=None, max_length=4000)
    skill_ids: list[UUID] = Field(default_factory=list)
    missing_capabilities: list[str] = Field(default_factory=list)
    connected_agent_ids: list[UUID] = Field(default_factory=list)
    can_spawn_subagents: bool = False
    memory_scopes: list[MemoryScope] = Field(default_factory=list)
    requested_permissions: list[RequestedPermission] = Field(default_factory=list)
    budget_profile: RuntimeBudgets | None = None

    @field_validator("name", "role", "goal")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("Agent identity and purpose cannot be blank")
        return clean


class CrewTaskBlueprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, max_length=4000)
    expected_output: str = Field(min_length=1, max_length=2000)
    agent_index: int = Field(ge=0)


class CrewBlueprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    purpose: str = Field(min_length=1, max_length=4000)
    agents: list[AgentBlueprint] = Field(default_factory=list)
    existing_agent_ids: list[UUID] = Field(default_factory=list)
    process: CrewProcess = CrewProcess.SEQUENTIAL
    tasks: list[CrewTaskBlueprint] = Field(min_length=1)
    required_skill_ids: list[UUID] = Field(default_factory=list)
    missing_capabilities: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_agents(self) -> "CrewBlueprint":
        if not self.agents and not self.existing_agent_ids:
            raise ValueError("Crew needs at least one agent")
        return self

    @field_validator("name", "purpose")
    @classmethod
    def strip_crew_text(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("Crew name and purpose cannot be blank")
        return clean


class CrewDefinition(BaseModel):
    """Backend resolved runtime input, never raw AI output."""

    id: UUID | None = None
    user_id: int | None = None
    name: str
    purpose: str = ""
    config_version: int = 1
    agents: list[RuntimeAgentDefinition] = Field(min_length=1)
    tasks: list[RuntimeCrewTask] = Field(min_length=1)
    required_skill_ids: list[UUID] = Field(default_factory=list)
    process: CrewProcess = CrewProcess.SEQUENTIAL
    budgets: RuntimeBudgets = Field(default_factory=RuntimeBudgets)
