"""Application contracts for platform and user-owned skills."""

import json
from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SkillOwnerType(StrEnum):
    SYSTEM = "SYSTEM"
    USER = "USER"


class SkillStatus(StrEnum):
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    ARCHIVED = "ARCHIVED"


class Skill(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    owner_type: SkillOwnerType
    owner_user_id: int | None = None
    key: str
    name: str
    description: str
    version: int
    status: SkillStatus
    storage_uri: str | None = None
    manifest: dict[str, object] = Field(default_factory=dict)
    required_permissions: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_owner(self) -> "Skill":
        _validate_owner(self.owner_type, self.owner_user_id)
        return self


class SkillCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    owner_type: SkillOwnerType
    owner_user_id: int | None = None
    key: str
    name: str
    description: str
    version: int = 1
    status: SkillStatus = SkillStatus.ACTIVE
    storage_uri: str | None = None
    manifest: dict[str, object] = Field(default_factory=dict)
    required_permissions: list[str] = Field(default_factory=list)

    @field_validator("key", "name")
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("Skill key and name cannot be empty")
        return clean

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Skill version must be positive")
        return value

    @model_validator(mode="after")
    def validate_definition(self) -> "SkillCreate":
        _validate_owner(self.owner_type, self.owner_user_id)
        _validate_json_object(self.manifest, "manifest")
        return self


class AgentSkill(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    agent_id: UUID
    skill_id: UUID
    enabled: bool
    created_at: datetime


class AgentSkillCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    agent_id: UUID
    skill_id: UUID
    enabled: bool = True


def _validate_owner(owner_type: SkillOwnerType, owner_user_id: int | None) -> None:
    if owner_type == SkillOwnerType.USER and owner_user_id is None:
        raise ValueError("User skills require owner_user_id")
    if owner_type == SkillOwnerType.SYSTEM and owner_user_id is not None:
        raise ValueError("System skills cannot have owner_user_id")


def _validate_json_object(value: dict[str, object], field: str) -> None:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain JSON-compatible values") from exc
