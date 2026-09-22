"""Application contracts for platform and user-owned skills."""

import json
import re
from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from models.permission import ActionClass, PermissionSubjectType


_SKILL_KEY_RE = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")


class SkillOwnerType(StrEnum):
    SYSTEM = "SYSTEM"
    USER = "USER"


class SkillStatus(StrEnum):
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    ARCHIVED = "ARCHIVED"


class SkillExecutionStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"
    DENIED = "DENIED"
    SANDBOX_ERROR = "SANDBOX_ERROR"


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
    package_checksum: str | None = None
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
    package_checksum: str | None = None
    manifest: dict[str, object] = Field(default_factory=dict)
    required_permissions: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("Skill name cannot be empty")
        return clean

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        clean = value.strip()
        if not _SKILL_KEY_RE.fullmatch(clean):
            raise ValueError("Skill key must match [a-z0-9][a-z0-9_-]*")
        return clean

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Skill version must be positive")
        return value

    @field_validator("required_permissions")
    @classmethod
    def validate_required_permissions(cls, value: list[str]) -> list[str]:
        return _validate_required_permissions(value)

    @model_validator(mode="after")
    def validate_definition(self) -> "SkillCreate":
        _validate_owner(self.owner_type, self.owner_user_id)
        _validate_json_object(self.manifest, "manifest")
        if self.package_checksum is not None and not re.fullmatch(r"[0-9a-f]{64}", self.package_checksum):
            raise ValueError("package_checksum must be a SHA-256 hex digest")
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


class SkillPackageFile(BaseModel):
    """One file in an uploadable skill package, addressed by a relative POSIX path."""

    path: str
    content: bytes


class PreparedSkillPackage(BaseModel):
    """Validated, deterministic package ready for immutable storage upload."""

    files: tuple[SkillPackageFile, ...]
    manifest: dict[str, object]
    checksum: str


class ExecutableSkillDefinition(BaseModel):
    """Immutable, backend-resolved execution metadata for one skill version."""

    skill_id: UUID
    owner_type: SkillOwnerType
    owner_user_id: int | None = None
    key: str
    version: int
    storage_uri: str
    package_checksum: str
    runtime: str
    entrypoint: str
    action_class: ActionClass = ActionClass.EXECUTE
    required_permissions: list[str] = Field(default_factory=list)


class MaterializedSkillPackage(BaseModel):
    """A verified package copied into a temporary host workspace for mounting."""

    skill_id: UUID
    key: str
    version: int
    root_path: str
    entrypoint: str
    checksum: str
    work_path: str | None = None
    output_path: str | None = None


class SandboxLimits(BaseModel):
    timeout_seconds: int = Field(gt=0)
    memory_mb: int = Field(gt=0)
    cpus: float = Field(gt=0)
    pids_limit: int = Field(gt=0)
    max_stdout_bytes: int = Field(gt=0)
    max_stderr_bytes: int = Field(gt=0)


class SkillExecutionRequest(BaseModel):
    run_id: UUID
    user_id: int
    agent_id: UUID | None = None
    skill_id: UUID
    skill_version: int | None = Field(default=None, gt=0)
    skill_key: str = ""
    requesting_subject_type: PermissionSubjectType = PermissionSubjectType.PERSISTENT_AGENT
    requesting_subject_id: str = ""
    arguments: dict[str, object] = Field(default_factory=dict)
    execution_id: UUID | None = None
    idempotency_key: str | None = None


class SkillExecutionResult(BaseModel):
    status: SkillExecutionStatus
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    output: object | None = None
    duration_ms: int = 0
    truncated: bool = False
    error: str | None = None


class SkillExecution(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    run_id: UUID
    user_id: int
    skill_id: UUID
    skill_version: int
    requesting_subject_type: PermissionSubjectType
    requesting_subject_id: str
    idempotency_key: str
    status: SkillExecutionStatus
    arguments_sanitized: dict[str, object]
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None
    exit_code: int | None = None
    stdout_preview: str = ""
    stderr_preview: str = ""
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class SkillExecutionCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    user_id: int
    skill_id: UUID
    skill_version: int = Field(gt=0)
    requesting_subject_type: PermissionSubjectType
    requesting_subject_id: str
    idempotency_key: str
    status: SkillExecutionStatus = SkillExecutionStatus.PENDING
    arguments_sanitized: dict[str, object] = Field(default_factory=dict)


class SkillPackage(BaseModel):
    """Transport contract for a skill package; validation happens in SkillPackageService."""

    skill_md: str | None = None
    manifest: dict[str, object] | None = None
    files: list[SkillPackageFile] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_manifest_json(self) -> "SkillPackage":
        if self.manifest is not None:
            _validate_json_object(self.manifest, "manifest")
        return self

    def canonical_files(self) -> list[SkillPackageFile]:
        """Return explicit files plus convenient top-level SKILL.md/manifest inputs."""
        files = list(self.files)
        if self.skill_md is not None:
            files.append(SkillPackageFile(path="SKILL.md", content=self.skill_md.encode("utf-8")))
        if self.manifest is not None:
            files.append(
                SkillPackageFile(
                    path="manifest.json",
                    content=(
                        json.dumps(self.manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    ).encode("utf-8"),
                )
            )
        return files


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


def _validate_required_permissions(value: list[str]) -> list[str]:
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("required_permissions must contain strings")
        action, separator, scope = item.strip().partition(":")
        if not separator or not scope.strip():
            raise ValueError("required_permissions must use action:resource_scope format")
        try:
            ActionClass(action)
        except ValueError as exc:
            raise ValueError(f"Unknown required permission action: {action}") from exc
        clean = f"{action}:{scope.strip()}"
        if clean not in normalized:
            normalized.append(clean)
    return normalized
