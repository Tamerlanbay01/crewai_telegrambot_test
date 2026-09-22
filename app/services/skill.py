"""Application rules for skill ownership, package lifecycle, and runtime resolution."""

import logging
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import config
from integrations.storage.factory import create_skill_storage
from integrations.storage.skill_storage import SkillStorage
from models.agent import Agent, AgentStatus
from models.permission import ActionClass, PermissionSubjectType
from models.runtime import RuntimeSkillDefinition
from models.skill import (
    AgentSkill,
    AgentSkillCreate,
    ExecutableSkillDefinition,
    PreparedSkillPackage,
    Skill,
    SkillCreate,
    SkillOwnerType,
    SkillPackage,
    SkillPackageFile,
    SkillStatus,
)
from repositories.agent import AgentRepository
from repositories.skill import SkillRepository
from services.skill_package import SkillPackageService, SkillPackageValidationError

logger = logging.getLogger(__name__)


class SkillPackagePersistenceError(RuntimeError):
    """Storage succeeded but PostgreSQL metadata could not be persisted."""


class SkillExecutionDeniedError(PermissionError):
    """The requested skill version is not executable by the active subject."""


class SkillService:
    """Owns authorization and metadata; package bytes stay behind SkillStorage."""

    def __init__(self, session: AsyncSession, *, storage: SkillStorage | None = None) -> None:
        self._session = session
        self._skills = SkillRepository(session)
        self._agents = AgentRepository(session)
        self._storage_adapter = storage
        self._packages = SkillPackageService(storage_provider=self._storage, config=config.s3)

    async def create_user_skill(
        self,
        *,
        user_id: int,
        key: str,
        name: str,
        description: str,
        version: int = 1,
        status: SkillStatus = SkillStatus.ACTIVE,
        storage_uri: str | None = None,
        manifest: dict[str, object] | None = None,
        required_permissions: list[str] | None = None,
        package: SkillPackage | None = None,
    ) -> Skill:
        if package is not None:
            if storage_uri is not None or manifest is not None:
                raise ValueError("Package-backed skills derive storage_uri and manifest from the package")
            return await self._create_package_backed_skill(
                owner_type=SkillOwnerType.USER,
                owner_user_id=user_id,
                key=key,
                name=name,
                description=description,
                version=version,
                required_permissions=required_permissions or [],
                package=package,
            )
        return await self._create_metadata_skill(
            owner_type=SkillOwnerType.USER,
            owner_user_id=user_id,
            key=key,
            name=name,
            description=description,
            version=version,
            status=status,
            storage_uri=storage_uri,
            manifest=manifest or {},
            required_permissions=required_permissions or [],
        )

    async def upload_user_skill(
        self, *, user_id: int, key: str, name: str, description: str, package: SkillPackage,
        version: int = 1, required_permissions: list[str] | None = None,
    ) -> Skill:
        """Upload first, then create ACTIVE user-owned metadata on success."""
        return await self.create_user_skill(
            user_id=user_id,
            key=key,
            name=name,
            description=description,
            version=version,
            required_permissions=required_permissions,
            package=package,
        )

    async def create_system_skill(
        self,
        *,
        key: str,
        name: str,
        description: str,
        version: int = 1,
        status: SkillStatus = SkillStatus.ACTIVE,
        storage_uri: str | None = None,
        manifest: dict[str, object] | None = None,
        required_permissions: list[str] | None = None,
        package: SkillPackage | None = None,
    ) -> Skill:
        """Administrative-only entry point for platform-owned skills."""
        if package is not None:
            if storage_uri is not None or manifest is not None:
                raise ValueError("Package-backed skills derive storage_uri and manifest from the package")
            return await self._create_package_backed_skill(
                owner_type=SkillOwnerType.SYSTEM,
                owner_user_id=None,
                key=key,
                name=name,
                description=description,
                version=version,
                required_permissions=required_permissions or [],
                package=package,
            )
        return await self._create_metadata_skill(
            owner_type=SkillOwnerType.SYSTEM,
            owner_user_id=None,
            key=key,
            name=name,
            description=description,
            version=version,
            status=status,
            storage_uri=storage_uri,
            manifest=manifest or {},
            required_permissions=required_permissions or [],
        )

    async def get_skill(self, *, user_id: int, skill_id: UUID) -> Skill:
        skill = await self._get_accessible_skill(user_id=user_id, skill_id=skill_id)
        if skill.owner_type == SkillOwnerType.SYSTEM and skill.status != SkillStatus.ACTIVE:
            raise LookupError(f"Skill not found: {skill_id}")
        return skill

    async def list_available_skills(self, *, user_id: int) -> list[Skill]:
        return [
            *await self._skills.list_system(active_only=True),
            *await self._skills.list_by_owner(owner_user_id=user_id, status=SkillStatus.ACTIVE),
        ]

    async def assign_to_agent(self, *, user_id: int, agent_id: UUID, skill_id: UUID) -> AgentSkill:
        await self._require_agent(user_id=user_id, agent_id=agent_id, active_only=True)
        skill = await self._get_accessible_skill(user_id=user_id, skill_id=skill_id)
        if skill.status != SkillStatus.ACTIVE:
            raise ValueError("Cannot assign an inactive skill")
        assignment = await self._skills.assign(
            AgentSkillCreate(agent_id=agent_id, skill_id=skill_id, enabled=True)
        )
        await self._session.commit()
        return assignment

    async def remove_from_agent(self, *, user_id: int, agent_id: UUID, skill_id: UUID) -> bool:
        await self._require_agent(user_id=user_id, agent_id=agent_id)
        await self._get_accessible_skill(user_id=user_id, skill_id=skill_id)
        removed = await self._skills.remove(agent_id=agent_id, skill_id=skill_id)
        if removed:
            await self._session.commit()
        return removed

    async def list_agent_skills(self, *, user_id: int, agent_id: UUID) -> list[Skill]:
        await self._require_agent(user_id=user_id, agent_id=agent_id)
        skills: list[Skill] = []
        for assignment in await self._skills.list_agent_skills(agent_id):
            if not assignment.enabled:
                continue
            skill = await self._skills.get_by_id(assignment.skill_id)
            if skill is not None and skill.status == SkillStatus.ACTIVE and self._is_accessible_to_user(skill, user_id):
                skills.append(skill)
        return skills

    async def archive_skill(self, *, user_id: int, skill_id: UUID) -> Skill:
        skill = await self._skills.get_by_id(skill_id)
        if skill is None or skill.owner_type != SkillOwnerType.USER or skill.owner_user_id != user_id:
            raise LookupError(f"Skill not found: {skill_id}")
        archived = await self._skills.update_status(skill_id, SkillStatus.ARCHIVED)
        if archived is None:
            raise LookupError(f"Skill not found: {skill_id}")
        await self._session.commit()
        # Deliberately retain the package for audit, reproducibility, and rollback.
        return archived

    async def list_package_files(self, *, user_id: int, skill_id: UUID) -> list[str]:
        skill = await self._get_accessible_skill(user_id=user_id, skill_id=skill_id)
        return await self._storage().list_files(prefix=self._storage_prefix_for(skill))

    async def get_package_file(self, *, user_id: int, skill_id: UUID, path: str) -> bytes:
        skill = await self._get_accessible_skill(user_id=user_id, skill_id=skill_id)
        self._packages.validate_path(path)
        return await self._storage().get_file(
            prefix=self._storage_prefix_for(skill), path=path, max_bytes=config.s3.max_skill_file_bytes
        )

    async def verify_skill_storage(self, *, user_id: int, skill_id: UUID) -> bool:
        skill = await self._get_accessible_skill(user_id=user_id, skill_id=skill_id)
        checksum = await self._storage().get_package_checksum(prefix=self._storage_prefix_for(skill))
        return checksum is not None and checksum == skill.package_checksum

    async def delete_skill_version_storage(self, *, skill_id: UUID) -> None:
        """Privileged cleanup operation. Call only from an authenticated admin workflow."""
        skill = await self._skills.get_by_id(skill_id)
        if skill is None:
            raise LookupError(f"Skill not found: {skill_id}")
        if skill.status != SkillStatus.ARCHIVED:
            raise ValueError("Only archived skill packages may be physically deleted")
        await self._storage().delete_version(prefix=self._storage_prefix_for(skill))

    async def resolve_agent_skills(self, *, user_id: int, agent_id: UUID) -> list[RuntimeSkillDefinition]:
        resolved: list[RuntimeSkillDefinition] = []
        for skill in await self.list_agent_skills(user_id=user_id, agent_id=agent_id):
            runtime = await self._resolve_runtime_skill(skill)
            if runtime is not None:
                resolved.append(runtime)
        return resolved

    async def resolve_system_skills(
        self, *, allowed_keys: list[str], default_keys: list[str]
    ) -> list[RuntimeSkillDefinition]:
        allowed = set(allowed_keys)
        resolved: list[RuntimeSkillDefinition] = []
        for key in dict.fromkeys(default_keys):
            if key not in allowed:
                continue
            skill = await self._skills.get_by_key(key=key, owner_type=SkillOwnerType.SYSTEM)
            if skill is None or skill.status != SkillStatus.ACTIVE:
                continue
            runtime = await self._resolve_runtime_skill(skill)
            if runtime is not None:
                resolved.append(runtime)
        return resolved

    async def resolve_executable_skill(
        self,
        *,
        user_id: int,
        requesting_subject_type: PermissionSubjectType = PermissionSubjectType.PERSISTENT_AGENT,
        requesting_subject_id: str | None = None,
        skill_key: str = "",
        skill_version: int | None = None,
        skill_id: UUID | None = None,
        runtime_skill_catalog: Sequence[RuntimeSkillDefinition] = (),
        agent_id: UUID | None = None,
        version: int | None = None,
    ) -> ExecutableSkillDefinition:
        """Resolve exactly one active, assigned and package-backed skill version.

        The runtime catalog is the pin produced while the AgentRun context was
        built.  It is intentionally required for system and temporary agents;
        those subjects must never turn a tool call into a "latest by key" lookup.
        Persistent agents may additionally be resolved from an explicit immutable
        skill id/version supplied by the authority.
        """

        if requesting_subject_id is None and agent_id is not None:
            requesting_subject_id = str(agent_id)
        if skill_version is None:
            skill_version = version
        referenced = await self._skills.get_by_id(skill_id) if skill_id is not None else None
        clean_key = skill_key.strip() or (referenced.key if referenced is not None else "")
        if skill_version is None and referenced is not None:
            # An immutable skill id already pins its version; this is not a
            # latest-by-key lookup.
            skill_version = referenced.version
        if not clean_key:
            raise SkillExecutionDeniedError("Skill key is required")
        if requesting_subject_id is None:
            raise SkillExecutionDeniedError("Skill execution subject is required")
        candidates = [
            item
            for item in runtime_skill_catalog
            if item.key == clean_key
            and (skill_version is None or item.version == skill_version)
            and (skill_id is None or item.id == skill_id)
        ]
        if len(candidates) > 1:
            raise SkillExecutionDeniedError("Skill version is ambiguous for this run")
        candidate = candidates[0] if candidates else None

        if candidate is None and requesting_subject_type == PermissionSubjectType.PERSISTENT_AGENT:
            if skill_id is None or skill_version is None:
                raise SkillExecutionDeniedError("Skill version is not pinned for this run")
            candidate = RuntimeSkillDefinition(
                id=skill_id,
                key=clean_key,
                name=clean_key,
                description="",
                version=skill_version,
            )
        if candidate is None or candidate.id is None:
            raise SkillExecutionDeniedError("Skill is not active for this agent")

        skill = await self._skills.get_by_id(candidate.id)
        if skill is None or skill.status != SkillStatus.ACTIVE:
            raise SkillExecutionDeniedError("Skill is not active for this agent")
        if skill.key != clean_key or skill.version != candidate.version:
            raise SkillExecutionDeniedError("Skill version pin does not match backend metadata")
        if skill_id is not None and skill.id != skill_id:
            raise SkillExecutionDeniedError("Skill id does not match the pinned runtime skill")

        if requesting_subject_type == PermissionSubjectType.PERSISTENT_AGENT:
            try:
                agent_id = UUID(requesting_subject_id)
            except ValueError as exc:
                raise SkillExecutionDeniedError("Persistent agent identity is invalid") from exc
            assigned = await self.list_agent_skills(user_id=user_id, agent_id=agent_id)
            if not any(item.id == skill.id and item.version == skill.version for item in assigned):
                raise SkillExecutionDeniedError("Skill is not assigned to this agent")
        elif requesting_subject_type == PermissionSubjectType.SYSTEM_AGENT:
            if skill.owner_type != SkillOwnerType.SYSTEM:
                raise SkillExecutionDeniedError("System agents may use only system skills")
        elif requesting_subject_type == PermissionSubjectType.TEMPORARY_SUBAGENT:
            if not self._is_accessible_to_user(skill, user_id):
                raise SkillExecutionDeniedError("Skill is outside the inherited runtime scope")
        else:  # pragma: no cover - PermissionSubjectType is exhaustive
            raise SkillExecutionDeniedError("Unsupported skill execution subject")

        if skill.storage_uri is None or skill.package_checksum is None:
            raise SkillExecutionDeniedError("This skill is instructional and cannot be executed")
        entrypoint = skill.manifest.get("entrypoint")
        runtime = skill.manifest.get("runtime")
        try:
            self._packages.validate_manifest(
                skill.manifest,
                key=skill.key,
                version=skill.version,
                package_files=None,
            )
        except SkillPackageValidationError as exc:
            # The package upload path validates this already. Revalidate the
            # persisted JSON before allowing an executable path to proceed.
            raise SkillExecutionDeniedError("Skill manifest is not executable") from exc
        if not isinstance(entrypoint, str) or not entrypoint:
            raise SkillExecutionDeniedError("Skill has no executable entrypoint")
        if runtime != "python":
            raise SkillExecutionDeniedError("Only Python skills can be executed")

        execution_action = (
            ActionClass.EXTERNAL_SIDE_EFFECT
            if skill.manifest.get("external_side_effect") is True
            else ActionClass.EXECUTE
        )
        return ExecutableSkillDefinition(
            skill_id=skill.id,
            owner_type=skill.owner_type,
            owner_user_id=skill.owner_user_id,
            key=skill.key,
            version=skill.version,
            storage_uri=skill.storage_uri,
            package_checksum=skill.package_checksum,
            runtime=runtime,
            entrypoint=entrypoint,
            action_class=execution_action,
            required_permissions=list(skill.required_permissions),
        )

    async def get_verified_package(
        self, definition: ExecutableSkillDefinition
    ) -> PreparedSkillPackage:
        """Download and verify the exact immutable package selected by the run."""

        skill = await self._skills.get_by_id(definition.skill_id)
        if skill is None or skill.status != SkillStatus.ACTIVE:
            raise SkillPackageValidationError("Skill is no longer active")
        if (
            skill.key != definition.key
            or skill.version != definition.version
            or skill.package_checksum != definition.package_checksum
        ):
            raise SkillPackageValidationError("Skill metadata changed after resolution")
        prefix = self._storage_prefix_for(skill)
        stored_checksum = await self._storage().get_package_checksum(prefix=prefix)
        if stored_checksum != skill.package_checksum:
            raise SkillPackageValidationError("Skill package checksum mismatch")
        package_files: list[SkillPackageFile] = []
        for path in await self._storage().list_files(prefix=prefix):
            self._packages.validate_path(path)
            package_files.append(
                SkillPackageFile(
                    path=path,
                    content=await self._storage().get_file(
                        prefix=prefix,
                        path=path,
                        max_bytes=config.s3.max_skill_file_bytes,
                    ),
                )
            )
        prepared = self._packages.validate(
            package=SkillPackage(files=package_files),
            key=skill.key,
            version=skill.version,
        )
        if prepared.checksum != skill.package_checksum:
            raise SkillPackageValidationError("Skill package checksum mismatch")
        if prepared.manifest != skill.manifest:
            raise SkillPackageValidationError("Skill manifest does not match stored metadata")
        if prepared.manifest.get("entrypoint") != definition.entrypoint:
            raise SkillPackageValidationError("Skill entrypoint does not match stored metadata")
        return prepared

    async def _create_metadata_skill(self, **values: object) -> Skill:
        skill = await self._skills.create(SkillCreate(**values))
        await self._session.commit()
        return skill

    async def _create_package_backed_skill(
        self,
        *,
        owner_type: SkillOwnerType,
        owner_user_id: int | None,
        key: str,
        name: str,
        description: str,
        version: int,
        required_permissions: list[str],
        package: SkillPackage,
    ) -> Skill:
        prepared = self._packages.validate(package=package, key=key, version=version)
        existing = await self._skills.get_by_key(
            key=key, owner_type=owner_type, owner_user_id=owner_user_id, version=version
        )
        if existing is not None and existing.package_checksum != prepared.checksum:
            raise ValueError("Skill key and version are immutable; create a new version")

        prefix = self._packages.object_prefix(
            owner_type=owner_type, owner_user_id=owner_user_id, key=key, version=version
        )
        await self._packages.upload(prefix=prefix, package=prepared)
        if existing is not None:
            return existing

        try:
            skill = await self._skills.create(
                SkillCreate(
                    owner_type=owner_type,
                    owner_user_id=owner_user_id,
                    key=key,
                    name=name,
                    description=description,
                    version=version,
                    status=SkillStatus.ACTIVE,
                    storage_uri=self._packages.storage_uri(prefix=prefix),
                    package_checksum=prepared.checksum,
                    manifest=prepared.manifest,
                    required_permissions=required_permissions,
                )
            )
            await self._session.commit()
            return skill
        except Exception as exc:
            await self._session.rollback()
            logger.error(
                "Skill metadata persistence failed after package upload; orphan may remain at %s",
                self._packages.storage_uri(prefix=prefix),
            )
            raise SkillPackagePersistenceError(
                "Skill package uploaded but metadata persistence failed; inspect orphaned storage"
            ) from exc

    async def _resolve_runtime_skill(self, skill: Skill) -> RuntimeSkillDefinition | None:
        if skill.storage_uri is None:
            return self._runtime_definition(skill)
        try:
            instructions = (
                await self._storage().get_file(
                    prefix=self._storage_prefix_for(skill),
                    path="SKILL.md",
                    max_bytes=config.s3.max_skill_md_bytes,
                )
            ).decode("utf-8")
        except Exception as exc:
            logger.warning("Excluding unavailable active skill %s@%s: %s", skill.key, skill.version, exc)
            return None
        return self._runtime_definition(skill, instructions=instructions)

    async def _get_accessible_skill(self, *, user_id: int, skill_id: UUID) -> Skill:
        skill = await self._skills.get_by_id(skill_id)
        if skill is None or not self._is_accessible_to_user(skill, user_id):
            raise LookupError(f"Skill not found: {skill_id}")
        return skill

    async def _require_agent(self, *, user_id: int, agent_id: UUID, active_only: bool = False) -> Agent:
        agent = await self._agents.get_by_id(agent_id)
        if agent is None or agent.user_id != user_id:
            raise LookupError(f"Agent not found: {agent_id}")
        if active_only and agent.status != AgentStatus.ACTIVE:
            raise ValueError("Cannot assign skills to an inactive agent")
        return agent

    def _storage(self) -> SkillStorage:
        if self._storage_adapter is None:
            self._storage_adapter = create_skill_storage(config.s3)
        return self._storage_adapter

    def _storage_prefix_for(self, skill: Skill) -> str:
        prefix = self._packages.object_prefix(
            owner_type=skill.owner_type,
            owner_user_id=skill.owner_user_id,
            key=skill.key,
            version=skill.version,
        )
        if skill.storage_uri != self._packages.storage_uri(prefix=prefix):
            raise ValueError("Skill storage URI is not a canonical backend-owned package location")
        return prefix

    @staticmethod
    def _is_accessible_to_user(skill: Skill, user_id: int) -> bool:
        return skill.owner_type == SkillOwnerType.SYSTEM or skill.owner_user_id == user_id

    @staticmethod
    def _runtime_definition(skill: Skill, *, instructions: str | None = None) -> RuntimeSkillDefinition:
        if instructions is None:
            raw = skill.manifest.get("instructions")
            instructions = raw if isinstance(raw, str) else None
        raw_constraints = skill.manifest.get("constraints", [])
        if isinstance(raw_constraints, str):
            constraints = [raw_constraints]
        elif isinstance(raw_constraints, list):
            constraints = [item for item in raw_constraints if isinstance(item, str)]
        else:
            constraints = []
        return RuntimeSkillDefinition(
            id=skill.id,
            key=skill.key,
            name=skill.name,
            description=skill.description,
            version=skill.version,
            required_permissions=list(skill.required_permissions),
            instructions=instructions,
            constraints=constraints,
            storage_uri=skill.storage_uri,
            package_checksum=skill.package_checksum,
            runtime=(
                skill.manifest.get("runtime")
                if isinstance(skill.manifest.get("runtime"), str)
                else None
            ),
            entrypoint=(
                skill.manifest.get("entrypoint")
                if isinstance(skill.manifest.get("entrypoint"), str)
                else None
            ),
            action_class=(
                ActionClass.EXTERNAL_SIDE_EFFECT
                if skill.manifest.get("external_side_effect") is True
                else ActionClass.EXECUTE
            ),
        )
