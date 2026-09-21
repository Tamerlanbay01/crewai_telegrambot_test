"""Application rules for skill ownership, assignment, and runtime resolution."""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from models.agent import Agent, AgentStatus
from models.runtime import RuntimeSkillDefinition
from models.skill import (
    AgentSkill,
    AgentSkillCreate,
    Skill,
    SkillCreate,
    SkillOwnerType,
    SkillStatus,
)
from repositories.agent import AgentRepository
from repositories.skill import SkillRepository


class SkillService:
    def __init__(self, session: AsyncSession):
        self._session = session
        self._skills = SkillRepository(session)
        self._agents = AgentRepository(session)

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
    ) -> Skill:
        skill = await self._skills.create(
            SkillCreate(
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
        )
        await self._session.commit()
        return skill

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
    ) -> Skill:
        """Create platform-owned metadata; this is an administrative operation."""
        skill = await self._skills.create(
            SkillCreate(
                owner_type=SkillOwnerType.SYSTEM,
                key=key,
                name=name,
                description=description,
                version=version,
                status=status,
                storage_uri=storage_uri,
                manifest=manifest or {},
                required_permissions=required_permissions or [],
            )
        )
        await self._session.commit()
        return skill

    async def get_skill(self, *, user_id: int, skill_id: UUID) -> Skill:
        skill = await self._get_accessible_skill(user_id=user_id, skill_id=skill_id)
        if skill.owner_type == SkillOwnerType.SYSTEM and skill.status != SkillStatus.ACTIVE:
            raise LookupError(f"Skill not found: {skill_id}")
        return skill

    async def list_available_skills(self, *, user_id: int) -> list[Skill]:
        system_skills = await self._skills.list_system(active_only=True)
        user_skills = await self._skills.list_by_owner(
            owner_user_id=user_id,
            status=SkillStatus.ACTIVE,
        )
        return [*system_skills, *user_skills]

    async def assign_to_agent(
        self, *, user_id: int, agent_id: UUID, skill_id: UUID
    ) -> AgentSkill:
        await self._require_agent(user_id=user_id, agent_id=agent_id, active_only=True)
        skill = await self._get_accessible_skill(user_id=user_id, skill_id=skill_id)
        if skill.status != SkillStatus.ACTIVE:
            raise ValueError("Cannot assign an inactive skill")
        assignment = await self._skills.assign(
            AgentSkillCreate(agent_id=agent_id, skill_id=skill_id, enabled=True)
        )
        await self._session.commit()
        return assignment

    async def remove_from_agent(
        self, *, user_id: int, agent_id: UUID, skill_id: UUID
    ) -> bool:
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
            if skill is None or skill.status != SkillStatus.ACTIVE:
                continue
            if not self._is_accessible_to_user(skill, user_id):
                continue
            skills.append(skill)
        return skills

    async def archive_skill(self, *, user_id: int, skill_id: UUID) -> Skill:
        skill = await self._skills.get_by_id(skill_id)
        if (
            skill is None
            or skill.owner_type != SkillOwnerType.USER
            or skill.owner_user_id != user_id
        ):
            raise LookupError(f"Skill not found: {skill_id}")
        archived = await self._skills.update_status(skill_id, SkillStatus.ARCHIVED)
        if archived is None:
            raise LookupError(f"Skill not found: {skill_id}")
        await self._session.commit()
        return archived

    async def resolve_agent_skills(
        self, *, user_id: int, agent_id: UUID
    ) -> list[RuntimeSkillDefinition]:
        return [
            self._runtime_definition(skill)
            for skill in await self.list_agent_skills(user_id=user_id, agent_id=agent_id)
        ]

    async def resolve_system_skills(
        self, *, allowed_keys: list[str], default_keys: list[str]
    ) -> list[RuntimeSkillDefinition]:
        allowed = set(allowed_keys)
        resolved: list[RuntimeSkillDefinition] = []
        for key in dict.fromkeys(default_keys):
            if key not in allowed:
                continue
            skill = await self._skills.get_by_key(
                key=key,
                owner_type=SkillOwnerType.SYSTEM,
            )
            if skill is None or skill.status != SkillStatus.ACTIVE:
                continue
            resolved.append(self._runtime_definition(skill))
        return resolved

    async def _get_accessible_skill(self, *, user_id: int, skill_id: UUID) -> Skill:
        skill = await self._skills.get_by_id(skill_id)
        if skill is None or not self._is_accessible_to_user(skill, user_id):
            raise LookupError(f"Skill not found: {skill_id}")
        return skill

    async def _require_agent(
        self, *, user_id: int, agent_id: UUID, active_only: bool = False
    ) -> Agent:
        agent = await self._agents.get_by_id(agent_id)
        if agent is None or agent.user_id != user_id:
            raise LookupError(f"Agent not found: {agent_id}")
        if active_only and agent.status != AgentStatus.ACTIVE:
            raise ValueError("Cannot assign skills to an inactive agent")
        return agent

    @staticmethod
    def _is_accessible_to_user(skill: Skill, user_id: int) -> bool:
        if skill.owner_type == SkillOwnerType.SYSTEM:
            return skill.owner_user_id is None
        return skill.owner_user_id == user_id

    @staticmethod
    def _runtime_definition(skill: Skill) -> RuntimeSkillDefinition:
        raw_instructions = skill.manifest.get("instructions")
        instructions = raw_instructions if isinstance(raw_instructions, str) else None
        raw_constraints = skill.manifest.get("constraints", [])
        if isinstance(raw_constraints, str):
            constraints = [raw_constraints]
        elif isinstance(raw_constraints, list):
            constraints = [item for item in raw_constraints if isinstance(item, str)]
        else:
            constraints = []
        return RuntimeSkillDefinition(
            key=skill.key,
            name=skill.name,
            description=skill.description,
            version=skill.version,
            required_permissions=list(skill.required_permissions),
            instructions=instructions,
            constraints=constraints,
        )
