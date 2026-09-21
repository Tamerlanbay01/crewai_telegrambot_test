"""Persistence operations for skills and persistent-agent assignments."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.entities.agent_skill import AgentSkillEntity
from database.entities.skill import SkillEntity
from models.skill import (
    AgentSkill,
    AgentSkillCreate,
    Skill,
    SkillCreate,
    SkillOwnerType,
    SkillStatus,
)


class SkillRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_id(self, skill_id: UUID) -> Skill | None:
        entity = (
            await self._session.execute(
                select(SkillEntity).where(SkillEntity.id == skill_id)
            )
        ).scalar_one_or_none()
        return Skill.model_validate(entity) if entity is not None else None

    async def get_by_key(
        self,
        *,
        key: str,
        owner_type: SkillOwnerType,
        owner_user_id: int | None = None,
        version: int | None = None,
    ) -> Skill | None:
        statement = select(SkillEntity).where(
            SkillEntity.key == key,
            SkillEntity.owner_type == owner_type,
        )
        if owner_type == SkillOwnerType.USER:
            if owner_user_id is None:
                return None
            statement = statement.where(SkillEntity.owner_user_id == owner_user_id)
        else:
            if owner_user_id is not None:
                return None
            statement = statement.where(SkillEntity.owner_user_id.is_(None))
        if version is not None:
            statement = statement.where(SkillEntity.version == version)
        else:
            statement = statement.order_by(SkillEntity.version.desc()).limit(1)
        entity = (await self._session.execute(statement)).scalar_one_or_none()
        return Skill.model_validate(entity) if entity is not None else None

    async def list_system(self, *, active_only: bool = False) -> list[Skill]:
        statement = (
            select(SkillEntity)
            .where(
                SkillEntity.owner_type == SkillOwnerType.SYSTEM,
                SkillEntity.owner_user_id.is_(None),
            )
            .order_by(SkillEntity.key, SkillEntity.version.desc())
        )
        if active_only:
            statement = statement.where(SkillEntity.status == SkillStatus.ACTIVE)
        entities = (await self._session.execute(statement)).scalars().all()
        return [Skill.model_validate(entity) for entity in entities]

    async def list_by_owner(
        self, *, owner_user_id: int, status: SkillStatus | None = None
    ) -> list[Skill]:
        statement = (
            select(SkillEntity)
            .where(
                SkillEntity.owner_type == SkillOwnerType.USER,
                SkillEntity.owner_user_id == owner_user_id,
            )
            .order_by(SkillEntity.key, SkillEntity.version.desc())
        )
        if status is not None:
            statement = statement.where(SkillEntity.status == status)
        entities = (await self._session.execute(statement)).scalars().all()
        return [Skill.model_validate(entity) for entity in entities]

    async def create(self, data: SkillCreate) -> Skill:
        entity = SkillEntity(**data.model_dump())
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        return Skill.model_validate(entity)

    async def update_status(self, skill_id: UUID, status: SkillStatus) -> Skill | None:
        entity = (
            await self._session.execute(
                select(SkillEntity).where(SkillEntity.id == skill_id)
            )
        ).scalar_one_or_none()
        if entity is None:
            return None
        entity.status = status
        await self._session.flush()
        await self._session.refresh(entity)
        return Skill.model_validate(entity)

    async def get_agent_skill(
        self, *, agent_id: UUID, skill_id: UUID
    ) -> AgentSkill | None:
        entity = (
            await self._session.execute(
                select(AgentSkillEntity).where(
                    AgentSkillEntity.agent_id == agent_id,
                    AgentSkillEntity.skill_id == skill_id,
                )
            )
        ).scalar_one_or_none()
        return AgentSkill.model_validate(entity) if entity is not None else None

    async def list_agent_skills(self, agent_id: UUID) -> list[AgentSkill]:
        entities = (
            await self._session.execute(
                select(AgentSkillEntity)
                .where(AgentSkillEntity.agent_id == agent_id)
                .order_by(AgentSkillEntity.created_at, AgentSkillEntity.id)
            )
        ).scalars().all()
        return [AgentSkill.model_validate(entity) for entity in entities]

    async def assign(self, data: AgentSkillCreate) -> AgentSkill:
        existing = await self.get_agent_skill(agent_id=data.agent_id, skill_id=data.skill_id)
        if existing is not None:
            if existing.enabled != data.enabled:
                entity = (
                    await self._session.execute(
                        select(AgentSkillEntity).where(AgentSkillEntity.id == existing.id)
                    )
                ).scalar_one()
                entity.enabled = data.enabled
                await self._session.flush()
                await self._session.refresh(entity)
                return AgentSkill.model_validate(entity)
            return existing

        entity = AgentSkillEntity(**data.model_dump())
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        return AgentSkill.model_validate(entity)

    async def remove(self, *, agent_id: UUID, skill_id: UUID) -> bool:
        entity = (
            await self._session.execute(
                select(AgentSkillEntity).where(
                    AgentSkillEntity.agent_id == agent_id,
                    AgentSkillEntity.skill_id == skill_id,
                )
            )
        ).scalar_one_or_none()
        if entity is None:
            return False
        await self._session.delete(entity)
        await self._session.flush()
        return True
