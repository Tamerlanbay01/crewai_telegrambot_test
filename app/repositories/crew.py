"""Persistence-only operations for crews, members, and task definitions."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.entities.crew import CrewEntity
from database.entities.crew_agent import CrewAgentEntity
from database.entities.crew_task import CrewTaskEntity
from database.entities.crew_skill import CrewSkillEntity
from models.crew import (
    CrewMember, CrewMemberCreate, CrewSkill, CrewSkillCreate, CrewStatus, CrewTask, CrewTaskCreate,
    UserCrew, UserCrewCreate,
)


class CrewRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, data: UserCrewCreate) -> UserCrew:
        entity = CrewEntity(**data.model_dump(mode="python"))
        entity.process = data.process.value
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        return UserCrew.model_validate(entity)

    async def get_by_id(self, crew_id: UUID) -> UserCrew | None:
        entity = (
            await self._session.execute(select(CrewEntity).where(CrewEntity.id == crew_id))
        ).scalar_one_or_none()
        return UserCrew.model_validate(entity) if entity is not None else None

    async def list_by_user(self, user_id: int) -> list[UserCrew]:
        rows = (
            await self._session.execute(
                select(CrewEntity).where(CrewEntity.user_id == user_id)
                .order_by(CrewEntity.created_at, CrewEntity.id)
            )
        ).scalars().all()
        return [UserCrew.model_validate(row) for row in rows]

    async def set_status(self, crew_id: UUID, status: CrewStatus) -> UserCrew | None:
        entity = (
            await self._session.execute(select(CrewEntity).where(CrewEntity.id == crew_id))
        ).scalar_one_or_none()
        if entity is None:
            return None
        entity.status = status.value
        await self._session.flush()
        await self._session.refresh(entity)
        return UserCrew.model_validate(entity)

    async def add_member(self, data: CrewMemberCreate) -> CrewMember:
        entity = CrewAgentEntity(**data.model_dump())
        self._session.add(entity)
        await self._session.flush()
        return CrewMember.model_validate(entity)

    async def list_members(self, crew_id: UUID) -> list[CrewMember]:
        rows = (
            await self._session.execute(
                select(CrewAgentEntity).where(CrewAgentEntity.crew_id == crew_id)
                .order_by(CrewAgentEntity.position)
            )
        ).scalars().all()
        return [CrewMember.model_validate(row) for row in rows]

    async def add_task(self, data: CrewTaskCreate) -> CrewTask:
        entity = CrewTaskEntity(**data.model_dump())
        self._session.add(entity)
        await self._session.flush()
        return CrewTask.model_validate(entity)

    async def list_tasks(self, crew_id: UUID) -> list[CrewTask]:
        rows = (
            await self._session.execute(
                select(CrewTaskEntity).where(CrewTaskEntity.crew_id == crew_id)
                .order_by(CrewTaskEntity.position)
            )
        ).scalars().all()
        return [CrewTask.model_validate(row) for row in rows]

    async def add_skill(self, data: CrewSkillCreate) -> CrewSkill:
        entity = CrewSkillEntity(**data.model_dump())
        self._session.add(entity)
        await self._session.flush()
        return CrewSkill.model_validate(entity)

    async def list_skills(self, crew_id: UUID) -> list[CrewSkill]:
        rows = (
            await self._session.execute(
                select(CrewSkillEntity).where(CrewSkillEntity.crew_id == crew_id)
                .order_by(CrewSkillEntity.skill_id)
            )
        ).scalars().all()
        return [CrewSkill.model_validate(row) for row in rows]
