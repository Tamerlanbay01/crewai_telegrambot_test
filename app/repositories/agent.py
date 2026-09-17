from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.entities.agent import AgentEntity
from models.agent import Agent, AgentCreate, AgentKind, AgentUpdate


class AgentRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_id(self, agent_id: UUID) -> Agent | None:
        entity = await self._get_entity_by_id(
            agent_id
        )

        if entity is None:
            return None

        return Agent.model_validate(entity)

    async def get_primary(self, user_id: int) -> Agent | None:
        result = await self._session.execute(
            select(AgentEntity)
            .where(
                AgentEntity.user_id == user_id,
                AgentEntity.kind
                == AgentKind.PRIMARY,
            )
        )

        entity = result.scalar_one_or_none()

        if entity is None:
            return None

        return Agent.model_validate(entity)

    async def list_by_user(self, user_id: int) -> list[Agent]:
        result = await self._session.execute(
            select(AgentEntity)
            .where(
                AgentEntity.user_id == user_id
            )
            .order_by(
                AgentEntity.created_at.asc()
            )
        )

        return [
            Agent.model_validate(entity)
            for entity in result.scalars().all()
        ]

    async def create(self, data: AgentCreate) -> Agent:
        entity = AgentEntity(
            **data.model_dump()
        )

        self._session.add(entity)

        await self._session.flush()
        await self._session.refresh(entity)

        return Agent.model_validate(entity)

    async def update(self, agent_id: UUID, data: AgentUpdate) -> Agent | None:
        entity = await self._get_entity_by_id(
            agent_id
        )

        if entity is None:
            return None

        changes = data.model_dump(
            exclude_unset=True,
        )

        for field, value in changes.items():
            setattr(entity, field, value)

        await self._session.flush()
        await self._session.refresh(entity)

        return Agent.model_validate(entity)

    async def _get_entity_by_id(self, agent_id: UUID) -> AgentEntity | None:
        result = await self._session.execute(
            select(AgentEntity)
            .where(AgentEntity.id == agent_id)
        )

        return result.scalar_one_or_none()