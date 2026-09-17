from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.entities.agent_connection import AgentConnectionEntity
from models.agent_connection import AgentConnection, AgentConnectionCreate



class AgentConnectionRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get(self, *, parent_agent_id: UUID, child_agent_id: UUID) -> AgentConnection | None:
        result = await self._session.execute(
            select(AgentConnectionEntity)
            .where(AgentConnectionEntity.parent_agent_id == parent_agent_id,
                AgentConnectionEntity.child_agent_id == child_agent_id)
        )

        entity = result.scalar_one_or_none()

        if entity is None:
            return None

        return AgentConnection.model_validate(entity)

    async def list_children(self, parent_agent_id: UUID) -> list[AgentConnection]:
        result = await self._session.execute(
            select(AgentConnectionEntity)
            .where(AgentConnectionEntity.parent_agent_id == parent_agent_id)
            .order_by(AgentConnectionEntity.created_at.asc())
        )

        return [
            AgentConnection.model_validate(entity)
            for entity in result.scalars().all()
        ]

    async def create(
        self,
        data: AgentConnectionCreate,
    ) -> AgentConnection:
        entity = AgentConnectionEntity(
            **data.model_dump()
        )

        self._session.add(entity)

        await self._session.flush()
        await self._session.refresh(entity)

        return AgentConnection.model_validate(entity)

    async def delete(
        self,
        *,
        parent_agent_id: UUID,
        child_agent_id: UUID,
    ) -> bool:
        result = await self._session.execute(
            select(AgentConnectionEntity)
            .where(AgentConnectionEntity.parent_agent_id == parent_agent_id,
                AgentConnectionEntity.child_agent_id == child_agent_id)
        )

        entity = result.scalar_one_or_none()

        if entity is None:
            return False

        await self._session.delete(entity) 
        await self._session.flush()

        return True