from uuid import UUID
from sqlalchemy import func, select

from sqlalchemy.ext.asyncio import AsyncSession

from database.entities.agent_prompt import AgentPromptVersionEntity
from models.agent_prompt import AgentPromptVersion, AgentPromptVersionCreate

class AgentPromptRepositiry:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_id(self, prompt_version_id: UUID) -> AgentPromptVersion | None:
        result = await self._session.execute(
            select(AgentPromptVersionEntity)
            .where(AgentPromptVersionEntity.id == prompt_version_id)
        )
        entity = result.scalar_one_or_none()
        if entity is None:
            return None
        return AgentPromptVersion.model_validate(entity)

    async def get_latest(self, agent_id: UUID) -> AgentPromptVersion | None:
        result = await self._session.execute(
            select(AgentPromptVersionEntity)
            .where(AgentPromptVersionEntity.agent_id == agent_id)
            .order_by(AgentPromptVersionEntity.version.decs())
            .limit(1)
        )

        entity = result.scalar_one_or_none()

        if entity is None:
            return None
        return AgentPromptVersion.model_validate(entity)

    async def get_next_version(self, agent_id: UUID):
        result = await self._session.execute(
            select(func.max(AgentPromptVersionEntity.version))
            .where(AgentPromptVersionEntity.agent_id == agent_id)
        )
        current_version = result.scalar_one()

        if current_version is None:
            return 1
        return current_version + 1

    async def create(self, data: AgentPromptVersionCreate) -> AgentPromptVersion:
        entity = AgentPromptVersionEntity(**data.model_dump())
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)

        return AgentPromptVersion.model_validate(entity)

