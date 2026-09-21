"""Repository operations for agent runs and their immutable events."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.entities.agent_run import AgentRunEntity
from database.entities.agent_run_event import AgentRunEventEntity
from models.agent_run import (
    AgentRun,
    AgentRunCreate,
    AgentRunEvent,
    AgentRunEventCreate,
    AgentRunStatus,
)


class AgentRunRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, data: AgentRunCreate) -> AgentRun:
        entity = AgentRunEntity(**data.model_dump())
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        return AgentRun.model_validate(entity)

    async def get_by_id(self, run_id: UUID) -> AgentRun | None:
        entity = (
            await self._session.execute(
                select(AgentRunEntity).where(AgentRunEntity.id == run_id)
            )
        ).scalar_one_or_none()
        return AgentRun.model_validate(entity) if entity else None

    async def update_status(
        self,
        *,
        run_id: UUID,
        status: AgentRunStatus,
        error: str | None = None,
        result_metadata: dict[str, object] | None = None,
        usage: dict[str, object] | None = None,
        checkpoint: dict[str, object] | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> AgentRun | None:
        entity = (
            await self._session.execute(
                select(AgentRunEntity).where(AgentRunEntity.id == run_id)
            )
        ).scalar_one_or_none()
        if entity is None:
            return None
        entity.status = status
        if error is not None:
            entity.error = error
        if result_metadata is not None:
            entity.result_metadata = result_metadata
        if usage is not None:
            entity.usage = usage
        if checkpoint is not None:
            entity.checkpoint = checkpoint
        if started_at is not None:
            entity.started_at = started_at
        if completed_at is not None:
            entity.completed_at = completed_at
        await self._session.flush()
        await self._session.refresh(entity)
        return AgentRun.model_validate(entity)

    async def update_runtime_state(
        self,
        *,
        run_id: UUID,
        usage: dict[str, object],
        checkpoint: dict[str, object],
    ) -> AgentRun | None:
        entity = (
            await self._session.execute(
                select(AgentRunEntity).where(AgentRunEntity.id == run_id)
            )
        ).scalar_one_or_none()
        if entity is None:
            return None
        entity.usage = usage
        entity.checkpoint = checkpoint
        await self._session.flush()
        await self._session.refresh(entity)
        return AgentRun.model_validate(entity)

    async def append_event(self, data: AgentRunEventCreate) -> AgentRunEvent:
        entity = AgentRunEventEntity(**data.model_dump())
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        return AgentRunEvent.model_validate(entity)

    async def list_events(self, run_id: UUID) -> list[AgentRunEvent]:
        result = await self._session.execute(
            select(AgentRunEventEntity)
            .where(AgentRunEventEntity.run_id == run_id)
            .order_by(AgentRunEventEntity.created_at, AgentRunEventEntity.id)
        )
        return [AgentRunEvent.model_validate(entity) for entity in result.scalars().all()]
