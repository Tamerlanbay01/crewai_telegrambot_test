"""Persistence operations for bounded executable-skill results."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.entities.skill_execution import SkillExecutionEntity
from models.skill import SkillExecution, SkillExecutionCreate, SkillExecutionResult, SkillExecutionStatus


class SkillExecutionRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_idempotency_key(self, key: str) -> SkillExecution | None:
        entity = (
            await self._session.execute(
                select(SkillExecutionEntity).where(SkillExecutionEntity.idempotency_key == key)
            )
        ).scalar_one_or_none()
        return SkillExecution.model_validate(entity) if entity is not None else None

    async def create(self, data: SkillExecutionCreate) -> SkillExecution:
        entity = SkillExecutionEntity(**data.model_dump())
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        return SkillExecution.model_validate(entity)

    async def update_result(
        self,
        *,
        execution_id: UUID,
        result: SkillExecutionResult,
        started_at: datetime | None,
        completed_at: datetime,
        stdout_preview: str,
        stderr_preview: str,
        error: str | None,
    ) -> SkillExecution | None:
        entity = (
            await self._session.execute(
                select(SkillExecutionEntity).where(SkillExecutionEntity.id == execution_id)
            )
        ).scalar_one_or_none()
        if entity is None:
            return None
        entity.status = result.status
        entity.started_at = started_at
        entity.completed_at = completed_at
        entity.duration_ms = result.duration_ms
        entity.exit_code = result.exit_code
        entity.stdout_preview = stdout_preview
        entity.stderr_preview = stderr_preview
        entity.error = error or result.error
        await self._session.flush()
        await self._session.refresh(entity)
        return SkillExecution.model_validate(entity)

    async def mark_running(self, *, execution_id: UUID, started_at: datetime) -> SkillExecution | None:
        entity = (
            await self._session.execute(
                select(SkillExecutionEntity).where(SkillExecutionEntity.id == execution_id)
            )
        ).scalar_one_or_none()
        if entity is None:
            return None
        entity.status = SkillExecutionStatus.RUNNING
        entity.started_at = started_at
        await self._session.flush()
        await self._session.refresh(entity)
        return SkillExecution.model_validate(entity)
