"""Persistence operations for schedules and atomic due-job claims."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.entities.schedule import ScheduleEntity
from models.schedule import Schedule, ScheduleCreate, ScheduleStatus, ScheduleUpdate


class ScheduleRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, data: ScheduleCreate) -> Schedule:
        entity = ScheduleEntity(**data.model_dump())
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        return Schedule.model_validate(entity)

    async def get_by_id(self, schedule_id: UUID) -> Schedule | None:
        entity = (
            await self._session.execute(
                select(ScheduleEntity).where(ScheduleEntity.id == schedule_id)
            )
        ).scalar_one_or_none()
        return Schedule.model_validate(entity) if entity is not None else None

    async def list_by_user(self, user_id: int) -> list[Schedule]:
        entities = (
            await self._session.execute(
                select(ScheduleEntity)
                .where(ScheduleEntity.user_id == user_id)
                .order_by(ScheduleEntity.created_at.desc(), ScheduleEntity.id)
            )
        ).scalars().all()
        return [Schedule.model_validate(entity) for entity in entities]

    async def update(self, schedule_id: UUID, data: ScheduleUpdate) -> Schedule | None:
        entity = (
            await self._session.execute(
                select(ScheduleEntity).where(ScheduleEntity.id == schedule_id)
            )
        ).scalar_one_or_none()
        if entity is None:
            return None
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(entity, field, value)
        await self._session.flush()
        await self._session.refresh(entity)
        return Schedule.model_validate(entity)

    async def update_state(
        self,
        schedule_id: UUID,
        *,
        status: ScheduleStatus,
        next_run_at: datetime | None,
        last_run_at: datetime | None,
        now: datetime,
    ) -> Schedule | None:
        result = await self._session.execute(
            update(ScheduleEntity)
            .where(ScheduleEntity.id == schedule_id)
            .values(
                status=status,
                next_run_at=next_run_at,
                last_run_at=last_run_at,
                claim_token=None,
                claim_expires_at=None,
                updated_at=now,
            )
        )
        if result.rowcount != 1:
            return None
        return await self.get_by_id(schedule_id)

    async def list_due(self, now: datetime) -> list[Schedule]:
        entities = (
            await self._session.execute(
                select(ScheduleEntity)
                .where(
                    ScheduleEntity.status == ScheduleStatus.ACTIVE,
                    ScheduleEntity.next_run_at.is_not(None),
                    ScheduleEntity.next_run_at <= now,
                    or_(
                        ScheduleEntity.claim_token.is_(None),
                        ScheduleEntity.claim_expires_at.is_(None),
                        ScheduleEntity.claim_expires_at <= now,
                    ),
                )
                .order_by(ScheduleEntity.next_run_at, ScheduleEntity.id)
            )
        ).scalars().all()
        return [Schedule.model_validate(entity) for entity in entities]

    async def claim_due(
        self,
        *,
        schedule_id: UUID,
        token: UUID,
        now: datetime,
        expires_at: datetime,
    ) -> Schedule | None:
        result = await self._session.execute(
            update(ScheduleEntity)
            .where(
                ScheduleEntity.id == schedule_id,
                ScheduleEntity.status == ScheduleStatus.ACTIVE,
                ScheduleEntity.next_run_at.is_not(None),
                ScheduleEntity.next_run_at <= now,
                or_(
                    ScheduleEntity.claim_token.is_(None),
                    ScheduleEntity.claim_expires_at.is_(None),
                    ScheduleEntity.claim_expires_at <= now,
                ),
            )
            .values(
                claim_token=token,
                claim_expires_at=expires_at,
                updated_at=now,
            )
        )
        if result.rowcount != 1:
            return None
        return await self.get_by_id(schedule_id)

    async def finish_claim(
        self,
        *,
        schedule_id: UUID,
        token: UUID,
        status: ScheduleStatus,
        next_run_at: datetime | None,
        last_run_at: datetime,
        now: datetime,
    ) -> Schedule | None:
        result = await self._session.execute(
            update(ScheduleEntity)
            .where(
                ScheduleEntity.id == schedule_id,
                ScheduleEntity.claim_token == token,
            )
            .values(
                status=status,
                next_run_at=next_run_at,
                last_run_at=last_run_at,
                claim_token=None,
                claim_expires_at=None,
                updated_at=now,
            )
        )
        if result.rowcount != 1:
            return None
        return await self.get_by_id(schedule_id)
