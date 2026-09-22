"""Repository operations for one-action approvals."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.entities.approval import ApprovalEntity
from models.approval import Approval, ApprovalCreate, ApprovalStatus


class ApprovalRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_id(self, approval_id: UUID) -> Approval | None:
        entity = (
            await self._session.execute(
                select(ApprovalEntity).where(ApprovalEntity.id == approval_id)
            )
        ).scalar_one_or_none()
        return Approval.model_validate(entity) if entity else None

    async def get_pending_for_run(self, run_id: UUID) -> Approval | None:
        entity = (
            await self._session.execute(
                select(ApprovalEntity)
                .where(
                    ApprovalEntity.run_id == run_id,
                    ApprovalEntity.status == ApprovalStatus.PENDING,
                )
                .order_by(ApprovalEntity.requested_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        return Approval.model_validate(entity) if entity else None

    async def create(self, data: ApprovalCreate) -> Approval:
        entity = ApprovalEntity(**data.model_dump())
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        return Approval.model_validate(entity)

    async def decide(
        self,
        *,
        approval_id: UUID,
        status: ApprovalStatus,
        decided_at: datetime,
        decision_metadata: dict[str, object],
    ) -> Approval | None:
        result = await self._session.execute(
            update(ApprovalEntity)
            .where(
                ApprovalEntity.id == approval_id,
                ApprovalEntity.status == ApprovalStatus.PENDING,
            )
            .values(
                status=status,
                decided_at=decided_at,
                decision_metadata=decision_metadata,
            )
            .execution_options(synchronize_session="fetch")
        )
        if result.rowcount != 1:
            return None
        return await self.get_by_id(approval_id)

    async def update_decision_metadata(
        self, *, approval_id: UUID, decision_metadata: dict[str, object]
    ) -> Approval | None:
        entity = (
            await self._session.execute(
                select(ApprovalEntity).where(ApprovalEntity.id == approval_id)
            )
        ).scalar_one_or_none()
        if entity is None:
            return None
        entity.decision_metadata = decision_metadata
        await self._session.flush()
        await self._session.refresh(entity)
        return Approval.model_validate(entity)
