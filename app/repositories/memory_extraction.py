"""Persistence-only state transitions for durable memory extraction work."""

from datetime import datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import and_, case, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.entities.memory_candidate import MemoryCandidateEntity
from database.entities.memory_extraction_run import MemoryExtractionRunEntity
from models.memory import (
    MemoryCandidate,
    MemoryCandidateAudit,
    MemoryCandidateDecision,
    MemoryExtractionRun,
    MemoryExtractionStatus,
)

TERMINAL_CANDIDATE_DECISIONS = frozenset(
    {
        MemoryCandidateDecision.SAVED,
        MemoryCandidateDecision.UPDATED,
        MemoryCandidateDecision.MERGED,
        MemoryCandidateDecision.IGNORED,
        MemoryCandidateDecision.REJECTED,
    }
)


class MemoryExtractionRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_id(self, extraction_run_id: UUID) -> MemoryExtractionRun | None:
        entity = (
            await self._session.execute(
                select(MemoryExtractionRunEntity).where(
                    MemoryExtractionRunEntity.id == extraction_run_id
                )
            )
        ).scalar_one_or_none()
        return MemoryExtractionRun.model_validate(entity) if entity is not None else None

    async def get_by_agent_run(self, agent_run_id: UUID) -> MemoryExtractionRun | None:
        entity = (
            await self._session.execute(
                select(MemoryExtractionRunEntity).where(
                    MemoryExtractionRunEntity.agent_run_id == agent_run_id
                )
            )
        ).scalar_one_or_none()
        return MemoryExtractionRun.model_validate(entity) if entity is not None else None

    async def ensure_pending(
        self, *, agent_run_id: UUID, user_id: int, model_name: str | None
    ) -> MemoryExtractionRun:
        existing = await self.get_by_agent_run(agent_run_id)
        if existing is not None:
            return existing
        entity = MemoryExtractionRunEntity(
            id=uuid4(),
            agent_run_id=agent_run_id,
            user_id=user_id,
            status=MemoryExtractionStatus.PENDING,
            model_name=model_name,
            attempt_count=0,
        )
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        return MemoryExtractionRun.model_validate(entity)

    async def claim_by_id(
        self,
        *,
        extraction_run_id: UUID,
        now: datetime,
        claim_token: UUID,
        claim_expires_at: datetime,
    ) -> MemoryExtractionRun | None:
        result = await self._session.execute(
            update(MemoryExtractionRunEntity)
            .where(
                MemoryExtractionRunEntity.id == extraction_run_id,
                self._claimable_condition(now),
            )
            .values(
                status=MemoryExtractionStatus.RUNNING,
                claim_token=claim_token,
                claim_expires_at=claim_expires_at,
                attempt_count=MemoryExtractionRunEntity.attempt_count + 1,
                next_retry_at=None,
                error=None,
                started_at=case(
                    (MemoryExtractionRunEntity.started_at.is_(None), now),
                    else_=MemoryExtractionRunEntity.started_at,
                ),
            )
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            return None
        return await self.get_by_id(extraction_run_id)

    async def claim_batch(
        self,
        *,
        now: datetime,
        limit: int,
        lease_seconds: int,
    ) -> list[MemoryExtractionRun]:
        if limit <= 0:
            return []
        candidate_ids = (
            await self._session.execute(
                select(MemoryExtractionRunEntity.id)
                .where(self._claimable_condition(now))
                .order_by(MemoryExtractionRunEntity.created_at.asc(), MemoryExtractionRunEntity.id.asc())
                .limit(limit)
            )
        ).scalars().all()
        claimed: list[MemoryExtractionRun] = []
        for extraction_run_id in candidate_ids:
            claim = await self.claim_by_id(
                extraction_run_id=extraction_run_id,
                now=now,
                claim_token=uuid4(),
                claim_expires_at=now + timedelta(seconds=lease_seconds),
            )
            if claim is not None:
                claimed.append(claim)
        return claimed

    async def owns_live_claim(
        self, *, extraction_run_id: UUID, claim_token: UUID, now: datetime
    ) -> bool:
        result = await self._session.execute(
            select(MemoryExtractionRunEntity.id).where(
                MemoryExtractionRunEntity.id == extraction_run_id,
                MemoryExtractionRunEntity.status == MemoryExtractionStatus.RUNNING,
                MemoryExtractionRunEntity.claim_token == claim_token,
                MemoryExtractionRunEntity.claim_expires_at.is_not(None),
                MemoryExtractionRunEntity.claim_expires_at > now,
            )
        )
        return result.scalar_one_or_none() is not None

    async def mark_candidates_extracted(
        self,
        *,
        extraction_run_id: UUID,
        claim_token: UUID,
        at: datetime,
    ) -> bool:
        result = await self._session.execute(
            update(MemoryExtractionRunEntity)
            .where(
                MemoryExtractionRunEntity.id == extraction_run_id,
                MemoryExtractionRunEntity.status == MemoryExtractionStatus.RUNNING,
                MemoryExtractionRunEntity.claim_token == claim_token,
            )
            .values(candidates_extracted_at=at)
            .execution_options(synchronize_session=False)
        )
        return result.rowcount == 1

    async def mark_completed(
        self,
        *,
        extraction_run_id: UUID,
        claim_token: UUID,
        at: datetime,
    ) -> MemoryExtractionRun | None:
        result = await self._session.execute(
            update(MemoryExtractionRunEntity)
            .where(
                MemoryExtractionRunEntity.id == extraction_run_id,
                MemoryExtractionRunEntity.status == MemoryExtractionStatus.RUNNING,
                MemoryExtractionRunEntity.claim_token == claim_token,
            )
            .values(
                status=MemoryExtractionStatus.COMPLETED,
                completed_at=at,
                next_retry_at=None,
                claim_token=None,
                claim_expires_at=None,
                error=None,
            )
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            return None
        return await self.get_by_id(extraction_run_id)

    async def mark_failed(
        self,
        *,
        extraction_run_id: UUID,
        claim_token: UUID,
        error: str,
        next_retry_at: datetime | None,
    ) -> MemoryExtractionRun | None:
        result = await self._session.execute(
            update(MemoryExtractionRunEntity)
            .where(
                MemoryExtractionRunEntity.id == extraction_run_id,
                MemoryExtractionRunEntity.status == MemoryExtractionStatus.RUNNING,
                MemoryExtractionRunEntity.claim_token == claim_token,
            )
            .values(
                status=MemoryExtractionStatus.FAILED,
                next_retry_at=next_retry_at,
                claim_token=None,
                claim_expires_at=None,
                error=error[:2048],
            )
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            return None
        return await self.get_by_id(extraction_run_id)

    async def reset_for_manual_retry(
        self, *, extraction_run_id: UUID, now: datetime
    ) -> MemoryExtractionRun | None:
        result = await self._session.execute(
            update(MemoryExtractionRunEntity)
            .where(
                MemoryExtractionRunEntity.id == extraction_run_id,
                or_(
                    MemoryExtractionRunEntity.status == MemoryExtractionStatus.FAILED,
                    and_(
                        MemoryExtractionRunEntity.status == MemoryExtractionStatus.RUNNING,
                        MemoryExtractionRunEntity.claim_expires_at <= now,
                    ),
                    MemoryExtractionRunEntity.status == MemoryExtractionStatus.PENDING,
                ),
            )
            .values(
                status=MemoryExtractionStatus.PENDING,
                attempt_count=0,
                next_retry_at=None,
                claim_token=None,
                claim_expires_at=None,
                error=None,
            )
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            return None
        return await self.get_by_id(extraction_run_id)

    async def list_terminal_failed(self, *, limit: int) -> list[MemoryExtractionRun]:
        if limit <= 0:
            return []
        entities = (
            await self._session.execute(
                select(MemoryExtractionRunEntity)
                .where(
                    MemoryExtractionRunEntity.status == MemoryExtractionStatus.FAILED,
                    MemoryExtractionRunEntity.next_retry_at.is_(None),
                )
                .order_by(MemoryExtractionRunEntity.created_at.asc(), MemoryExtractionRunEntity.id.asc())
                .limit(limit)
            )
        ).scalars().all()
        return [MemoryExtractionRun.model_validate(entity) for entity in entities]

    async def list_candidates(self, extraction_run_id: UUID) -> list[MemoryCandidateAudit]:
        entities = (
            await self._session.execute(
                select(MemoryCandidateEntity)
                .where(MemoryCandidateEntity.extraction_run_id == extraction_run_id)
                .order_by(MemoryCandidateEntity.created_at.asc(), MemoryCandidateEntity.id.asc())
            )
        ).scalars().all()
        return [MemoryCandidateAudit.model_validate(entity) for entity in entities]

    async def create_candidate(
        self, *, extraction_run_id: UUID, candidate: MemoryCandidate
    ) -> MemoryCandidateAudit:
        entity = MemoryCandidateEntity(
            id=uuid4(),
            extraction_run_id=extraction_run_id,
            **candidate.model_dump(),
            decision=MemoryCandidateDecision.PENDING,
        )
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        return MemoryCandidateAudit.model_validate(entity)

    async def set_candidate_decision(
        self, *, candidate_id: UUID, decision: MemoryCandidateDecision
    ) -> MemoryCandidateAudit:
        entity = (
            await self._session.execute(
                select(MemoryCandidateEntity).where(MemoryCandidateEntity.id == candidate_id)
            )
        ).scalar_one()
        entity.decision = decision
        await self._session.flush()
        await self._session.refresh(entity)
        return MemoryCandidateAudit.model_validate(entity)

    @staticmethod
    def _claimable_condition(now: datetime):
        return or_(
            MemoryExtractionRunEntity.status == MemoryExtractionStatus.PENDING,
            and_(
                MemoryExtractionRunEntity.status == MemoryExtractionStatus.FAILED,
                MemoryExtractionRunEntity.next_retry_at.is_not(None),
                MemoryExtractionRunEntity.next_retry_at <= now,
            ),
            and_(
                MemoryExtractionRunEntity.status == MemoryExtractionStatus.RUNNING,
                MemoryExtractionRunEntity.claim_expires_at.is_not(None),
                MemoryExtractionRunEntity.claim_expires_at <= now,
            ),
        )
