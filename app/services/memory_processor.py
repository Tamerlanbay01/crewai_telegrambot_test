"""Transport-neutral processor for persisted memory extraction work."""

from datetime import datetime, timedelta, timezone
import logging
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from agents.memory.extractor import MemoryExtractionAgent
from core.config import config
from models.memory import MemoryExtractionRun, MemoryExtractionStatus
from repositories.memory_extraction import MemoryExtractionRepository
from services.memory_extraction import MemoryClaimLostError, MemoryExtractionService

logger = logging.getLogger(__name__)


class MemoryExtractionProcessor:
    """Claims bounded batches, then runs external work outside the claim transaction."""

    def __init__(
        self, session: AsyncSession, *, extractor: MemoryExtractionAgent | None = None
    ):
        self._session = session
        self._extractions = MemoryExtractionRepository(session)
        self._service = MemoryExtractionService(session, extractor=extractor)

    async def run_once(self, *, now: datetime | None = None) -> list[MemoryExtractionRun]:
        return await self.process_pending(now=now)

    async def process_pending(
        self, *, limit: int | None = None, now: datetime | None = None
    ) -> list[MemoryExtractionRun]:
        current_time = now or datetime.now(timezone.utc)
        batch_size = config.memory_processing_batch_size if limit is None else limit
        claims = await self._extractions.claim_batch(
            now=current_time,
            limit=batch_size,
            lease_seconds=config.memory_extraction_lease_seconds,
        )
        await self._session.commit()
        for claim in claims:
            logger.info(
                "Memory extraction claimed extraction_run_id=%s agent_run_id=%s attempt_count=%s "
                "status=%s",
                claim.id,
                claim.agent_run_id,
                claim.attempt_count,
                claim.status.value,
            )
        processed: list[MemoryExtractionRun] = []
        for claim in claims:
            result = await self._process_claim(claim, now=now)
            if result is not None:
                processed.append(result)
        return processed

    async def process_one(
        self, *, extraction_run_id: UUID, now: datetime | None = None
    ) -> MemoryExtractionRun | None:
        current_time = now or datetime.now(timezone.utc)
        claim = await self._extractions.claim_by_id(
            extraction_run_id=extraction_run_id,
            now=current_time,
            claim_token=uuid4(),
            claim_expires_at=current_time
            + timedelta(seconds=config.memory_extraction_lease_seconds),
        )
        await self._session.commit()
        if claim is None:
            return None
        return await self._process_claim(claim, now=now)

    async def retry(
        self, *, extraction_run_id: UUID, now: datetime | None = None
    ) -> MemoryExtractionRun:
        current_time = now or datetime.now(timezone.utc)
        existing = await self._extractions.get_by_id(extraction_run_id)
        if existing is None:
            raise LookupError(f"Memory extraction run not found: {extraction_run_id}")
        if existing.status == MemoryExtractionStatus.COMPLETED:
            raise ValueError("Completed memory extraction cannot be retried")
        retried = await self._extractions.reset_for_manual_retry(
            extraction_run_id=extraction_run_id,
            now=current_time,
        )
        if retried is None:
            raise ValueError("Memory extraction has a live claim and cannot be retried")
        await self._session.commit()
        return retried

    async def retry_failed(
        self, *, limit: int | None = None, now: datetime | None = None
    ) -> list[MemoryExtractionRun]:
        current_time = now or datetime.now(timezone.utc)
        batch_size = config.memory_processing_batch_size if limit is None else limit
        retried: list[MemoryExtractionRun] = []
        for extraction in await self._extractions.list_terminal_failed(limit=batch_size):
            retried.append(
                await self.retry(extraction_run_id=extraction.id, now=current_time)
            )
        return retried

    async def _process_claim(
        self, claim: MemoryExtractionRun, *, now: datetime | None
    ) -> MemoryExtractionRun | None:
        assert claim.claim_token is not None
        try:
            await self._service.process_claimed(
                extraction_run_id=claim.id,
                claim_token=claim.claim_token,
                now=now,
            )
            transition_time = now or datetime.now(timezone.utc)
            completed = await self._extractions.mark_completed(
                extraction_run_id=claim.id,
                claim_token=claim.claim_token,
                at=transition_time,
            )
            await self._session.commit()
            if completed is not None:
                logger.info(
                    "Memory extraction completed extraction_run_id=%s agent_run_id=%s attempt_count=%s",
                    completed.id,
                    completed.agent_run_id,
                    completed.attempt_count,
                )
            return completed
        except MemoryClaimLostError:
            await self._session.rollback()
            logger.warning(
                "Memory extraction claim lost extraction_run_id=%s agent_run_id=%s attempt_count=%s",
                claim.id,
                claim.agent_run_id,
                claim.attempt_count,
            )
            return None
        except Exception as exc:
            await self._session.rollback()
            transition_time = now or datetime.now(timezone.utc)
            retry_at = self._retry_at(claim=claim, error=exc, now=transition_time)
            failed = await self._extractions.mark_failed(
                extraction_run_id=claim.id,
                claim_token=claim.claim_token,
                error=f"{type(exc).__name__}: {exc}",
                next_retry_at=retry_at,
            )
            await self._session.commit()
            logger.warning(
                "Memory extraction failed extraction_run_id=%s agent_run_id=%s attempt_count=%s "
                "retry_at=%s error_type=%s",
                claim.id,
                claim.agent_run_id,
                claim.attempt_count,
                retry_at.isoformat() if retry_at else None,
                type(exc).__name__,
            )
            return failed

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        return not isinstance(error, (LookupError, ValidationError, ValueError))

    def _retry_at(
        self, *, claim: MemoryExtractionRun, error: Exception, now: datetime
    ) -> datetime | None:
        if not self._is_retryable(error) or claim.attempt_count >= config.memory_extraction_max_attempts:
            return None
        delay_seconds = config.memory_extraction_retry_base_seconds * (2 ** (claim.attempt_count - 1))
        return now + timedelta(seconds=delay_seconds)
