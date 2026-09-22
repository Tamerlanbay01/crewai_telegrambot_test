"""Enqueue and claimed-work execution for post-run memory extraction."""

from datetime import datetime, timezone
import logging
import re
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from agents.memory.extractor import MemoryExtractionAgent, OpenAICompatibleMemoryAgent
from core.config import config
from models.agent_run import AgentRun, AgentRunStatus
from models.memory import (
    MemoryCandidate,
    MemoryExtractionContext,
    MemoryExtractionRun,
    MemoryExtractionStatus,
    MemoryScope,
)
from repositories.agent_run import AgentRunRepository
from repositories.memory_extraction import (
    TERMINAL_CANDIDATE_DECISIONS,
    MemoryExtractionRepository,
)
from services.memory import MemoryService
from services.memory_policy import MemoryPolicyService
from services.system_agent import SystemAgentService

_SKIPPED_COMMANDS = {"/start", "/help"}
_EXPLICIT_MEMORY = re.compile(r"\b(remember|memorize|запомни|запоминай)\b", re.IGNORECASE)
logger = logging.getLogger(__name__)


class MemoryClaimLostError(RuntimeError):
    """A lease expired or another processor reclaimed the extraction."""


class MemoryExtractionService:
    """Keeps enqueue, LLM extraction, and policy application separately resumable."""

    def __init__(
        self, session: AsyncSession, *, extractor: MemoryExtractionAgent | None = None
    ):
        self._session = session
        self._runs = AgentRunRepository(session)
        self._extractions = MemoryExtractionRepository(session)
        self._memory = MemoryService(session)
        self._policy = MemoryPolicyService(session, memory=self._memory)
        self._systems = SystemAgentService(session)
        self._extractor = extractor

    async def ensure_pending(self, *, run_id: UUID) -> MemoryExtractionRun | None:
        """Persist lightweight durable work without invoking the Memory Agent."""
        run = await self._runs.get_by_id(run_id)
        if run is None:
            raise LookupError(f"Agent run not found: {run_id}")
        if not self.is_eligible(run):
            return None
        try:
            extraction = await self._extractions.ensure_pending(
                agent_run_id=run.id,
                user_id=run.user_id,
                model_name=config.llm.model or None,
            )
            await self._session.commit()
            logger.info(
                "Memory extraction enqueued extraction_run_id=%s agent_run_id=%s status=%s",
                extraction.id,
                extraction.agent_run_id,
                extraction.status.value,
            )
            return extraction
        except IntegrityError:
            await self._session.rollback()
            existing = await self._extractions.get_by_agent_run(run.id)
            if existing is None:
                raise
            return existing

    async def process_claimed(
        self,
        *,
        extraction_run_id: UUID,
        claim_token: UUID,
        now: datetime | None = None,
    ) -> None:
        """Execute only a live lease; status transitions stay in MemoryExtractionProcessor."""
        current_time = now or datetime.now(timezone.utc)
        await self._require_live_claim(extraction_run_id, claim_token, current_time)
        extraction = await self._extractions.get_by_id(extraction_run_id)
        if extraction is None:
            raise LookupError(f"Memory extraction run not found: {extraction_run_id}")
        run = await self._runs.get_by_id(extraction.agent_run_id)
        if run is None:
            raise LookupError(f"Agent run not found: {extraction.agent_run_id}")

        if extraction.candidates_extracted_at is None:
            context = await self._build_context(run)
            extractor = await self._get_extractor(run.user_id)
            await self._session.commit()
            result = await extractor.extract(context)
            current_time = now or datetime.now(timezone.utc)
            await self._require_live_claim(extraction_run_id, claim_token, current_time)
            for raw_candidate in result.candidates:
                await self._extractions.create_candidate(
                    extraction_run_id=extraction_run_id,
                    candidate=self._with_explicit_signal(raw_candidate, context),
                )
            if not await self._extractions.mark_candidates_extracted(
                extraction_run_id=extraction_run_id,
                claim_token=claim_token,
                at=current_time,
            ):
                raise MemoryClaimLostError(f"Memory extraction claim lost: {extraction_run_id}")
            await self._session.commit()

        context = await self._build_context(run)
        eligible_agents = set(context.eligible_agent_ids)
        for audit in await self._extractions.list_candidates(extraction_run_id):
            if audit.decision in TERMINAL_CANDIDATE_DECISIONS:
                continue
            current_time = now or datetime.now(timezone.utc)
            await self._require_live_claim(extraction_run_id, claim_token, current_time)
            decision, memory = await self._policy.apply(
                user_id=run.user_id,
                candidate=audit,
                eligible_agent_ids=eligible_agents,
                commit=False,
            )
            await self._extractions.set_candidate_decision(
                candidate_id=audit.id,
                decision=decision,
            )
            await self._session.commit()
            if memory is not None:
                await self._memory.sync_index(memory)

    @staticmethod
    def is_eligible(run: AgentRun) -> bool:
        if run.status != AgentRunStatus.COMPLETED:
            return False
        original = str(run.checkpoint.get("original_message") or "").strip()
        command = original.casefold().split(maxsplit=1)[0] if original else ""
        return bool(original) and command not in _SKIPPED_COMMANDS

    async def _require_live_claim(
        self, extraction_run_id: UUID, claim_token: UUID, now: datetime
    ) -> None:
        if not await self._extractions.owns_live_claim(
            extraction_run_id=extraction_run_id,
            claim_token=claim_token,
            now=now,
        ):
            raise MemoryClaimLostError(f"Memory extraction claim lost: {extraction_run_id}")

    async def _get_extractor(self, user_id: int) -> MemoryExtractionAgent:
        if self._extractor is not None:
            return self._extractor
        await self._systems.ensure_memory_template()
        system_agent = await self._systems.resolve_for_user(user_id=user_id, key="memory")
        if system_agent is None:
            raise LookupError("Internal Memory Agent template is unavailable")
        instructions = " ".join(
            value
            for value in [
                system_agent.role,
                system_agent.goal,
                system_agent.backstory,
                system_agent.custom_instructions,
            ]
            if value
        )
        return OpenAICompatibleMemoryAgent(config.llm, instructions=instructions)

    async def _build_context(self, run: AgentRun) -> MemoryExtractionContext:
        checkpoint = run.checkpoint
        original = str(checkpoint.get("original_message") or "").strip()
        final_response = str(run.result_metadata.get("content") or "").strip()
        eligible_agent_ids = self._eligible_agent_ids(run)
        existing_memory = [
            {
                "memory_type": memory.memory_type.value,
                "scope": memory.scope.value,
                "agent_id": str(memory.agent_id) if memory.agent_id else None,
                "key": memory.key,
                "content": memory.content,
            }
            for memory in await self._existing_memory(run.user_id, eligible_agent_ids)
        ]
        summaries = [
            str(record.get("task_summary", "")).strip()
            for record in checkpoint.get("delegation_state", [])
            if isinstance(record, dict) and str(record.get("task_summary", "")).strip()
        ]
        return MemoryExtractionContext(
            agent_run_id=run.id,
            user_id=run.user_id,
            original_message=original,
            final_response=final_response,
            delegation_summaries=summaries,
            existing_memory=existing_memory,
            eligible_agent_ids=sorted(eligible_agent_ids, key=str),
            explicit_memory_request=bool(_EXPLICIT_MEMORY.search(original)),
        )

    async def _existing_memory(self, user_id: int, agent_ids: set[UUID]):
        existing = await self._memory.list(user_id=user_id, scope=MemoryScope.USER_GLOBAL)
        for agent_id in agent_ids:
            existing.extend(
                await self._memory.list(
                    user_id=user_id,
                    scope=MemoryScope.AGENT_PRIVATE,
                    agent_id=agent_id,
                )
            )
        return existing

    @staticmethod
    def _eligible_agent_ids(run: AgentRun) -> set[UUID]:
        agent_ids = {run.starting_agent_id}
        for record in run.checkpoint.get("delegation_state", []):
            if not isinstance(record, dict):
                continue
            for field in ("source", "target"):
                identity = record.get(field)
                if not isinstance(identity, dict) or identity.get("kind") not in {"primary", "user"}:
                    continue
                try:
                    agent_ids.add(UUID(str(identity.get("subject_id"))))
                except (TypeError, ValueError):
                    continue
        return agent_ids

    @staticmethod
    def _with_explicit_signal(
        candidate: MemoryCandidate, context: MemoryExtractionContext
    ) -> MemoryCandidate:
        return candidate.model_copy(
            update={
                "explicit_user_request": (
                    candidate.explicit_user_request or context.explicit_memory_request
                )
            }
        )
