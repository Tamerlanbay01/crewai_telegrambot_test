"""Authorization and consolidation rules for Memory Agent proposals."""

import re
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import config
from models.agent import AgentStatus
from models.memory import (
    Memory,
    MemoryCandidate,
    MemoryCandidateDecision,
    MemoryScope,
)
from repositories.agent import AgentRepository
from repositories.memory import MemoryRepository
from services.memory import MemoryService

_TRIVIAL = {"ok", "okay", "thanks", "thank you", "спасибо", "привет", "hello", "hi"}


class MemoryPolicyService:
    """The sole application authority that may turn a candidate into a memory write."""

    def __init__(self, session: AsyncSession, *, memory: MemoryService | None = None):
        self._memories = MemoryRepository(session)
        self._agents = AgentRepository(session)
        self._memory = memory or MemoryService(session)

    async def apply(
        self,
        *,
        user_id: int,
        candidate: MemoryCandidate,
        eligible_agent_ids: set[UUID],
        commit: bool = True,
    ) -> tuple[MemoryCandidateDecision, Memory | None]:
        decision = await self.decide(
            user_id=user_id,
            candidate=candidate,
            eligible_agent_ids=eligible_agent_ids,
        )
        memory: Memory | None = None
        if decision in {MemoryCandidateDecision.SAVED, MemoryCandidateDecision.UPDATED}:
            memory = await self._memory.put(
                user_id=user_id,
                scope=candidate.scope,
                memory_type=candidate.memory_type,
                agent_id=candidate.agent_id,
                key=candidate.key.strip(),
                content=candidate.content.strip(),
                commit=commit,
            )
        return decision, memory

    async def decide(
        self,
        *,
        user_id: int,
        candidate: MemoryCandidate,
        eligible_agent_ids: set[UUID],
    ) -> MemoryCandidateDecision:
        if not self._is_well_formed(candidate):
            return MemoryCandidateDecision.REJECTED
        threshold = config.memory_min_confidence
        if candidate.explicit_user_request:
            threshold *= 0.8
        if candidate.confidence < threshold:
            return MemoryCandidateDecision.IGNORED
        if candidate.scope not in {MemoryScope.USER_GLOBAL, MemoryScope.AGENT_PRIVATE}:
            return MemoryCandidateDecision.REJECTED
        if candidate.scope == MemoryScope.USER_GLOBAL and candidate.agent_id is not None:
            return MemoryCandidateDecision.REJECTED
        if candidate.scope == MemoryScope.AGENT_PRIVATE:
            if candidate.agent_id is None or candidate.agent_id not in eligible_agent_ids:
                return MemoryCandidateDecision.REJECTED
            agent = await self._agents.get_by_id(candidate.agent_id)
            if agent is None or agent.user_id != user_id or agent.status != AgentStatus.ACTIVE:
                return MemoryCandidateDecision.REJECTED

        key = candidate.key.strip()
        existing = await self._memories.get_by_key(
            user_id=user_id,
            scope=candidate.scope,
            key=key,
            agent_id=candidate.agent_id,
        )
        if existing is not None:
            if self._fingerprint(existing.content) == self._fingerprint(candidate.content):
                return MemoryCandidateDecision.IGNORED
            return MemoryCandidateDecision.UPDATED

        for memory in await self._memories.list_for_user(user_id):
            if memory.scope != candidate.scope or memory.agent_id != candidate.agent_id:
                continue
            if self._fingerprint(memory.content) == self._fingerprint(candidate.content):
                return MemoryCandidateDecision.MERGED
        return MemoryCandidateDecision.SAVED

    @staticmethod
    def _is_well_formed(candidate: MemoryCandidate) -> bool:
        content = candidate.content.strip()
        normalized = re.sub(r"[^\w\s]", "", content.casefold()).strip()
        return bool(candidate.key.strip() and content and normalized not in _TRIVIAL)

    @staticmethod
    def _fingerprint(content: str) -> str:
        return re.sub(r"\s+", " ", content).strip().casefold()
