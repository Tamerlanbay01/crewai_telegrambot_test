"""Tenant-aware application rules for persisted runtime memory."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import config
from models.agent import Agent
from models.agent_run import AgentRun
from models.memory import Memory, MemoryCreate, MemoryScope
from models.runtime import RuntimeMemoryItem
from repositories.agent import AgentRepository
from repositories.agent_run import AgentRunRepository
from repositories.memory import MemoryRepository


class MemoryService:
    def __init__(self, session: AsyncSession, *, max_items: int | None = None):
        self._session = session
        self._memories = MemoryRepository(session)
        self._agents = AgentRepository(session)
        self._runs = AgentRunRepository(session)
        self._max_items = config.max_memory_items if max_items is None else max_items
        if self._max_items < 0:
            raise ValueError("max_items cannot be negative")

    async def put(
        self,
        *,
        user_id: int,
        scope: MemoryScope,
        key: str,
        content: str,
        agent_id: UUID | None = None,
        run_id: UUID | None = None,
        metadata: dict[str, object] | None = None,
        expires_at: datetime | None = None,
    ) -> Memory:
        clean_key = key.strip()
        if not clean_key:
            raise ValueError("Memory key cannot be empty")
        if not content.strip():
            raise ValueError("Memory content cannot be empty")
        if agent_id is not None:
            await self._require_agent(user_id=user_id, agent_id=agent_id)
        if run_id is not None:
            await self._require_run(user_id=user_id, run_id=run_id)
        memory = await self._memories.upsert(
            MemoryCreate(
                user_id=user_id,
                scope=scope,
                agent_id=agent_id,
                run_id=run_id,
                key=clean_key,
                content=content,
                metadata=metadata or {},
                expires_at=expires_at,
            )
        )
        await self._session.commit()
        return memory

    async def get(self, *, user_id: int, memory_id: UUID) -> Memory | None:
        memory = await self._memories.get_by_id(memory_id)
        if memory is None:
            return None
        if memory.user_id != user_id:
            raise LookupError(f"Memory not found: {memory_id}")
        if self._is_expired(memory, datetime.now(timezone.utc)):
            return None
        return memory

    async def list(
        self,
        *,
        user_id: int,
        scope: MemoryScope | None = None,
        agent_id: UUID | None = None,
        run_id: UUID | None = None,
    ) -> list[Memory]:
        if agent_id is not None:
            await self._require_agent(user_id=user_id, agent_id=agent_id)
        if run_id is not None:
            await self._require_run(user_id=user_id, run_id=run_id)
        now = datetime.now(timezone.utc)
        return [
            memory
            for memory in await self._memories.list_for_user(user_id)
            if (scope is None or memory.scope == scope)
            and (agent_id is None or memory.agent_id == agent_id)
            and (run_id is None or memory.run_id == run_id)
            and not self._is_expired(memory, now)
        ]

    async def delete(self, *, user_id: int, memory_id: UUID) -> bool:
        memory = await self._memories.get_by_id(memory_id)
        if memory is None:
            return False
        if memory.user_id != user_id:
            raise LookupError(f"Memory not found: {memory_id}")
        deleted = await self._memories.delete(memory_id)
        if deleted:
            await self._session.commit()
        return deleted

    async def build_runtime_context(
        self,
        *,
        user_id: int,
        run_id: UUID,
        agent_id: UUID | None = None,
    ) -> list[RuntimeMemoryItem]:
        await self._require_run(user_id=user_id, run_id=run_id)
        if agent_id is not None:
            await self._require_agent(user_id=user_id, agent_id=agent_id)

        candidates = await self._memories.list_for_user(
            user_id,
            scope=MemoryScope.USER_GLOBAL,
        )
        if agent_id is not None:
            candidates.extend(
                memory
                for memory in await self._memories.list_for_agent(agent_id)
                if memory.scope == MemoryScope.AGENT_PRIVATE
                and memory.user_id == user_id
            )
        candidates.extend(
            memory
            for memory in await self._memories.list_for_run(run_id)
            if memory.user_id == user_id
            and memory.scope in {MemoryScope.CREW_SHARED, MemoryScope.RUN_EPHEMERAL}
        )

        now = datetime.now(timezone.utc)
        recent = [memory for memory in candidates if not self._is_expired(memory, now)]
        recent.sort(key=self._sort_key, reverse=True)
        return [
            RuntimeMemoryItem(scope=memory.scope, key=memory.key, content=memory.content)
            for memory in recent[: self._max_items]
        ]

    async def _require_agent(self, *, user_id: int, agent_id: UUID) -> Agent:
        agent = await self._agents.get_by_id(agent_id)
        if agent is None or agent.user_id != user_id:
            raise LookupError(f"Agent not found: {agent_id}")
        return agent

    async def _require_run(self, *, user_id: int, run_id: UUID) -> AgentRun:
        run = await self._runs.get_by_id(run_id)
        if run is None or run.user_id != user_id:
            raise LookupError(f"Agent run not found: {run_id}")
        return run

    @staticmethod
    def _is_expired(memory: Memory, now: datetime) -> bool:
        expires_at = memory.expires_at
        if expires_at is None:
            return False
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return expires_at <= now

    @staticmethod
    def _sort_key(memory: Memory) -> tuple[float, str]:
        updated_at = memory.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        return updated_at.timestamp(), str(memory.id)
