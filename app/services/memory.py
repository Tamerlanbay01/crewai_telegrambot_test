"""Tenant-aware application rules for persisted runtime memory."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import config
from integrations.memory.factory import create_semantic_memory_dependencies
from integrations.memory.index import EmbeddingProvider, MemorySearchIndex
from models.agent import Agent
from models.agent_run import AgentRun
from models.memory import Memory, MemoryCreate, MemoryScope, MemoryType
from models.runtime import RuntimeMemoryItem
from repositories.agent import AgentRepository
from repositories.agent_run import AgentRunRepository
from repositories.memory import MemoryRepository

logger = logging.getLogger(__name__)


class MemoryService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        max_items: int | None = None,
        semantic_top_k: int | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        search_index: MemorySearchIndex | None = None,
    ):
        self._session = session
        self._memories = MemoryRepository(session)
        self._agents = AgentRepository(session)
        self._runs = AgentRunRepository(session)
        self._max_items = config.max_memory_items if max_items is None else max_items
        self._semantic_top_k = (
            config.memory_semantic_top_k if semantic_top_k is None else semantic_top_k
        )
        if embedding_provider is None and search_index is None:
            embedding_provider, search_index = create_semantic_memory_dependencies()
        self._embeddings = embedding_provider
        self._index = search_index
        if self._max_items < 0:
            raise ValueError("max_items cannot be negative")
        if self._semantic_top_k < 0:
            raise ValueError("semantic_top_k cannot be negative")

    async def put(
        self,
        *,
        user_id: int,
        scope: MemoryScope,
        memory_type: MemoryType = MemoryType.FACT,
        key: str,
        content: str,
        agent_id: UUID | None = None,
        run_id: UUID | None = None,
        metadata: dict[str, object] | None = None,
        expires_at: datetime | None = None,
        commit: bool = True,
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
                memory_type=memory_type,
                agent_id=agent_id,
                run_id=run_id,
                key=clean_key,
                content=content,
                metadata=metadata or {},
                expires_at=expires_at,
            )
        )
        if commit:
            await self._session.commit()
            await self._index_memory(memory)
        return memory

    async def sync_index(self, memory: Memory) -> None:
        """Best-effort index sync after an external transaction commits."""
        await self._index_memory(memory)

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
            await self._delete_from_index(memory)
        return deleted

    async def build_runtime_context(
        self,
        *,
        user_id: int,
        run_id: UUID,
        agent_id: UUID | None = None,
        query: str | None = None,
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
        semantic = await self.search_runtime_memory(
            user_id=user_id,
            query=query or "",
            run_id=run_id,
            agent_id=agent_id,
            limit=self._semantic_top_k,
        )
        return self._fuse_runtime_memory(semantic=semantic, recent=recent)

    async def search_runtime_memory(
        self,
        *,
        user_id: int,
        query: str,
        run_id: UUID,
        agent_id: UUID | None,
        limit: int | None = None,
    ) -> list[RuntimeMemoryItem]:
        """Return Qdrant suggestions only after canonical PostgreSQL authorization."""
        await self._require_run(user_id=user_id, run_id=run_id)
        if agent_id is not None:
            await self._require_agent(user_id=user_id, agent_id=agent_id)
        if not query.strip() or limit == 0 or self._embeddings is None or self._index is None:
            return []

        scopes = [MemoryScope.USER_GLOBAL]
        if agent_id is not None:
            scopes.append(MemoryScope.AGENT_PRIVATE)
        now = datetime.now(timezone.utc)
        try:
            vector = await self._embeddings.embed(query.strip())
            hits = await self._index.search(
                vector,
                user_id=user_id,
                scopes=scopes,
                agent_id=agent_id,
                limit=limit if limit is not None else self._semantic_top_k,
                now=now,
            )
            canonical = {
                memory.id: memory
                for memory in await self._memories.get_by_ids(
                    [hit.memory_id for hit in hits]
                )
            }
        except Exception:
            logger.exception("Semantic memory retrieval failed; using PostgreSQL recent-memory fallback")
            return []

        results: list[RuntimeMemoryItem] = []
        for hit in hits:
            memory = canonical.get(hit.memory_id)
            if memory is None or not self._is_semantically_visible(
                memory, user_id=user_id, agent_id=agent_id, now=now
            ):
                continue
            results.append(self._runtime_item(memory))
        return results

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

    async def _index_memory(self, memory: Memory) -> None:
        if not self._is_indexable(memory) or self._embeddings is None or self._index is None:
            return
        try:
            await self._index.ensure_collection()
            vector = await self._embeddings.embed(self._embedding_text(memory))
            await self._index.upsert(memory, vector)
        except Exception:
            logger.exception("Memory %s was committed but could not be indexed", memory.id)

    async def _delete_from_index(self, memory: Memory) -> None:
        if not self._is_indexable(memory) or self._index is None:
            return
        try:
            await self._index.delete(memory.id)
        except Exception:
            logger.exception("Memory %s was deleted from PostgreSQL but remains indexed", memory.id)

    def _fuse_runtime_memory(
        self, *, semantic: list[RuntimeMemoryItem], recent: list[Memory]
    ) -> list[RuntimeMemoryItem]:
        fused: list[RuntimeMemoryItem] = []
        seen_ids: set[UUID] = set()
        seen_keys: set[tuple[MemoryScope, str]] = set()
        for item in semantic:
            if item.memory_id in seen_ids or (item.scope, item.key) in seen_keys:
                continue
            fused.append(item)
            seen_ids.add(item.memory_id)
            seen_keys.add((item.scope, item.key))
        for memory in recent:
            if memory.id in seen_ids or (memory.scope, memory.key) in seen_keys:
                continue
            fused.append(self._runtime_item(memory))
            seen_ids.add(memory.id)
            seen_keys.add((memory.scope, memory.key))
        return fused[: self._max_items]

    @staticmethod
    def _embedding_text(memory: Memory) -> str:
        return (
            f"type: {memory.memory_type.value}\n"
            f"key: {memory.key}\n"
            f"content: {memory.content}"
        )

    @staticmethod
    def _is_indexable(memory: Memory) -> bool:
        return memory.scope in {MemoryScope.USER_GLOBAL, MemoryScope.AGENT_PRIVATE}

    @classmethod
    def _is_semantically_visible(
        cls, memory: Memory, *, user_id: int, agent_id: UUID | None, now: datetime
    ) -> bool:
        if memory.user_id != user_id or cls._is_expired(memory, now):
            return False
        if memory.scope == MemoryScope.USER_GLOBAL:
            return memory.agent_id is None and memory.run_id is None
        return (
            memory.scope == MemoryScope.AGENT_PRIVATE
            and agent_id is not None
            and memory.agent_id == agent_id
            and memory.run_id is None
        )

    @staticmethod
    def _runtime_item(memory: Memory) -> RuntimeMemoryItem:
        return RuntimeMemoryItem(
            memory_id=memory.id,
            scope=memory.scope,
            key=memory.key,
            content=memory.content,
        )

    @staticmethod
    def _sort_key(memory: Memory) -> tuple[float, str]:
        updated_at = memory.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        return updated_at.timestamp(), str(memory.id)
