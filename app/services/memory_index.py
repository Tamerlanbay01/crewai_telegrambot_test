"""Explicit repair path for rebuilding Qdrant from canonical PostgreSQL memory."""

from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from integrations.memory.factory import create_semantic_memory_dependencies
from integrations.memory.index import EmbeddingProvider, MemorySearchIndex
from models.memory import Memory, MemoryScope
from repositories.memory import MemoryRepository


class MemoryIndexService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        search_index: MemorySearchIndex | None = None,
    ):
        self._memories = MemoryRepository(session)
        if embedding_provider is None and search_index is None:
            embedding_provider, search_index = create_semantic_memory_dependencies()
        self._embeddings = embedding_provider
        self._index = search_index

    async def reindex_user(self, *, user_id: int) -> int:
        return await self._reindex(await self._memories.list_for_user(user_id))

    async def reindex_all(self) -> int:
        return await self._reindex(await self._memories.list_all())

    async def _reindex(self, memories: list[Memory]) -> int:
        if self._embeddings is None or self._index is None:
            return 0
        now = datetime.now(timezone.utc)
        eligible = [
            memory
            for memory in memories
            if memory.scope in {MemoryScope.USER_GLOBAL, MemoryScope.AGENT_PRIVATE}
            and not self._is_expired(memory, now)
        ]
        if not eligible:
            return 0
        await self._index.ensure_collection()
        vectors = await self._embeddings.embed_many(
            [self._embedding_text(memory) for memory in eligible]
        )
        if len(vectors) != len(eligible):
            raise ValueError("Embedding provider returned the wrong number of vectors")
        for memory, vector in zip(eligible, vectors, strict=True):
            await self._index.upsert(memory, vector)
        return len(eligible)

    @staticmethod
    def _embedding_text(memory: Memory) -> str:
        return (
            f"type: {memory.memory_type.value}\n"
            f"key: {memory.key}\n"
            f"content: {memory.content}"
        )

    @staticmethod
    def _is_expired(memory: Memory, now: datetime) -> bool:
        if memory.expires_at is None:
            return False
        expires_at = memory.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return expires_at <= now
