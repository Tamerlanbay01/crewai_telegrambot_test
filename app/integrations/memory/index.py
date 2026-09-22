"""Provider-neutral contracts for semantic memory."""

from datetime import datetime
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel

from models.memory import Memory, MemoryScope


class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> list[float]:
        ...

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        ...


class MemorySearchHit(BaseModel):
    memory_id: UUID
    score: float


class MemorySearchIndex(Protocol):
    async def ensure_collection(self) -> None:
        ...

    async def upsert(self, memory: Memory, vector: list[float]) -> None:
        ...

    async def delete(self, memory_id: UUID) -> None:
        ...

    async def search(
        self,
        vector: list[float],
        *,
        user_id: int,
        scopes: list[MemoryScope],
        agent_id: UUID | None,
        limit: int,
        now: datetime,
    ) -> list[MemorySearchHit]:
        ...
