"""Persistence for per-agent memory visibility policy."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.entities.agent_memory_policy import AgentMemoryPolicyEntity
from models.memory import AgentMemoryPolicy, MemoryScope, PERSISTENT_AGENT_MEMORY_SCOPES


class AgentMemoryPolicyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_scopes(self, agent_id: UUID) -> list[MemoryScope] | None:
        rows = (
            await self._session.execute(
                select(AgentMemoryPolicyEntity).where(AgentMemoryPolicyEntity.agent_id == agent_id)
            )
        ).scalars().all()
        if not rows:
            return None  # Legacy agents retain their previous runtime behavior.
        return [MemoryScope(row.scope) for row in rows if row.enabled]

    async def set_scopes(self, agent_id: UUID, scopes: list[MemoryScope]) -> list[AgentMemoryPolicy]:
        rows = (
            await self._session.execute(
                select(AgentMemoryPolicyEntity).where(AgentMemoryPolicyEntity.agent_id == agent_id)
            )
        ).scalars().all()
        by_scope = {row.scope: row for row in rows}
        selected = set(scopes)
        result = []
        for scope in PERSISTENT_AGENT_MEMORY_SCOPES:
            row = by_scope.get(scope.value)
            if row is None:
                row = AgentMemoryPolicyEntity(agent_id=agent_id, scope=scope.value, enabled=scope in selected)
                self._session.add(row)
            else:
                row.enabled = scope in selected
            result.append(AgentMemoryPolicy(agent_id=agent_id, scope=scope, enabled=row.enabled))
        await self._session.flush()
        return result
