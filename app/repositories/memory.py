"""Persistence and query operations for scoped memory records."""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from database.entities.memory import MemoryEntity
from models.memory import Memory, MemoryCreate, MemoryScope


class MemoryRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_id(self, memory_id: UUID) -> Memory | None:
        entity = (
            await self._session.execute(
                select(MemoryEntity).where(MemoryEntity.id == memory_id)
            )
        ).scalar_one_or_none()
        return self._to_model(entity) if entity is not None else None

    async def get_by_ids(self, memory_ids: list[UUID]) -> list[Memory]:
        if not memory_ids:
            return []
        entities = (
            await self._session.execute(
                select(MemoryEntity).where(MemoryEntity.id.in_(memory_ids))
            )
        ).scalars().all()
        return [self._to_model(entity) for entity in entities]

    async def get_by_key(
        self,
        *,
        user_id: int,
        scope: MemoryScope,
        key: str,
        agent_id: UUID | None = None,
    ) -> Memory | None:
        statement = select(MemoryEntity).where(
            MemoryEntity.user_id == user_id,
            MemoryEntity.scope == scope,
            MemoryEntity.key == key,
        )
        if scope == MemoryScope.AGENT_PRIVATE:
            statement = statement.where(MemoryEntity.agent_id == agent_id)
        entity = (await self._session.execute(statement)).scalar_one_or_none()
        return self._to_model(entity) if entity is not None else None

    async def upsert(self, data: MemoryCreate) -> Memory:
        statement = select(MemoryEntity).where(
            MemoryEntity.user_id == data.user_id,
            MemoryEntity.scope == data.scope,
            MemoryEntity.key == data.key,
        )
        if data.scope == MemoryScope.AGENT_PRIVATE:
            statement = statement.where(MemoryEntity.agent_id == data.agent_id)
        elif data.scope in {MemoryScope.CREW_SHARED, MemoryScope.RUN_EPHEMERAL}:
            statement = statement.where(MemoryEntity.run_id == data.run_id)
        entity = (await self._session.execute(statement)).scalar_one_or_none()
        if entity is None:
            entity = MemoryEntity(
                id=data.id,
                user_id=data.user_id,
                scope=data.scope,
                agent_id=data.agent_id,
                run_id=data.run_id,
                key=data.key,
                content=data.content,
                memory_metadata=data.metadata,
                expires_at=data.expires_at,
            )
            self._session.add(entity)
        else:
            entity.agent_id = data.agent_id
            entity.run_id = data.run_id
            entity.content = data.content
            entity.memory_metadata = data.metadata
            entity.expires_at = data.expires_at
        await self._session.flush()
        await self._session.refresh(entity)
        return self._to_model(entity)

    async def list_for_user(
        self, user_id: int, *, scope: MemoryScope | None = None
    ) -> list[Memory]:
        statement = select(MemoryEntity).where(MemoryEntity.user_id == user_id)
        if scope is not None:
            statement = statement.where(MemoryEntity.scope == scope)
        entities = (
            await self._session.execute(
                statement.order_by(MemoryEntity.updated_at.desc(), MemoryEntity.created_at.desc())
            )
        ).scalars().all()
        return [self._to_model(entity) for entity in entities]

    async def list_all(self) -> list[Memory]:
        entities = (
            await self._session.execute(
                select(MemoryEntity).order_by(
                    MemoryEntity.user_id,
                    MemoryEntity.updated_at.desc(),
                    MemoryEntity.created_at.desc(),
                )
            )
        ).scalars().all()
        return [self._to_model(entity) for entity in entities]

    async def list_for_agent(self, agent_id: UUID) -> list[Memory]:
        entities = (
            await self._session.execute(
                select(MemoryEntity)
                .where(MemoryEntity.agent_id == agent_id)
                .order_by(MemoryEntity.updated_at.desc(), MemoryEntity.created_at.desc())
            )
        ).scalars().all()
        return [self._to_model(entity) for entity in entities]

    async def list_for_run(self, run_id: UUID) -> list[Memory]:
        entities = (
            await self._session.execute(
                select(MemoryEntity)
                .where(MemoryEntity.run_id == run_id)
                .order_by(MemoryEntity.updated_at.desc(), MemoryEntity.created_at.desc())
            )
        ).scalars().all()
        return [self._to_model(entity) for entity in entities]

    async def delete(self, memory_id: UUID) -> bool:
        entity = (
            await self._session.execute(
                select(MemoryEntity).where(MemoryEntity.id == memory_id)
            )
        ).scalar_one_or_none()
        if entity is None:
            return False
        await self._session.delete(entity)
        await self._session.flush()
        return True

    async def delete_expired(self, *, now: datetime | None = None) -> int:
        result = await self._session.execute(
            delete(MemoryEntity).where(
                MemoryEntity.expires_at.is_not(None),
                MemoryEntity.expires_at <= (now or datetime.now(timezone.utc)),
            )
        )
        await self._session.flush()
        return result.rowcount or 0

    @staticmethod
    def _to_model(entity: MemoryEntity) -> Memory:
        return Memory(
            id=entity.id,
            user_id=entity.user_id,
            scope=entity.scope,
            memory_type=entity.memory_type,
            agent_id=entity.agent_id,
            run_id=entity.run_id,
            key=entity.key,
            content=entity.content,
            metadata=entity.memory_metadata,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            expires_at=entity.expires_at,
        )
