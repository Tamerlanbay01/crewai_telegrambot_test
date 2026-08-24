from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.entities.chat import ChatEntity
from models.chat import Chat, ChatCreate, ChatUpdate


class ChatRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_id(
        self,
        chat_id: UUID,
    ) -> Chat | None:
        entity = await self._get_entity_by_id(chat_id)

        if entity is None:
            return None

        return Chat.model_validate(entity)

    async def list_by_user(
        self,
        user_id: int,
    ) -> list[Chat]:
        result = await self._session.execute(
            select(ChatEntity)
            .where(ChatEntity.user_id == user_id)
            .order_by(ChatEntity.updated_at.desc())
        )

        entities = result.scalars().all()

        return [
            Chat.model_validate(entity)
            for entity in entities
        ]

    async def create(
        self,
        data: ChatCreate,
    ) -> Chat:
        entity = ChatEntity(
            **data.model_dump()
        )

        self._session.add(entity)

        await self._session.flush()
        await self._session.refresh(entity)

        return Chat.model_validate(entity)

    async def update(
        self,
        chat_id: UUID,
        data: ChatUpdate,
    ) -> Chat | None:
        entity = await self._get_entity_by_id(chat_id)

        if entity is None:
            return None

        changes = data.model_dump(
            exclude_unset=True
        )

        for field, value in changes.items():
            setattr(entity, field, value)

        await self._session.flush()
        await self._session.refresh(entity)

        return Chat.model_validate(entity)

    async def delete(
        self,
        chat_id: UUID,
    ) -> bool:
        entity = await self._get_entity_by_id(chat_id)

        if entity is None:
            return False

        await self._session.delete(entity)
        await self._session.flush()

        return True

    async def touch(
        self,
        chat_id: UUID,
    ) -> bool:
        entity = await self._get_entity_by_id(chat_id)

        if entity is None:
            return False

        from datetime import datetime, timezone

        entity.updated_at = datetime.now(timezone.utc)

        await self._session.flush()

        return True

    async def _get_entity_by_id(
        self,
        chat_id: UUID,
    ) -> ChatEntity | None:
        result = await self._session.execute(
            select(ChatEntity).where(
                ChatEntity.id == chat_id
            )
        )

        return result.scalar_one_or_none()