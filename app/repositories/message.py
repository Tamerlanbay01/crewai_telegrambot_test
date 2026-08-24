from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.entities.message import MessageEntity
from models.message import Message, MessageCreate


class MessageRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_id(
        self,
        message_id: UUID,
    ) -> Message | None:
        entity = await self._get_entity_by_id(message_id)

        if entity is None:
            return None

        return Message.model_validate(entity)

    async def list_by_chat(
        self,
        chat_id: UUID,
    ) -> list[Message]:
        result = await self._session.execute(
            select(MessageEntity)
            .where(MessageEntity.chat_id == chat_id)
            .order_by(MessageEntity.created_at.asc())
        )

        return [
            Message.model_validate(entity)
            for entity in result.scalars().all()
        ]

    async def create(
        self,
        data: MessageCreate,
    ) -> Message:
        entity = MessageEntity(
            **data.model_dump()
        )

        self._session.add(entity)

        await self._session.flush()
        await self._session.refresh(entity)

        return Message.model_validate(entity)

    async def delete(
        self,
        message_id: UUID,
    ) -> bool:
        entity = await self._get_entity_by_id(
            message_id
        )

        if entity is None:
            return False

        await self._session.delete(entity)
        await self._session.flush()

        return True

    async def _get_entity_by_id(
        self,
        message_id: UUID,
    ) -> MessageEntity | None:
        result = await self._session.execute(
            select(MessageEntity).where(
                MessageEntity.id == message_id
            )
        )

        return result.scalar_one_or_none()