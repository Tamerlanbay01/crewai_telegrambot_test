from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from models.message import Message
from services.chat import ChatService


class AssistantService:
    def __init__(
        self,
        session: AsyncSession,
    ):
        self._session = session
        self._chats = ChatService(session)

    async def handle_message(
        self,
        *,
        chat_id: UUID,
        text: str,
    ) -> Message:
        await self._chats.add_message(
            chat_id=chat_id,
            role="user",
            content=text,
        )

        response_text = (
            "Обработка сообщения пока не реализована."
        )

        return await self._chats.add_message(
            chat_id=chat_id,
            role="assistant",
            content=response_text,
        )