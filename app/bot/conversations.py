"""Map Telegram conversations to stable application chat identifiers."""

from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy.ext.asyncio import AsyncSession

from models.chat import Chat
from services.chat import ChatService


def telegram_conversation_id(*, user_id: int, telegram_chat_id: int) -> UUID:
    return uuid5(
        NAMESPACE_URL,
        f"telegram-conversation:{user_id}:{telegram_chat_id}",
    )


async def resolve_telegram_chat(
    session: AsyncSession,
    *,
    user_id: int,
    telegram_chat_id: int,
    title: str = "Telegram chat",
) -> Chat:
    return await ChatService(session).ensure_chat(
        chat_id=telegram_conversation_id(
            user_id=user_id,
            telegram_chat_id=telegram_chat_id,
        ),
        user_id=user_id,
        title=title,
    )
