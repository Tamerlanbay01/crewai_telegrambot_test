from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from models.chat import Chat, ChatCreate, ChatUpdate
from models.message import Message, MessageCreate
from repositories.chat import ChatRepository
from repositories.message import MessageRepository


class ChatService:
    def __init__(self, session: AsyncSession):
        self._session = session
        self._chats = ChatRepository(session)
        self._messages = MessageRepository(session)

    async def create_chat(
        self,
        *,
        user_id: int,
        title: str = "Новый чат",
    ) -> Chat:
        chat = await self._chats.create(
            ChatCreate(
                user_id=user_id,
                title=title,
            )
        )

        await self._session.commit()

        return chat

    async def get_chat(
        self,
        chat_id: UUID,
    ) -> Chat:
        chat = await self._chats.get_by_id(
            chat_id
        )

        if chat is None:
            raise LookupError(
                f"Chat not found: {chat_id}"
            )

        return chat

    async def list_user_chats(
        self,
        user_id: int,
    ) -> list[Chat]:
        return await self._chats.list_by_user(
            user_id
        )

    async def rename_chat(
        self,
        *,
        chat_id: UUID,
        title: str,
    ) -> Chat:
        clean_title = title.strip()

        if not clean_title:
            raise ValueError(
                "Chat title cannot be empty"
            )

        chat = await self._chats.update(
            chat_id,
            ChatUpdate(
                title=clean_title,
            ),
        )

        if chat is None:
            raise LookupError(
                f"Chat not found: {chat_id}"
            )

        await self._session.commit()

        return chat

    async def delete_chat(
        self,
        chat_id: UUID,
    ) -> None:
        deleted = await self._chats.delete(
            chat_id
        )

        if not deleted:
            raise LookupError(
                f"Chat not found: {chat_id}"
            )

        await self._session.commit()

    async def add_message(
        self,
        *,
        chat_id: UUID,
        role: str,
        content: str,
    ) -> Message:
        chat = await self._chats.get_by_id(
            chat_id
        )

        if chat is None:
            raise LookupError(
                f"Chat not found: {chat_id}"
            )

        message = await self._messages.create(
            MessageCreate(
                chat_id=chat_id,
                role=role,
                content=content,
            )
        )

        await self._chats.touch(
            chat_id
        )

        await self._session.commit()

        return message

    async def list_messages(
        self,
        chat_id: UUID,
    ) -> list[Message]:
        chat = await self._chats.get_by_id(
            chat_id
        )

        if chat is None:
            raise LookupError(
                f"Chat not found: {chat_id}"
            )

        return await self._messages.list_by_chat(
            chat_id
        )