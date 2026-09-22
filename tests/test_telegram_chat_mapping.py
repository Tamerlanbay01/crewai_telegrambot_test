from __future__ import annotations

import asyncio

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from bot.conversations import resolve_telegram_chat
from database.base import Base
from database.entities.chat import ChatEntity
from database.entities.user import UserEntity
from models.system_agent import UserAgentOverrideUpdate
from services.chat import ChatService
from services.system_agent import SystemAgentService


def test_telegram_conversation_reuses_its_application_chat() -> None:
    async def scenario() -> None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        try:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
            factory = async_sessionmaker(engine, expire_on_commit=False)
            async with factory() as session:
                session.add_all(
                    [
                        UserEntity(id=11, telegram_id=101),
                        UserEntity(id=22, telegram_id=202),
                    ]
                )
                await session.commit()

                first = await resolve_telegram_chat(
                    session,
                    user_id=11, telegram_chat_id=7001, title="First title"
                )
                repeated = await resolve_telegram_chat(
                    session,
                    user_id=11, telegram_chat_id=7001, title="Updated title"
                )
                another_conversation = await resolve_telegram_chat(
                    session,
                    user_id=11, telegram_chat_id=7002
                )
                another_user = await resolve_telegram_chat(
                    session,
                    user_id=22, telegram_chat_id=7001
                )

                assert first.id == repeated.id
                assert first.title == repeated.title == "First title"
                assert len({first.id, another_conversation.id, another_user.id}) == 3
                assert len(await ChatService(session).list_user_chats(11)) == 2
                assert await session.get(ChatEntity, first.id) is not None
        finally:
            await engine.dispose()

    asyncio.run(scenario())


def test_system_agent_listing_exposes_user_override_state() -> None:
    async def scenario() -> None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        try:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
            factory = async_sessionmaker(engine, expire_on_commit=False)
            async with factory() as session:
                session.add(UserEntity(id=11, telegram_id=101))
                await session.commit()
                service = SystemAgentService(session)
                await service.ensure_initial_templates()
                await service.set_override(
                    user_id=11,
                    key="calendar",
                    data=UserAgentOverrideUpdate(enabled=False),
                )

                configured = {agent.key: agent.enabled for agent in await service.list_for_user(user_id=11)}
                available = {agent.key for agent in await service.list_available_for_user(user_id=11)}
                assert configured == {"information": True, "mail": True, "calendar": False}
                assert available == {"information", "mail"}
        finally:
            await engine.dispose()

    asyncio.run(scenario())
