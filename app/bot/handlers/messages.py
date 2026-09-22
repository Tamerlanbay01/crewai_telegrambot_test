from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.types import Message
from sqlalchemy.ext.asyncio import AsyncSession

from bot.conversations import resolve_telegram_chat
from bot.handlers.common import (
    report_error,
    resolve_internal_user,
    send_assistant_response,
    send_text,
)
from services.agent import AgentService
from services.assistant import AssistantService

router = Router(name="messages")


@router.message(CommandStart())
async def start_command(message: Message, session: AsyncSession) -> None:
    if message.from_user is None:
        return
    try:
        user = await resolve_internal_user(session, message.from_user)
        await AgentService(session).ensure_primary_agent(user_id=user.id)
        await resolve_telegram_chat(
            session,
            user_id=user.id,
            telegram_chat_id=message.chat.id,
            title=message.from_user.first_name or "Telegram chat",
        )
    except Exception as error:
        await report_error(message, error, "start")
        return
    await send_text(
        message,
        "Welcome. Your assistant is ready; send a message to begin.",
    )


@router.message(Command("help"))
async def help_command(message: Message) -> None:
    await send_text(
        message,
        "Available commands:\n"
        "/start — set up your assistant\n"
        "/agents — manage agents\n"
        "/schedules — manage schedules\n"
        "/help — show this help\n"
        "/cancel — cancel the current setup",
    )


@router.message(F.text, StateFilter(None), ~F.text.startswith("/"))
async def handle_text_message(message: Message, session: AsyncSession) -> None:
    if message.from_user is None or message.text is None:
        return
    try:
        user = await resolve_internal_user(session, message.from_user)
        chat = await resolve_telegram_chat(
            session,
            user_id=user.id,
            telegram_chat_id=message.chat.id,
            title=message.from_user.first_name or "Telegram chat",
        )
        await message.bot.send_chat_action(
            chat_id=message.chat.id,
            action=ChatAction.TYPING,
        )
        response = await AssistantService(session).handle_message(
            user_id=user.id,
            chat_id=chat.id,
            text=message.text,
        )
    except Exception as error:
        await report_error(message, error, "assistant message")
        return
    await send_assistant_response(message, response)
