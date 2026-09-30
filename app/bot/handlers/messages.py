from uuid import uuid4

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from sqlalchemy.ext.asyncio import AsyncSession

from bot.conversations import resolve_telegram_chat
from bot.handlers.agents import AgentDesignConfirmation
from bot.handlers.crews import CrewDesignConfirmation
from bot.handlers.common import (
    report_error,
    resolve_internal_user,
    send_assistant_response,
    send_text,
)
from bot.keyboards.agents import agent_design_keyboard
from bot.keyboards.crews import crew_design_keyboard
from services.agent import AgentService
from services.agent_design import AgentDesignService, design_request_kind
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
async def handle_text_message(
    message: Message, session: AsyncSession, state: FSMContext
) -> None:
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
        # Transitional bridge: Primary Agent runtime routing will own this decision.
        design_kind = design_request_kind(message.text)
        if design_kind == "crew":
            design = AgentDesignService(session)
            blueprint = await design.design_crew(user_id=user.id, user_request=message.text)
            preview = await design.crew_preview_text(user_id=user.id, blueprint=blueprint)
            wizard_id = uuid4().hex[:8]
            await state.update_data(
                wizard_id=wizard_id,
                blueprint=blueprint.model_dump(mode="json"),
            )
            await state.set_state(CrewDesignConfirmation.confirmation)
            can_create = not blueprint.missing_capabilities and all(
                not agent.missing_capabilities for agent in blueprint.agents
            )
            await send_text(
                message, preview,
                reply_markup=crew_design_keyboard(wizard_id, can_create=can_create),
            )
            return
        if design_kind == "agent":
            design = AgentDesignService(session)
            blueprint = await design.design_agent(user_id=user.id, user_request=message.text)
            preview = await design.preview_text(user_id=user.id, blueprint=blueprint)
            wizard_id = uuid4().hex[:8]
            await state.update_data(
                wizard_id=wizard_id,
                blueprint=blueprint.model_dump(mode="json"),
            )
            await state.set_state(AgentDesignConfirmation.confirmation)
            await send_text(message, preview, reply_markup=agent_design_keyboard(wizard_id))
            return
        response = await AssistantService(session).handle_message(
            user_id=user.id,
            chat_id=chat.id,
            text=message.text,
        )
    except Exception as error:
        await report_error(message, error, "assistant message")
        return
    await send_assistant_response(message, response)
