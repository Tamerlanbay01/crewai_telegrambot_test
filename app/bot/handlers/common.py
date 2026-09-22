from __future__ import annotations

import logging
from typing import Any

from aiogram.types import Message, User as TelegramUser
from sqlalchemy.ext.asyncio import AsyncSession

from models.assistant import AssistantResponse, AssistantResponseStatus
from models.user import User
from services.approval import ApprovalAlreadyProcessedError
from services.user import UserService

logger = logging.getLogger(__name__)

GENERIC_FAILURE = "Something went wrong while processing the request."
ALREADY_PROCESSED = "This action has already been processed."
NOT_AVAILABLE = "That item is not available or no longer exists."


async def resolve_internal_user(session: AsyncSession, telegram_user: TelegramUser) -> User:
    return await UserService(session).get_or_create(
        telegram_id=telegram_user.id,
        username=telegram_user.username,
        first_name=telegram_user.first_name,
        last_name=telegram_user.last_name,
    )


def split_telegram_message(text: str, max_units: int = 4096) -> list[str]:
    """Split plain text without dropping characters or breaking Unicode code points."""
    if max_units <= 0:
        raise ValueError("max_units must be positive")
    if not text:
        return []

    chunks: list[str] = []
    remaining = text
    while _utf16_units(remaining) > max_units:
        end = _end_within_limit(remaining, max_units)
        if end == 0:
            raise ValueError("A Unicode character exceeds the message chunk limit")
        window = remaining[:end]
        newline = window.rfind("\n")
        whitespace = window.rfind(" ")
        boundary = max(newline, whitespace)
        if boundary > 0:
            end = boundary + 1
        chunks.append(remaining[:end])
        remaining = remaining[end:]
    if remaining:
        chunks.append(remaining)
    return chunks


def _utf16_units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _end_within_limit(text: str, max_units: int) -> int:
    units = 0
    for index, character in enumerate(text):
        character_units = 2 if ord(character) > 0xFFFF else 1
        if units + character_units > max_units:
            return index
        units += character_units
    return len(text)


async def send_text(
    target: Message | Any | None,
    text: str,
    *,
    reply_markup: Any | None = None,
) -> None:
    if target is None:
        return
    chunks = split_telegram_message(text)
    for index, chunk in enumerate(chunks):
        options = {}
        if reply_markup is not None and index == len(chunks) - 1:
            options["reply_markup"] = reply_markup
        await target.answer(chunk, **options)


async def send_assistant_response(
    target: Message | Any | None,
    response: AssistantResponse,
) -> None:
    if response.status == AssistantResponseStatus.COMPLETED:
        content = response.message.content if response.message else response.content
        await send_text(target, content or "The assistant returned an empty response.")
        return
    if response.status == AssistantResponseStatus.WAITING_APPROVAL:
        from bot.keyboards.approvals import approval_keyboard

        text = "The assistant is asking permission to perform an action.\n\n"
        text += response.approval_summary or "Please review the requested action."
        markup = approval_keyboard(response.approval_id) if response.approval_id else None
        await send_text(target, text, reply_markup=markup)
        return
    await send_text(target, response.error or "The assistant could not complete this request.")


def expected_error_message(error: Exception) -> str | None:
    if isinstance(error, ApprovalAlreadyProcessedError):
        return ALREADY_PROCESSED
    if isinstance(error, LookupError):
        return NOT_AVAILABLE
    if not isinstance(error, ValueError):
        return None

    reason = str(error)
    safe_messages = {
        "Primary agent cannot be archived": "The primary agent cannot be archived.",
        "ONCE expression must be an ISO datetime": "Enter a date and time in ISO format, such as 2026-09-22T15:00.",
        "ONCE expression is a nonexistent local time in the selected timezone": "That local time does not exist in the selected timezone.",
        "INTERVAL expression must be a positive number of seconds": "Enter a positive interval in seconds, such as 3600.",
        "CRON expression must be a valid five-field cron expression": "Enter a valid five-field cron expression, such as 0 9 * * *.",
        "Could not calculate the next CRON occurrence": "The next run could not be calculated; check the cron expression.",
        "Schedule timezone cannot be empty": "Enter a timezone such as Asia/Almaty.",
        "Archived schedules cannot be updated": "Archived schedules cannot be changed.",
        "Cannot edit a schedule while it is being executed": "This schedule is running and cannot be changed yet.",
        "Schedule type cannot be empty": "Select a schedule type.",
    }
    if reason in safe_messages:
        return safe_messages[reason]
    if reason.startswith("Invalid timezone:"):
        return "Enter a valid IANA timezone, such as Asia/Almaty."
    if reason.endswith("cannot be empty"):
        return "Please provide a non-empty value."
    return "That value could not be used. Please check it and try again."


async def report_error(target: Message | Any | None, error: Exception, operation: str) -> None:
    user_message = expected_error_message(error)
    if user_message is None:
        logger.exception("Telegram request failed during %s", operation)
        user_message = GENERIC_FAILURE
    await send_text(target, user_message)
