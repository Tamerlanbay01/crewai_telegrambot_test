from uuid import UUID

from aiogram import F, Router
from aiogram.types import CallbackQuery
from sqlalchemy.ext.asyncio import AsyncSession

from bot.handlers.common import report_error, resolve_internal_user, send_assistant_response
from services.assistant import AssistantService

router = Router(name="approvals")


@router.callback_query(F.data.startswith("approval:"))
async def decide_approval(callback: CallbackQuery, session: AsyncSession) -> None:
    parts = (callback.data or "").split(":", 2)
    if len(parts) != 3 or parts[1] not in {"approve", "reject"}:
        await callback.answer("This action is no longer valid.", show_alert=True)
        return
    try:
        approval_id = UUID(parts[2])
    except ValueError:
        await callback.answer("This action is no longer valid.", show_alert=True)
        return

    await callback.answer()
    try:
        user = await resolve_internal_user(session, callback.from_user)
        response = await AssistantService(session).resolve_approval(
            user_id=user.id,
            approval_id=approval_id,
            approve=parts[1] == "approve",
        )
    except Exception as error:
        await report_error(callback.message, error, "approval decision")
        return
    await send_assistant_response(callback.message, response)
