from aiogram import Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot.handlers import agents, approvals, messages, schedules
from bot.handlers.common import send_text

router = Router(name="telegram")


@router.message(Command("cancel"))
async def cancel_flow(message: Message, state: FSMContext) -> None:
    if await state.get_state() is None:
        await send_text(message, "There is no active setup to cancel.")
        return
    await state.clear()
    await send_text(message, "Setup cancelled.")


router.include_router(approvals.router)
router.include_router(agents.router)
router.include_router(schedules.router)
router.include_router(messages.router)
