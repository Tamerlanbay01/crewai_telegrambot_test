"""Telegram confirmation transport for proposed user crews."""

from aiogram import F, Router
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery
from sqlalchemy.ext.asyncio import AsyncSession

from bot.handlers.common import report_error, resolve_internal_user, send_text
from models.agent_factory import CrewBlueprint
from services.crew import CrewService

router = Router(name="crews")


class CrewDesignConfirmation(StatesGroup):
    confirmation = State()


@router.callback_query(
    StateFilter(CrewDesignConfirmation.confirmation), F.data.startswith("crew:design:")
)
async def decide_crew_design(
    callback: CallbackQuery, state: FSMContext, session: AsyncSession
) -> None:
    parts = (callback.data or "").split(":", 3)
    data = await state.get_data()
    if len(parts) != 4 or parts[3] != data.get("wizard_id") or parts[2] not in {"confirm", "cancel"}:
        await callback.answer("This design has changed.", show_alert=True)
        return
    if parts[2] == "cancel":
        await state.clear()
        await callback.answer("Design cancelled.")
        await send_text(callback.message, "Crew design cancelled.")
        return
    try:
        blueprint = CrewBlueprint.model_validate(data["blueprint"])
        user = await resolve_internal_user(session, callback.from_user)
        crew = await CrewService(session).create_from_blueprint(
            user_id=user.id, blueprint=blueprint,
        )
    except (LookupError, PermissionError, ValueError) as error:
        await callback.answer(str(error)[:180], show_alert=True)
        return
    except Exception as error:
        await report_error(callback.message, error, "create crew")
        return
    await state.clear()
    await callback.answer()
    await send_text(callback.message, f"Crew {crew.name} was created.")


@router.callback_query(F.data.startswith("crew:design:"))
async def stale_crew_design(callback: CallbackQuery) -> None:
    await callback.answer("This design has changed.", show_alert=True)
