from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from aiogram import F, Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, Message
from sqlalchemy.ext.asyncio import AsyncSession

from bot.conversations import resolve_telegram_chat
from bot.handlers.common import report_error, resolve_internal_user, send_text
from bot.keyboards.schedules import (
    schedule_agent_keyboard,
    schedule_confirmation_keyboard,
    schedule_details_keyboard,
    schedule_type_keyboard,
    schedules_keyboard,
)
from models.schedule import Schedule, ScheduleType
from services.agent import AgentService
from services.schedule import ScheduleService

router = Router(name="schedules")


class ScheduleCreation(StatesGroup):
    name = State()
    agent = State()
    prompt = State()
    schedule_type = State()
    expression = State()
    timezone = State()
    confirmation = State()


def _next_run_label(schedule: Schedule) -> str:
    if schedule.next_run_at is None:
        return "not scheduled"
    try:
        local_time = schedule.next_run_at.astimezone(ZoneInfo(schedule.timezone))
    except (ValueError, KeyError):
        return schedule.next_run_at.isoformat()
    return local_time.strftime("%Y-%m-%d %H:%M %Z")


async def _show_schedules(target, session: AsyncSession, telegram_user) -> None:
    user = await resolve_internal_user(session, telegram_user)
    schedules = await ScheduleService(session).list(user_id=user.id)
    agents = await AgentService(session).list_active_agents(user.id)
    names = {agent.id: agent.name for agent in agents}
    if not schedules:
        text = "You have no schedules yet. Choose Create schedule to add one."
    else:
        lines = ["Your schedules", ""]
        for schedule in schedules:
            agent_name = names.get(schedule.agent_id, "unavailable agent")
            lines.append(
                f"• {schedule.name} — {agent_name}; {schedule.schedule_type.value}; "
                f"next: {_next_run_label(schedule)}; status: {schedule.status.value.lower()}"
            )
        text = "\n".join(lines)
    await send_text(target, text, reply_markup=schedules_keyboard(schedules))


@router.message(Command("schedules"))
async def list_schedules(message: Message, session: AsyncSession) -> None:
    if message.from_user is None:
        return
    try:
        await _show_schedules(message, session, message.from_user)
    except Exception as error:
        await report_error(message, error, "list schedules")


@router.callback_query(F.data == "schedule:list")
async def return_to_schedules(callback: CallbackQuery, session: AsyncSession) -> None:
    await callback.answer()
    try:
        await _show_schedules(callback.message, session, callback.from_user)
    except Exception as error:
        await report_error(callback.message, error, "list schedules")


@router.callback_query(F.data == "schedule:new")
async def begin_schedule_creation(
    callback: CallbackQuery,
    state: FSMContext,
    session: AsyncSession,
) -> None:
    if await state.get_state() is not None:
        await callback.answer("Finish or cancel the current setup first.", show_alert=True)
        return
    try:
        user = await resolve_internal_user(session, callback.from_user)
        agents = await AgentService(session).list_active_agents(user.id)
    except Exception as error:
        await report_error(callback.message, error, "start schedule setup")
        return
    if not agents:
        await callback.answer()
        await send_text(callback.message, "Create or activate an agent before adding a schedule.")
        return
    await callback.answer()
    await state.clear()
    await state.update_data(wizard_id=uuid4().hex[:8])
    await state.set_state(ScheduleCreation.name)
    await send_text(callback.message, "Send a name for the schedule. Use /cancel at any step.")


@router.message(ScheduleCreation.name, F.text & ~F.text.startswith("/"))
async def capture_schedule_name(message: Message, state: FSMContext, session: AsyncSession) -> None:
    value = (message.text or "").strip()
    if not value:
        await send_text(message, "The schedule name cannot be empty. Send a name.")
        return
    try:
        user = await resolve_internal_user(session, message.from_user)
        agents = await AgentService(session).list_active_agents(user.id)
    except Exception as error:
        await report_error(message, error, "load schedule agents")
        return
    if not agents:
        await state.clear()
        await send_text(message, "No active agent is available. Create or activate one first.")
        return
    await state.update_data(name=value)
    await state.set_state(ScheduleCreation.agent)
    data = await state.get_data()
    await send_text(
        message,
        "Choose the agent that should run this schedule.",
        reply_markup=schedule_agent_keyboard(agents, data["wizard_id"]),
    )


@router.callback_query(StateFilter(ScheduleCreation.agent), F.data.startswith("schedule:agent:"))
async def choose_schedule_agent(
    callback: CallbackQuery, state: FSMContext, session: AsyncSession
) -> None:
    parts = (callback.data or "").split(":")
    data = await state.get_data()
    if len(parts) != 4 or parts[3] != data.get("wizard_id"):
        await callback.answer("This setup has changed.", show_alert=True)
        return
    await callback.answer()
    try:
        agent_id = UUID(parts[2])
        user = await resolve_internal_user(session, callback.from_user)
        agent = await AgentService(session).get_agent(user_id=user.id, agent_id=agent_id)
    except Exception as error:
        await report_error(callback.message, error, "choose schedule agent")
        return
    await state.update_data(agent_id=str(agent.id), agent_name=agent.name)
    await state.set_state(ScheduleCreation.prompt)
    await send_text(callback.message, "Send the prompt this agent should run on each schedule.")


@router.message(ScheduleCreation.prompt, F.text & ~F.text.startswith("/"))
async def capture_schedule_prompt(message: Message, state: FSMContext) -> None:
    value = (message.text or "").strip()
    if not value:
        await send_text(message, "The prompt cannot be empty. Send a prompt.")
        return
    await state.update_data(prompt=value)
    await state.set_state(ScheduleCreation.schedule_type)
    data = await state.get_data()
    await send_text(
        message,
        "Choose the schedule type.",
        reply_markup=schedule_type_keyboard(data["wizard_id"]),
    )


@router.callback_query(
    StateFilter(ScheduleCreation.schedule_type), F.data.startswith("schedule:type:")
)
async def choose_schedule_type(callback: CallbackQuery, state: FSMContext) -> None:
    parts = (callback.data or "").split(":")
    data = await state.get_data()
    if len(parts) != 4 or parts[3] != data.get("wizard_id"):
        await callback.answer("This setup has changed.", show_alert=True)
        return
    raw_type = parts[2]
    try:
        schedule_type = ScheduleType(raw_type)
    except ValueError:
        await callback.answer("Choose one of the listed schedule types.", show_alert=True)
        return
    await callback.answer()
    await state.update_data(schedule_type=schedule_type.value)
    await state.set_state(ScheduleCreation.expression)
    prompt = {
        ScheduleType.ONCE: "Send an ISO date/time, such as 2026-09-22T15:00.",
        ScheduleType.INTERVAL: "Send a positive interval in seconds, such as 3600.",
        ScheduleType.CRON: "Send a five-field cron expression, such as 0 9 * * *.",
    }[schedule_type]
    await send_text(callback.message, prompt)


@router.message(ScheduleCreation.expression, F.text & ~F.text.startswith("/"))
async def capture_schedule_expression(message: Message, state: FSMContext) -> None:
    value = (message.text or "").strip()
    if not value:
        await send_text(message, "The expression cannot be empty. Send it again.")
        return
    await state.update_data(schedule_expression=value)
    await state.set_state(ScheduleCreation.timezone)
    await send_text(message, "Send an IANA timezone, such as Asia/Almaty, or /skip for Asia/Almaty.")


@router.message(ScheduleCreation.timezone, Command("skip"))
async def skip_schedule_timezone(message: Message, state: FSMContext) -> None:
    await _finish_schedule_timezone(message, state, "Asia/Almaty")


@router.message(ScheduleCreation.timezone, F.text & ~F.text.startswith("/"))
async def capture_schedule_timezone(message: Message, state: FSMContext) -> None:
    await _finish_schedule_timezone(message, state, (message.text or "").strip())


async def _finish_schedule_timezone(target, state: FSMContext, timezone: str) -> None:
    if not timezone:
        await send_text(target, "The timezone cannot be empty. Send a timezone or /skip.")
        return
    await state.update_data(timezone=timezone)
    data = await state.get_data()
    schedule_type = data["schedule_type"]
    wizard_id = data["wizard_id"]
    preview = (
        "Create this schedule?\n\n"
        f"Name: {data['name']}\n"
        f"Agent: {data.get('agent_name', 'selected agent')}\n"
        f"Prompt: {data['prompt']}\n"
        f"Type: {schedule_type}\n"
        f"Expression: {data['schedule_expression']}\n"
        f"Timezone: {timezone}"
    )
    await state.set_state(ScheduleCreation.confirmation)
    await send_text(target, preview, reply_markup=schedule_confirmation_keyboard(wizard_id))


@router.callback_query(
    StateFilter(ScheduleCreation.confirmation),
    F.data.startswith("schedule:create:confirm:"),
)
async def confirm_schedule_creation(
    callback: CallbackQuery,
    state: FSMContext,
    session: AsyncSession,
) -> None:
    parts = (callback.data or "").split(":")
    data = await state.get_data()
    if len(parts) != 4 or parts[3] != data.get("wizard_id"):
        await callback.answer("This setup has changed.", show_alert=True)
        return
    required = (
        "wizard_id",
        "name",
        "agent_id",
        "prompt",
        "schedule_type",
        "schedule_expression",
        "timezone",
    )
    if any(not data.get(key) for key in required):
        await state.clear()
        await callback.answer("This setup is incomplete; start again with /schedules.", show_alert=True)
        return
    await callback.answer()
    try:
        user = await resolve_internal_user(session, callback.from_user)
        chat = await resolve_telegram_chat(
            session,
            user_id=user.id,
            telegram_chat_id=callback.message.chat.id,
            title=callback.from_user.first_name or "Telegram chat",
        )
        schedule = await ScheduleService(session).create(
            user_id=user.id,
            agent_id=UUID(data["agent_id"]),
            name=data["name"],
            prompt=data["prompt"],
            schedule_type=ScheduleType(data["schedule_type"]),
            schedule_expression=data["schedule_expression"],
            timezone=data["timezone"],
            chat_id=chat.id,
        )
    except Exception as error:
        await report_error(callback.message, error, "create schedule")
        return
    await state.clear()
    await send_text(
        callback.message,
        f"Schedule {schedule.name} was created. Next run: {_next_run_label(schedule)}.",
    )


@router.callback_query(
    StateFilter(ScheduleCreation.confirmation),
    F.data.startswith("schedule:edit:expression:"),
)
async def edit_schedule_expression(callback: CallbackQuery, state: FSMContext) -> None:
    parts = (callback.data or "").split(":")
    data = await state.get_data()
    if len(parts) != 4 or parts[3] != data.get("wizard_id"):
        await callback.answer("This setup has changed.", show_alert=True)
        return
    if any(not data.get(key) for key in ("name", "agent_id", "prompt", "schedule_type")):
        await state.clear()
        await callback.answer("This setup is incomplete; start again with /schedules.", show_alert=True)
        return
    await callback.answer()
    await state.set_state(ScheduleCreation.expression)
    await send_text(callback.message, "Send the corrected schedule expression.")


@router.callback_query(
    StateFilter(ScheduleCreation.confirmation),
    F.data.startswith("schedule:edit:timezone:"),
)
async def edit_schedule_timezone(callback: CallbackQuery, state: FSMContext) -> None:
    parts = (callback.data or "").split(":")
    data = await state.get_data()
    if len(parts) != 4 or parts[3] != data.get("wizard_id"):
        await callback.answer("This setup has changed.", show_alert=True)
        return
    if any(not data.get(key) for key in ("name", "agent_id", "prompt", "schedule_type")):
        await state.clear()
        await callback.answer("This setup is incomplete; start again with /schedules.", show_alert=True)
        return
    await callback.answer()
    await state.set_state(ScheduleCreation.timezone)
    await send_text(callback.message, "Send a corrected IANA timezone, such as Asia/Almaty.")


@router.callback_query(
    StateFilter(ScheduleCreation.agent, ScheduleCreation.schedule_type, ScheduleCreation.confirmation),
    F.data.startswith("schedule:create:cancel:"),
)
async def cancel_schedule_creation(callback: CallbackQuery, state: FSMContext) -> None:
    parts = (callback.data or "").split(":")
    data = await state.get_data()
    if len(parts) != 4 or parts[3] != data.get("wizard_id"):
        await callback.answer("This setup has changed.", show_alert=True)
        return
    await state.clear()
    await callback.answer("Setup cancelled.")
    await send_text(callback.message, "Schedule setup cancelled.")


@router.callback_query(F.data.startswith("schedule:show:"))
async def show_schedule(callback: CallbackQuery, session: AsyncSession) -> None:
    await callback.answer()
    try:
        schedule_id = UUID((callback.data or "").rsplit(":", 1)[-1])
        user = await resolve_internal_user(session, callback.from_user)
        service = ScheduleService(session)
        schedule = await service.get(user_id=user.id, schedule_id=schedule_id)
        agents = await AgentService(session).list_active_agents(user.id)
        agent_name = next((item.name for item in agents if item.id == schedule.agent_id), "unavailable agent")
    except Exception as error:
        await report_error(callback.message, error, "show schedule")
        return
    detail = (
        f"{schedule.name}\n"
        f"Agent: {agent_name}\n"
        f"Type: {schedule.schedule_type.value}\n"
        f"Expression: {schedule.schedule_expression}\n"
        f"Next run: {_next_run_label(schedule)}\n"
        f"Status: {schedule.status.value.lower()}"
    )
    await send_text(callback.message, detail, reply_markup=schedule_details_keyboard(schedule))


@router.callback_query(F.data.startswith("schedule:"))
async def update_schedule(callback: CallbackQuery, session: AsyncSession) -> None:
    parts = (callback.data or "").split(":", 2)
    if len(parts) != 3 or parts[1] not in {"enable", "disable", "archive"}:
        await callback.answer("This action is no longer valid.", show_alert=True)
        return
    try:
        schedule_id = UUID(parts[2])
    except ValueError:
        await callback.answer("This action is no longer valid.", show_alert=True)
        return
    await callback.answer()
    try:
        user = await resolve_internal_user(session, callback.from_user)
        service = ScheduleService(session)
        if parts[1] == "enable":
            schedule = await service.enable(user_id=user.id, schedule_id=schedule_id)
        elif parts[1] == "disable":
            schedule = await service.disable(user_id=user.id, schedule_id=schedule_id)
        else:
            schedule = await service.archive(user_id=user.id, schedule_id=schedule_id)
    except Exception as error:
        await report_error(callback.message, error, "update schedule")
        return
    await send_text(
        callback.message,
        f"Schedule {schedule.name} is now {schedule.status.value.lower()}.",
        reply_markup=schedule_details_keyboard(schedule),
    )
