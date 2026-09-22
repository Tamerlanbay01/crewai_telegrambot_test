from uuid import UUID, uuid4

from aiogram import F, Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, Message
from sqlalchemy.ext.asyncio import AsyncSession

from bot.handlers.common import report_error, resolve_internal_user, send_text
from bot.keyboards.agents import (
    agent_confirmation_keyboard,
    agent_details_keyboard,
    agent_spawn_keyboard,
    agents_keyboard,
)
from models.agent import AgentKind
from models.system_agent import UserAgentOverrideUpdate
from services.agent import AgentService
from services.system_agent import SystemAgentService

router = Router(name="agents")


class AgentCreation(StatesGroup):
    name = State()
    role = State()
    goal = State()
    backstory = State()
    spawn_permissions = State()
    confirmation = State()


async def _show_agents(target, session: AsyncSession, telegram_user) -> None:
    user = await resolve_internal_user(session, telegram_user)
    agents = await AgentService(session).list_active_agents(user.id)
    system_agents = await SystemAgentService(session).list_for_user(user_id=user.id)
    primary = [agent for agent in agents if agent.kind == AgentKind.PRIMARY]
    user_agents = [agent for agent in agents if agent.kind == AgentKind.USER]
    lines = ["Your agents", "", "Primary"]
    lines.extend(f"• {agent.name}" for agent in primary)
    lines.extend(["", "User agents"])
    lines.extend(f"• {agent.name}" for agent in user_agents)
    lines.extend(["", "System agents"])
    lines.extend(
        f"• {agent.key} — {agent.name} ({'enabled' if agent.enabled else 'disabled'})"
        for agent in system_agents
    )
    await send_text(
        target,
        "\n".join(lines),
        reply_markup=agents_keyboard(agents, system_agents),
    )


@router.message(Command("agents"))
async def list_agents(message: Message, session: AsyncSession) -> None:
    if message.from_user is None:
        return
    try:
        await _show_agents(message, session, message.from_user)
    except Exception as error:
        await report_error(message, error, "list agents")


@router.callback_query(F.data == "agent:list")
async def return_to_agents(callback: CallbackQuery, session: AsyncSession) -> None:
    await callback.answer()
    try:
        await _show_agents(callback.message, session, callback.from_user)
    except Exception as error:
        await report_error(callback.message, error, "list agents")


@router.callback_query(F.data == "agent:new")
async def begin_agent_creation(
    callback: CallbackQuery, state: FSMContext
) -> None:
    if await state.get_state() is not None:
        await callback.answer("Finish or cancel the current setup first.", show_alert=True)
        return
    await callback.answer()
    await state.clear()
    await state.update_data(wizard_id=uuid4().hex[:8])
    await state.set_state(AgentCreation.name)
    await send_text(callback.message, "Send a name for the new agent. Use /cancel at any step.")


@router.message(AgentCreation.name, F.text & ~F.text.startswith("/"))
async def capture_agent_name(message: Message, state: FSMContext) -> None:
    value = (message.text or "").strip()
    if not value:
        await send_text(message, "The name cannot be empty. Send an agent name.")
        return
    await state.update_data(name=value)
    await state.set_state(AgentCreation.role)
    await send_text(message, "What role should this agent have?")


@router.message(AgentCreation.role, F.text & ~F.text.startswith("/"))
async def capture_agent_role(message: Message, state: FSMContext) -> None:
    value = (message.text or "").strip()
    if not value:
        await send_text(message, "The role cannot be empty. Send a role.")
        return
    await state.update_data(role=value)
    await state.set_state(AgentCreation.goal)
    await send_text(message, "What is the agent's goal?")


@router.message(AgentCreation.goal, F.text & ~F.text.startswith("/"))
async def capture_agent_goal(message: Message, state: FSMContext) -> None:
    value = (message.text or "").strip()
    if not value:
        await send_text(message, "The goal cannot be empty. Send a goal.")
        return
    await state.update_data(goal=value)
    await state.set_state(AgentCreation.backstory)
    await send_text(message, "Send optional backstory/instructions, or /skip.")


@router.message(AgentCreation.backstory, Command("skip"))
async def skip_agent_backstory(message: Message, state: FSMContext) -> None:
    await state.update_data(backstory=None)
    await state.set_state(AgentCreation.spawn_permissions)
    data = await state.get_data()
    await send_text(
        message,
        "May this agent spawn subagents?",
        reply_markup=agent_spawn_keyboard(data["wizard_id"]),
    )


@router.message(AgentCreation.backstory, F.text & ~F.text.startswith("/"))
async def capture_agent_backstory(message: Message, state: FSMContext) -> None:
    await state.update_data(backstory=(message.text or "").strip() or None)
    await state.set_state(AgentCreation.spawn_permissions)
    data = await state.get_data()
    await send_text(
        message,
        "May this agent spawn subagents?",
        reply_markup=agent_spawn_keyboard(data["wizard_id"]),
    )


@router.callback_query(
    StateFilter(AgentCreation.spawn_permissions), F.data.startswith("agent:spawn:")
)
async def choose_agent_spawn_permission(
    callback: CallbackQuery, state: FSMContext
) -> None:
    parts = (callback.data or "").split(":", 3)
    if len(parts) != 4:
        await callback.answer("This action is no longer valid.", show_alert=True)
        return
    data = await state.get_data()
    if parts[3] != data.get("wizard_id"):
        await callback.answer("This setup has changed.", show_alert=True)
        return
    answer = parts[2]
    if answer not in {"yes", "no"}:
        await callback.answer("Choose Yes or No.", show_alert=True)
        return
    await callback.answer()
    can_spawn = answer == "yes"
    await state.update_data(can_spawn_subagents=can_spawn)
    await state.set_state(AgentCreation.confirmation)
    summary = (
        "Create this agent?\n\n"
        f"Name: {data.get('name', '')}\n"
        f"Role: {data.get('role', '')}\n"
        f"Goal: {data.get('goal', '')}\n"
        f"Backstory/instructions: {data.get('backstory') or 'None'}\n"
        f"Can spawn subagents: {'yes' if can_spawn else 'no'}"
    )
    await send_text(
        callback.message,
        summary,
        reply_markup=agent_confirmation_keyboard(data["wizard_id"]),
    )


@router.callback_query(
    StateFilter(AgentCreation.confirmation), F.data.startswith("agent:create:confirm:")
)
async def confirm_agent_creation(
    callback: CallbackQuery, state: FSMContext, session: AsyncSession
) -> None:
    parts = (callback.data or "").split(":", 3)
    data = await state.get_data()
    if len(parts) != 4 or parts[3] != data.get("wizard_id"):
        await callback.answer("This setup has changed.", show_alert=True)
        return
    required = ("name", "role", "goal", "wizard_id")
    if any(not data.get(key) for key in required):
        await state.clear()
        await callback.answer("This setup is incomplete; start again with /agents.", show_alert=True)
        return
    await callback.answer()
    try:
        user = await resolve_internal_user(session, callback.from_user)
        agent = await AgentService(session).create_user_agent(
            user_id=user.id,
            name=data["name"],
            role=data["role"],
            goal=data["goal"],
            backstory=data.get("backstory"),
            can_spawn_subagents=bool(data.get("can_spawn_subagents", False)),
        )
    except Exception as error:
        await report_error(callback.message, error, "create agent")
        return
    await state.clear()
    await send_text(callback.message, f"Agent {agent.name} was created.")


@router.callback_query(
    StateFilter(AgentCreation.spawn_permissions, AgentCreation.confirmation),
    F.data.startswith("agent:create:cancel:"),
)
async def cancel_agent_creation(callback: CallbackQuery, state: FSMContext) -> None:
    parts = (callback.data or "").split(":", 3)
    data = await state.get_data()
    if len(parts) != 4 or parts[3] != data.get("wizard_id"):
        await callback.answer("This setup has changed.", show_alert=True)
        return
    await state.clear()
    await callback.answer("Setup cancelled.")
    await send_text(callback.message, "Agent setup cancelled.")


@router.callback_query(F.data.startswith("agent:show:"))
async def show_agent(callback: CallbackQuery, session: AsyncSession) -> None:
    await callback.answer()
    try:
        agent_id = UUID((callback.data or "").rsplit(":", 1)[-1])
        user = await resolve_internal_user(session, callback.from_user)
        service = AgentService(session)
        agent = await service.get_agent(user_id=user.id, agent_id=agent_id)
        prompt = await service.get_current_prompt(user_id=user.id, agent_id=agent_id)
        detail = (
            f"{agent.name}\n"
            f"Type: {agent.kind.value}\n"
            f"Role: {prompt.role}\n"
            f"Goal: {prompt.goal}\n"
            f"Backstory: {prompt.backstory or 'None'}"
        )
    except Exception as error:
        await report_error(callback.message, error, "show agent")
        return
    await send_text(
        callback.message,
        detail,
        reply_markup=agent_details_keyboard(agent),
    )


@router.callback_query(F.data.startswith("agent:archive:"))
async def archive_agent(callback: CallbackQuery, session: AsyncSession) -> None:
    await callback.answer()
    try:
        agent_id = UUID((callback.data or "").rsplit(":", 1)[-1])
        user = await resolve_internal_user(session, callback.from_user)
        agent = await AgentService(session).archive_agent(
            user_id=user.id,
            agent_id=agent_id,
        )
    except Exception as error:
        await report_error(callback.message, error, "archive agent")
        return
    await send_text(callback.message, f"Agent {agent.name} was archived.")


@router.callback_query(F.data.startswith("system:"))
async def toggle_system_agent(callback: CallbackQuery, session: AsyncSession) -> None:
    parts = (callback.data or "").split(":", 2)
    if len(parts) != 3 or parts[1] not in {"enable", "disable"}:
        await callback.answer("This action is no longer valid.", show_alert=True)
        return
    await callback.answer()
    try:
        user = await resolve_internal_user(session, callback.from_user)
        enabled = parts[1] == "enable"
        await SystemAgentService(session).set_override(
            user_id=user.id,
            key=parts[2],
            data=UserAgentOverrideUpdate(enabled=enabled),
        )
        system_agents = await SystemAgentService(session).list_for_user(user_id=user.id)
        agents = await AgentService(session).list_active_agents(user.id)
        updated_agent = next(
            (item for item in system_agents if item.key == parts[2]),
            None,
        )
    except Exception as error:
        await report_error(callback.message, error, "update system agent")
        return
    await send_text(
        callback.message,
        f"System agent {parts[2]} is now "
        f"{'enabled' if updated_agent and updated_agent.enabled else 'disabled'}.",
        reply_markup=agents_keyboard(agents, system_agents),
    )


@router.callback_query(F.data.startswith("agent:create:"))
async def stale_agent_create_callback(callback: CallbackQuery) -> None:
    await callback.answer("This setup has changed.", show_alert=True)


@router.callback_query(F.data.startswith("agent:spawn:"))
async def stale_agent_spawn_callback(callback: CallbackQuery) -> None:
    await callback.answer("This setup has changed.", show_alert=True)
