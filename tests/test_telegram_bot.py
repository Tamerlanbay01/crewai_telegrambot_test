from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from bot.handlers import agents as agent_handlers
from bot.handlers import approvals as approval_handlers
from bot.handlers import messages as message_handlers
from bot.handlers import schedules as schedule_handlers
from bot.handlers.common import (
    ALREADY_PROCESSED,
    send_assistant_response,
    send_text,
    split_telegram_message,
)
from bot.middlewares.database import DatabaseSessionMiddleware
from bot.keyboards.schedules import schedule_type_keyboard
from models.assistant import AssistantResponse, AssistantResponseStatus
from models.agent import AgentKind
from models.schedule import ScheduleStatus, ScheduleType
from services.approval import ApprovalAlreadyProcessedError


class FakeMessage:
    def __init__(self, text: str | None = None, *, user_id: int = 900, chat_id: int = 501) -> None:
        self.text = text
        self.from_user = SimpleNamespace(
            id=user_id,
            username="telegram_user",
            first_name="Ada",
            last_name="Lovelace",
        )
        self.chat = SimpleNamespace(id=chat_id)
        self.bot = SimpleNamespace(send_chat_action=AsyncMock())
        self.answer = AsyncMock()


class FakeCallback:
    def __init__(self, data: str, *, user_id: int = 900, chat_id: int = 501) -> None:
        self.data = data
        self.from_user = SimpleNamespace(
            id=user_id,
            username="telegram_user",
            first_name="Ada",
            last_name="Lovelace",
        )
        self.message = FakeMessage(user_id=user_id, chat_id=chat_id)
        self.answer = AsyncMock()


class FakeState:
    def __init__(self) -> None:
        self.current = None
        self.data: dict[str, object] = {}

    async def get_state(self):
        return self.current

    async def set_state(self, state) -> None:
        self.current = state

    async def update_data(self, **values) -> None:
        self.data.update(values)

    async def get_data(self):
        return dict(self.data)

    async def clear(self) -> None:
        self.current = None
        self.data.clear()


def test_start_resolves_telegram_user_and_ensures_primary_and_stable_chat() -> None:
    async def scenario() -> None:
        telegram_message = FakeMessage("/start", user_id=98765, chat_id=501)
        internal_user = SimpleNamespace(id=38)
        primary = SimpleNamespace(ensure_primary_agent=AsyncMock())
        resolve_chat = AsyncMock()
        session = object()
        with (
            patch.object(message_handlers, "resolve_internal_user", AsyncMock(return_value=internal_user)),
            patch.object(message_handlers, "AgentService", return_value=primary),
            patch.object(message_handlers, "resolve_telegram_chat", resolve_chat),
        ):
            await message_handlers.start_command(telegram_message, session)

        primary.ensure_primary_agent.assert_awaited_once_with(user_id=38)
        resolve_chat.assert_awaited_once_with(
            session,
            user_id=38,
            telegram_chat_id=501,
            title="Ada",
        )
        assert "assistant is ready" in telegram_message.answer.await_args.args[0]

    asyncio.run(scenario())


def test_text_message_uses_assistant_response_and_internal_user_id() -> None:
    async def scenario() -> None:
        telegram_message = FakeMessage("Explain asyncio", user_id=98765, chat_id=502)
        internal_user = SimpleNamespace(id=38)
        chat = SimpleNamespace(id=uuid4())
        response = AssistantResponse(
            status=AssistantResponseStatus.COMPLETED,
            run_id=uuid4(),
            content="Persisted answer",
        )
        agent_service = SimpleNamespace(ensure_primary_agent=AsyncMock())
        resolve_chat = AsyncMock(return_value=chat)
        assistant = SimpleNamespace(handle_message=AsyncMock(return_value=response))
        session = object()
        with (
            patch.object(message_handlers, "resolve_internal_user", AsyncMock(return_value=internal_user)),
            patch.object(message_handlers, "AgentService", return_value=agent_service),
            patch.object(message_handlers, "resolve_telegram_chat", resolve_chat),
            patch.object(message_handlers, "AssistantService", return_value=assistant),
        ):
            await message_handlers.handle_text_message(telegram_message, session)

        telegram_message.bot.send_chat_action.assert_awaited_once()
        agent_service.ensure_primary_agent.assert_not_awaited()
        assistant.handle_message.assert_awaited_once_with(
            user_id=38,
            chat_id=chat.id,
            text="Explain asyncio",
        )
        assert resolve_chat.await_count == 1
        telegram_message.answer.assert_awaited_once_with("Persisted answer")

    asyncio.run(scenario())


def test_waiting_approval_keyboard_contains_only_action_and_uuid() -> None:
    async def scenario() -> None:
        target = FakeMessage()
        approval_id = uuid4()
        response = AssistantResponse(
            status=AssistantResponseStatus.WAITING_APPROVAL,
            run_id=uuid4(),
            approval_id=approval_id,
            approval_summary="Action: create_event\nResource: calendar:event\nArguments: {}",
        )
        await send_assistant_response(target, response)

        text, options = target.answer.await_args.args[0], target.answer.await_args.kwargs
        assert "calendar:event" in text
        keyboard = options["reply_markup"].inline_keyboard[0]
        callbacks = [button.callback_data for button in keyboard]
        assert callbacks == [
            f"approval:approve:{approval_id}",
            f"approval:reject:{approval_id}",
        ]
        assert all(len(value) <= 64 for value in callbacks)
        assert all("arguments" not in value and "user" not in value for value in callbacks)

    asyncio.run(scenario())


def test_duplicate_approval_callback_shows_neutral_message_without_runtime_work() -> None:
    async def scenario() -> None:
        callback = FakeCallback(f"approval:approve:{uuid4()}")
        internal_user = SimpleNamespace(id=38)
        assistant = SimpleNamespace(
            resolve_approval=AsyncMock(side_effect=ApprovalAlreadyProcessedError("approved"))
        )
        with (
            patch.object(approval_handlers, "resolve_internal_user", AsyncMock(return_value=internal_user)),
            patch.object(approval_handlers, "AssistantService", return_value=assistant),
        ):
            await approval_handlers.decide_approval(callback, object())

        callback.answer.assert_awaited_once()
        assistant.resolve_approval.assert_awaited_once_with(
            user_id=38,
            approval_id=UUID(callback.data.rsplit(":", 1)[-1]),
            approve=True,
        )
        callback.message.answer.assert_awaited_once_with(ALREADY_PROCESSED)

    asyncio.run(scenario())


def test_agent_wizard_defers_creation_until_confirmation() -> None:
    async def scenario() -> None:
        state = FakeState()
        callback = FakeCallback("agent:new")
        message = FakeMessage("Python Helper")
        service = SimpleNamespace(
            create_user_agent=AsyncMock(return_value=SimpleNamespace(name="Python Helper"))
        )
        internal_user = SimpleNamespace(id=38)
        with (
            patch.object(agent_handlers, "AgentService", return_value=service),
            patch.object(agent_handlers, "resolve_internal_user", AsyncMock(return_value=internal_user)),
        ):
            await agent_handlers.begin_agent_creation(callback, state)
            await agent_handlers.capture_agent_name(message, state)
            await agent_handlers.capture_agent_role(FakeMessage("Python specialist"), state)
            await agent_handlers.capture_agent_goal(FakeMessage("Explain Python"), state)
            await agent_handlers.capture_agent_backstory(FakeMessage("Be concise"), state)
            await agent_handlers.choose_agent_spawn_permission(
                FakeCallback(f"agent:spawn:yes:{state.data['wizard_id']}"), state
            )
            service.create_user_agent.assert_not_awaited()

            await agent_handlers.confirm_agent_creation(
                FakeCallback(f"agent:create:confirm:{state.data['wizard_id']}"), state, object()
            )

        service.create_user_agent.assert_awaited_once_with(
            user_id=38,
            name="Python Helper",
            role="Python specialist",
            goal="Explain Python",
            backstory="Be concise",
            can_spawn_subagents=True,
        )
        assert state.current is None

    asyncio.run(scenario())


def test_primary_agent_archive_error_is_presented_safely() -> None:
    async def scenario() -> None:
        callback = FakeCallback(f"agent:archive:{uuid4()}")
        service = SimpleNamespace(
            archive_agent=AsyncMock(side_effect=ValueError("Primary agent cannot be archived"))
        )
        with (
            patch.object(agent_handlers, "resolve_internal_user", AsyncMock(return_value=SimpleNamespace(id=38))),
            patch.object(agent_handlers, "AgentService", return_value=service),
        ):
            await agent_handlers.archive_agent(callback, object())

        callback.message.answer.assert_awaited_once_with("The primary agent cannot be archived.")

    asyncio.run(scenario())


def test_agents_list_separates_owned_agents_and_system_agent_state() -> None:
    async def scenario() -> None:
        message = FakeMessage("/agents", user_id=98765)
        primary = SimpleNamespace(id=uuid4(), name="Primary", kind=AgentKind.PRIMARY)
        personal = SimpleNamespace(id=uuid4(), name="My Researcher", kind=AgentKind.USER)
        system_agents = [
            SimpleNamespace(key="information", name="Information", enabled=True),
            SimpleNamespace(key="calendar", name="Calendar", enabled=False),
        ]
        agents = SimpleNamespace(list_active_agents=AsyncMock(return_value=[primary, personal]))
        systems = SimpleNamespace(list_for_user=AsyncMock(return_value=system_agents))
        with (
            patch.object(agent_handlers, "resolve_internal_user", AsyncMock(return_value=SimpleNamespace(id=38))),
            patch.object(agent_handlers, "AgentService", return_value=agents),
            patch.object(agent_handlers, "SystemAgentService", return_value=systems),
        ):
            await agent_handlers.list_agents(message, object())

        agents.list_active_agents.assert_awaited_once_with(38)
        systems.list_for_user.assert_awaited_once_with(user_id=38)
        text = message.answer.await_args.args[0]
        assert "Primary" in text
        assert "My Researcher" in text
        assert "information — Information (enabled)" in text
        assert "calendar — Calendar (disabled)" in text

    asyncio.run(scenario())


def test_cancel_button_clears_agent_wizard_state() -> None:
    async def scenario() -> None:
        state = FakeState()
        state.current = agent_handlers.AgentCreation.confirmation
        state.data = {"name": "unfinished", "wizard_id": "old-flow"}
        callback = FakeCallback("agent:create:cancel:old-flow")
        await agent_handlers.cancel_agent_creation(callback, state)
        assert state.current is None
        assert state.data == {}
        callback.message.answer.assert_awaited_once_with("Agent setup cancelled.")

    asyncio.run(scenario())


def test_stale_agent_cancel_button_cannot_clear_a_new_wizard() -> None:
    async def scenario() -> None:
        state = FakeState()
        state.current = agent_handlers.AgentCreation.confirmation
        state.data = {"wizard_id": "new-flow", "name": "Current agent"}
        callback = FakeCallback("agent:create:cancel:old-flow")
        await agent_handlers.cancel_agent_creation(callback, state)
        assert state.current == agent_handlers.AgentCreation.confirmation
        assert state.data["name"] == "Current agent"
        callback.answer.assert_awaited_once_with("This setup has changed.", show_alert=True)
        callback.message.answer.assert_not_awaited()

    asyncio.run(scenario())


def test_schedule_is_created_only_by_confirmation_handler() -> None:
    async def scenario() -> None:
        schedule_id = uuid4()
        chat_id = uuid4()
        callback = FakeCallback("schedule:create:confirm:flow1234")
        state = FakeState()
        state.data = {
            "wizard_id": "flow1234",
            "agent_id": str(uuid4()),
            "name": "Daily News",
            "prompt": "Summarize today's news",
            "schedule_type": ScheduleType.CRON.value,
            "schedule_expression": "0 9 * * *",
            "timezone": "Asia/Almaty",
        }
        internal_user = SimpleNamespace(id=38)
        telegram_chat = SimpleNamespace(id=chat_id)
        schedule = SimpleNamespace(name="Daily News", next_run_at=None)
        schedule_service = SimpleNamespace(create=AsyncMock(return_value=schedule))
        with (
            patch.object(schedule_handlers, "resolve_internal_user", AsyncMock(return_value=internal_user)),
            patch.object(schedule_handlers, "resolve_telegram_chat", AsyncMock(return_value=telegram_chat)),
            patch.object(schedule_handlers, "ScheduleService", return_value=schedule_service),
        ):
            await schedule_handlers.confirm_schedule_creation(callback, state, object())

        schedule_service.create.assert_awaited_once()
        assert schedule_service.create.await_args.kwargs["user_id"] == 38
        assert schedule_service.create.await_args.kwargs["chat_id"] == chat_id
        assert schedule_service.create.await_args.kwargs["schedule_type"] == ScheduleType.CRON
        assert state.current is None

    asyncio.run(scenario())


def test_invalid_schedule_expression_stays_in_wizard_and_shows_safe_validation() -> None:
    async def scenario() -> None:
        callback = FakeCallback("schedule:create:confirm:flow1234")
        state = FakeState()
        state.current = "schedule.confirmation"
        state.data = {
            "wizard_id": "flow1234",
            "agent_id": str(uuid4()),
            "name": "Daily News",
            "prompt": "Summarize news",
            "schedule_type": ScheduleType.CRON.value,
            "schedule_expression": "bad cron",
            "timezone": "Asia/Almaty",
        }
        chat = SimpleNamespace(id=uuid4())
        schedule_service = SimpleNamespace(
            create=AsyncMock(
                side_effect=ValueError(
                    "CRON expression must be a valid five-field cron expression"
                )
            )
        )
        with (
            patch.object(schedule_handlers, "resolve_internal_user", AsyncMock(return_value=SimpleNamespace(id=38))),
            patch.object(schedule_handlers, "resolve_telegram_chat", AsyncMock(return_value=chat)),
            patch.object(schedule_handlers, "ScheduleService", return_value=schedule_service),
        ):
            await schedule_handlers.confirm_schedule_creation(callback, state, object())

        callback.message.answer.assert_awaited_once_with(
            "Enter a valid five-field cron expression, such as 0 9 * * *."
        )
        assert state.current == "schedule.confirmation"

    asyncio.run(scenario())


def test_schedule_list_uses_current_internal_user_scope() -> None:
    async def scenario() -> None:
        message = FakeMessage("/schedules", user_id=98765)
        internal_user = SimpleNamespace(id=38)
        schedules = [
            SimpleNamespace(
                id=uuid4(),
                name="Own schedule",
                agent_id=uuid4(),
                schedule_type=ScheduleType.ONCE,
                next_run_at=None,
                status=SimpleNamespace(value="ACTIVE"),
                timezone="Asia/Almaty",
            )
        ]
        schedule_service = SimpleNamespace(list=AsyncMock(return_value=schedules))
        agents = SimpleNamespace(list_active_agents=AsyncMock(return_value=[]))
        with (
            patch.object(schedule_handlers, "resolve_internal_user", AsyncMock(return_value=internal_user)),
            patch.object(schedule_handlers, "ScheduleService", return_value=schedule_service),
            patch.object(schedule_handlers, "AgentService", return_value=agents),
        ):
            await schedule_handlers.list_schedules(message, object())

        schedule_service.list.assert_awaited_once_with(user_id=38)
        assert "Own schedule" in message.answer.await_args.args[0]

    asyncio.run(scenario())


def test_foreign_schedule_is_reported_as_unavailable() -> None:
    async def scenario() -> None:
        callback = FakeCallback(f"schedule:show:{uuid4()}")
        service = SimpleNamespace(get=AsyncMock(side_effect=LookupError("foreign schedule")))
        with (
            patch.object(schedule_handlers, "resolve_internal_user", AsyncMock(return_value=SimpleNamespace(id=38))),
            patch.object(schedule_handlers, "ScheduleService", return_value=service),
        ):
            await schedule_handlers.show_schedule(callback, object())
        callback.message.answer.assert_awaited_once_with("That item is not available or no longer exists.")

    asyncio.run(scenario())


def test_schedule_actions_call_the_matching_user_scoped_service() -> None:
    async def scenario() -> None:
        for action, status in (
            ("enable", ScheduleStatus.ACTIVE),
            ("disable", ScheduleStatus.DISABLED),
            ("archive", ScheduleStatus.ARCHIVED),
        ):
            schedule_id = uuid4()
            callback = FakeCallback(f"schedule:{action}:{schedule_id}")
            schedule = SimpleNamespace(
                id=schedule_id,
                name="Daily News",
                status=status,
            )
            service = SimpleNamespace(
                enable=AsyncMock(return_value=schedule),
                disable=AsyncMock(return_value=schedule),
                archive=AsyncMock(return_value=schedule),
            )
            with (
                patch.object(
                    schedule_handlers,
                    "resolve_internal_user",
                    AsyncMock(return_value=SimpleNamespace(id=38)),
                ),
                patch.object(schedule_handlers, "ScheduleService", return_value=service),
            ):
                await schedule_handlers.update_schedule(callback, object())

            getattr(service, action).assert_awaited_once_with(user_id=38, schedule_id=schedule_id)
            callback.message.answer.assert_awaited_once()

    asyncio.run(scenario())


def test_stale_schedule_cancel_does_not_clear_the_current_flow() -> None:
    async def scenario() -> None:
        state = FakeState()
        state.current = schedule_handlers.ScheduleCreation.confirmation
        state.data = {"wizard_id": "new-flow", "name": "Current schedule"}
        callback = FakeCallback("schedule:create:cancel:old-flow")
        await schedule_handlers.cancel_schedule_creation(callback, state)
        assert state.current == schedule_handlers.ScheduleCreation.confirmation
        assert state.data["name"] == "Current schedule"
        callback.answer.assert_awaited_once_with("This setup has changed.", show_alert=True)
        callback.message.answer.assert_not_awaited()

    asyncio.run(scenario())


def test_message_splitting_preserves_unicode_and_all_content() -> None:
    text = "🙂" * 5000 + " end"
    chunks = split_telegram_message(text, max_units=4096)
    assert "".join(chunks) == text
    assert all(len(chunk.encode("utf-16-le")) // 2 <= 4096 for chunk in chunks)


def test_long_message_is_sent_in_full_without_markdown_processing() -> None:
    async def scenario() -> None:
        target = FakeMessage()
        text = ("<tag> _literal_ [plain] " * 300)
        await send_text(target, text)
        chunks = [call.args[0] for call in target.answer.await_args_list]
        assert "".join(chunks) == text
        assert all(len(chunk.encode("utf-16-le")) // 2 <= 4096 for chunk in chunks)

    asyncio.run(scenario())


def test_bot_transport_does_not_import_persistence_or_crewai_layers() -> None:
    bot_root = Path(__file__).parents[1] / "app" / "bot"
    forbidden = ("database.entities", "repositories", "crewai")
    violations: list[str] = []
    for path in bot_root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            module = None
            if isinstance(node, ast.Import):
                module = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = [node.module or ""]
            if module is not None:
                for imported in module:
                    if any(imported == prefix or imported.startswith(prefix + ".") for prefix in forbidden):
                        violations.append(f"{path}: {imported}")
    assert violations == []


def test_schedule_type_keyboard_offers_only_supported_types() -> None:
    callbacks = {
        button.callback_data
        for row in schedule_type_keyboard("flow1234").inline_keyboard
        for button in row
    }
    assert callbacks == {
        "schedule:type:ONCE:flow1234",
        "schedule:type:INTERVAL:flow1234",
        "schedule:type:CRON:flow1234",
        "schedule:create:cancel:flow1234",
    }


def test_schedule_confirm_router_filter_accepts_tokenized_keyboard_callback() -> None:
    async def scenario() -> None:
        handler = next(
            item
            for item in schedule_handlers.router.callback_query.handlers
            if item.callback.__name__ == "confirm_schedule_creation"
        )
        callback = SimpleNamespace(data="schedule:create:confirm:flow1234")
        assert await handler.filters[0].call(
            callback,
            raw_state=schedule_handlers.ScheduleCreation.confirmation.state,
        )
        assert handler.filters[1].callback(callback)
        stale_format = SimpleNamespace(data="schedule:create:confirm")
        assert not handler.filters[1].callback(stale_format)

    asyncio.run(scenario())


def test_session_middleware_closes_one_session_around_each_update() -> None:
    async def scenario() -> None:
        lifecycle = SimpleNamespace(entered=0, exited=0)
        session = object()

        class SessionContext:
            def __call__(self):
                return self

            async def __aenter__(self):
                lifecycle.entered += 1
                return session

            async def __aexit__(self, *_args):
                lifecycle.exited += 1

        async def handler(event, data):
            assert data["session"] is session
            return event

        middleware = DatabaseSessionMiddleware(SessionContext())
        result = await middleware(handler, "update", {})
        assert result == "update"
        assert lifecycle.entered == lifecycle.exited == 1

    asyncio.run(scenario())


def test_root_router_connects_all_transport_modules() -> None:
    from bot.router import router

    assert [child.name for child in router.sub_routers] == [
        "approvals",
        "agents",
        "schedules",
        "messages",
    ]


def test_cancel_command_clears_any_active_fsm_flow() -> None:
    async def scenario() -> None:
        from bot.router import cancel_flow

        state = FakeState()
        state.current = schedule_handlers.ScheduleCreation.expression
        state.data = {"wizard_id": "current"}
        message = FakeMessage("/cancel")
        await cancel_flow(message, state)
        assert state.current is None
        assert state.data == {}
        message.answer.assert_awaited_once_with("Setup cancelled.")

    asyncio.run(scenario())
