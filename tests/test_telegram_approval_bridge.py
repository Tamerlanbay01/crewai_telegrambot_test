from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database.base import Base
from database.entities.user import UserEntity
from agents.assistant.crewai.runtime import CrewAIToolApprovalRuntime
from core.config import config
from database.base import Base
from database.entities.user import UserEntity
from models.approval import ApprovalStatus
from models.agent_run import AgentRunStatus
from models.permission import ActionClass, PermissionSubjectType
from models.runtime import (
    AgentRuntimeResult,
    AgentRuntimeStatus,
    DelegationDecision,
    DelegationType,
    RuntimeStep,
)
from models.tool import ToolExecutionResult, ToolExecutionStatus, ToolIntent, ToolRequest
from services.agent import AgentService
from services.agent_run import AgentRunService
from services.approval import (
    ApprovalAlreadyProcessedError,
    ApprovalService,
)
from services.assistant import AssistantService
from services.chat import ChatService
from services.agent_runtime import AgentRuntimeService
from services.permission import PermissionService
from services.system_agent import SystemAgentService
from services.tool_authority import FakeToolExecutor


async def _with_session(test) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            session.add(UserEntity(id=1, telegram_id=101))
            await session.commit()
            await test(session)
    finally:
        await engine.dispose()


async def _pending_approval(session):
    agent = await AgentService(session).ensure_primary_agent(user_id=1)
    chat = await ChatService(session).create_chat(user_id=1, title="Telegram")
    user_message = await ChatService(session).add_message(
        chat_id=chat.id, role="user", content="Create an event"
    )
    runs = AgentRunService(session)
    run = await runs.create_run(
        user_id=1,
        starting_agent_id=agent.id,
        model_name="test-model",
        chat_id=chat.id,
        message_id=user_message.id,
    )
    await runs.start(user_id=1, run_id=run.id)
    approval = await ApprovalService(session).request(
        user_id=1,
        run_id=run.id,
        flow_id="persisted-flow-id",
        requesting_subject_type=PermissionSubjectType.SYSTEM_AGENT,
        requesting_subject_id="calendar",
        action_name="create_event",
        action_class=ActionClass.WRITE,
        arguments={"title": "Team meeting"},
        resource="calendar:event",
    )
    return run, approval


def test_approval_is_resumed_once_and_final_answer_is_persisted() -> None:
    async def scenario() -> None:
        await _with_session(run_scenario)

    async def run_scenario(session) -> None:
        run, approval = await _pending_approval(session)
        resume_calls: list[tuple[int, object]] = []

        class FakeRuntimeService:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            async def resume(self, *, user_id: int, run_id) -> AgentRuntimeResult:
                resume_calls.append((user_id, run_id))
                return AgentRuntimeResult(
                    status=AgentRuntimeStatus.COMPLETED,
                    content="The event is ready.",
                )

        service = AssistantService(session, runtime=object())
        with patch("services.assistant.AgentRuntimeService", FakeRuntimeService):
            response = await service.resolve_approval(
                user_id=1,
                approval_id=approval.id,
                approve=True,
            )
            try:
                await service.resolve_approval(
                    user_id=1,
                    approval_id=approval.id,
                    approve=True,
                )
            except ApprovalAlreadyProcessedError:
                pass
            else:
                raise AssertionError("A repeated callback must be rejected")

        assert response.content == "The event is ready."
        assert response.message is not None
        assert response.message.content == "The event is ready."
        assert resume_calls == [(1, run.id)]
        assert (await ApprovalService(session).get(user_id=1, approval_id=approval.id)).status == ApprovalStatus.APPROVED
        assert (await AgentRunService(session).get_run(user_id=1, run_id=run.id)).status == AgentRunStatus.RUNNING

    asyncio.run(scenario())


def test_reject_resumes_runtime_and_returns_its_response() -> None:
    async def scenario() -> None:
        async def run_scenario(session) -> None:
            _run, approval = await _pending_approval(session)
            resumed = []

            class FakeRuntimeService:
                def __init__(self, *_args, **_kwargs) -> None:
                    pass

                async def resume(self, *, user_id: int, run_id) -> AgentRuntimeResult:
                    resumed.append((user_id, run_id))
                    return AgentRuntimeResult(
                        status=AgentRuntimeStatus.COMPLETED,
                        content="I did not create it; here is another option.",
                    )

            with patch("services.assistant.AgentRuntimeService", FakeRuntimeService):
                response = await AssistantService(session, runtime=object()).resolve_approval(
                    user_id=1,
                    approval_id=approval.id,
                    approve=False,
                )

            assert response.content == "I did not create it; here is another option."
            assert resumed == [(1, approval.run_id)]
            assert (await ApprovalService(session).get(user_id=1, approval_id=approval.id)).status == ApprovalStatus.REJECTED

        await _with_session(run_scenario)

    asyncio.run(scenario())


def test_pending_crewai_approval_resumes_after_runtime_reconstruction(tmp_path: Path) -> None:
    class FakeAuthority:
        def __init__(self) -> None:
            self.approval_id = uuid4()
            self.executions = 0

        async def request(self, request, *, flow_id, runtime_permission_scopes=None):
            assert request.user_id == 1
            return ToolExecutionResult(
                status=ToolExecutionStatus.WAITING_APPROVAL,
                approval_id=self.approval_id,
                flow_id=flow_id,
            )

        async def get_approval(self, *, user_id: int, approval_id):
            assert user_id == 1
            assert approval_id == self.approval_id
            return SimpleNamespace(status=ApprovalStatus.APPROVED)

        async def execute_approved(self, *, user_id: int, approval_id, runtime_permission_scopes=None):
            assert user_id == 1
            assert approval_id == self.approval_id
            self.executions += 1
            return ToolExecutionResult(
                status=ToolExecutionStatus.EXECUTED,
                output={"created": True},
            )

    async def scenario() -> None:
        authority = FakeAuthority()
        request = ToolRequest(
            run_id=uuid4(),
            user_id=1,
            requesting_subject_type=PermissionSubjectType.SYSTEM_AGENT,
            requesting_subject_id="calendar",
            name="external_action",
            action_class=ActionClass.EXTERNAL_SIDE_EFFECT,
            resource="calendar:event",
            arguments={"title": "Team meeting"},
        )
        path = tmp_path / "approval-flows.sqlite"
        first_runtime = CrewAIToolApprovalRuntime(
            authority=authority,
            persistence_path=path,
        )
        pending = await first_runtime.begin(request)
        del first_runtime

        restarted_runtime = CrewAIToolApprovalRuntime(
            authority=authority,
            persistence_path=path,
        )
        result = await restarted_runtime.resume(
            user_id=1,
            flow_id=pending.flow_id,
        )

        assert pending.status == ToolExecutionStatus.WAITING_APPROVAL
        assert result.status == ToolExecutionStatus.EXECUTED
        assert result.output == {"created": True}
        assert authority.executions == 1

    asyncio.run(scenario())


def test_approval_resumes_across_new_application_service_and_database_sessions(
    tmp_path: Path,
) -> None:
    class SequenceRuntime:
        def __init__(self, steps) -> None:
            self.steps = list(steps)

        async def run(self, _context, _request):
            return self.steps.pop(0)

    async def scenario() -> None:
        database_path = tmp_path / "application.sqlite"
        database_url = f"sqlite+aiosqlite:///{database_path.as_posix()}"
        engine = create_async_engine(database_url)
        try:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
            factory = async_sessionmaker(engine, expire_on_commit=False)
            flow_path = tmp_path / "persisted-flow.sqlite"
            executor = FakeToolExecutor()
            with patch.object(config.crewcfg, "approval_persistence_path", str(flow_path)):
                async with factory() as first_session:
                    first_session.add(UserEntity(id=1, telegram_id=101))
                    await first_session.commit()
                    await SystemAgentService(first_session).ensure_initial_templates()
                    await PermissionService(first_session).grant(
                        user_id=1,
                        subject_type=PermissionSubjectType.SYSTEM_AGENT,
                        subject_id="calendar",
                        action_class=ActionClass.EXTERNAL_SIDE_EFFECT,
                        resource_scope="calendar:*",
                    )
                    chat = await ChatService(first_session).create_chat(
                        user_id=1, title="Telegram chat"
                    )
                    pending_response = await AssistantService(
                        first_session,
                        runtime=SequenceRuntime(
                            [
                                RuntimeStep(
                                    decision=DelegationDecision(
                                        type=DelegationType.DELEGATE_SYSTEM_AGENT,
                                        target_id="calendar",
                                        task_summary="Prepare a planning event",
                                    )
                                ),
                                RuntimeStep(
                                    decision=DelegationDecision(type=DelegationType.RESPOND),
                                    tool_intent=ToolIntent(
                                        name="external_action",
                                        action_class=ActionClass.EXTERNAL_SIDE_EFFECT,
                                        resource="calendar:event",
                                        arguments={"title": "Team meeting"},
                                    ),
                                ),
                            ]
                        ),
                        tool_executor=executor,
                    ).handle_message(
                        user_id=1,
                        chat_id=chat.id,
                        text="Create a team meeting",
                    )
                    assert pending_response.status.value == "waiting_approval"
                    approval_id = pending_response.approval_id
                    run_id = pending_response.run_id

                await engine.dispose()
                engine = create_async_engine(database_url)
                factory = async_sessionmaker(engine, expire_on_commit=False)
                async with factory() as restarted_session:
                    restarted_assistant = AssistantService(
                        restarted_session,
                        runtime=SequenceRuntime(
                            [
                                RuntimeStep(
                                    decision=DelegationDecision(type=DelegationType.RESPOND),
                                    content="The calendar action completed.",
                                ),
                                RuntimeStep(
                                    decision=DelegationDecision(type=DelegationType.RESPOND),
                                    content="The event was created.",
                                )
                            ]
                        ),
                        tool_executor=executor,
                    )
                    response = await restarted_assistant.resolve_approval(
                        user_id=1,
                        approval_id=approval_id,
                        approve=True,
                    )
                    assert response.content == "The event was created."
                    assert response.message is not None
                    assert response.message.content == "The event was created."
                    assert len(executor.external_actions) == 1
                    assert executor.external_actions[0] == {"title": "Team meeting"}
                    assert (
                        await ApprovalService(restarted_session).get(
                            user_id=1, approval_id=approval_id
                        )
                    ).status == ApprovalStatus.APPROVED
                    assert (
                        await AgentRunService(restarted_session).get_run(
                            user_id=1, run_id=run_id
                        )
                    ).status == AgentRunStatus.COMPLETED
                    assert [
                        item.content
                        for item in await ChatService(restarted_session).list_messages(chat.id)
                        if item.role == "assistant"
                    ] == ["The event was created."]

                    try:
                        await restarted_assistant.resolve_approval(
                            user_id=1,
                            approval_id=approval_id,
                            approve=True,
                        )
                    except ApprovalAlreadyProcessedError:
                        pass
                    else:
                        raise AssertionError("A repeated callback must not resume again")
                    assert len(executor.external_actions) == 1
        finally:
            await engine.dispose()

    asyncio.run(scenario())


def test_runtime_uses_persistence_path_from_configuration(tmp_path: Path) -> None:
    expected_path = tmp_path / "configured-flows.sqlite"
    with patch.object(config.crewcfg, "approval_persistence_path", str(expected_path)):
        service = AgentRuntimeService(object(), runtime=object())
    assert service._approval_persistence_path == expected_path


def test_approval_redaction_covers_compound_and_camel_case_secret_keys() -> None:
    sanitized = ApprovalService._sanitize(
        {
            "access_token": "access-value",
            "client_secret": "client-value",
            "api-key": "api-value",
            "refreshToken": "refresh-value",
            "nested": {"Authorization": "header-value", "title": "visible"},
        }
    )
    assert sanitized == {
        "access_token": "[REDACTED]",
        "client_secret": "[REDACTED]",
        "api-key": "[REDACTED]",
        "refreshToken": "[REDACTED]",
        "nested": {"Authorization": "[REDACTED]", "title": "visible"},
    }
