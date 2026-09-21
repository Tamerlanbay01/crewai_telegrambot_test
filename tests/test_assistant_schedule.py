from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
import unittest
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agents.assistant.crewai.factory import DynamicCrewAIFactory
from database.base import Base
from database.entities.agent import AgentEntity
from database.entities.agent_connection import AgentConnectionEntity
from database.entities.agent_prompt import AgentPromptVersionEntity
from database.entities.agent_run import AgentRunEntity
from database.entities.agent_run_event import AgentRunEventEntity
from database.entities.agent_skill import AgentSkillEntity
from database.entities.approval import ApprovalEntity
from database.entities.chat import ChatEntity
from database.entities.memory import MemoryEntity
from database.entities.message import MessageEntity
from database.entities.permission import PermissionEntity
from database.entities.schedule import ScheduleEntity
from database.entities.skill import SkillEntity
from database.entities.system_agent_template import SystemAgentTemplateEntity
from database.entities.user import UserEntity
from database.entities.user_agent_override import UserAgentOverrideEntity
from models.agent import AgentStatus, AgentUpdate
from models.agent_run import AgentRunStatus
from models.assistant import AssistantResponseStatus
from models.memory import MemoryScope
from models.permission import ActionClass, PermissionSubjectType
from models.runtime import (
    AgentRuntimeStatus,
    DelegationDecision,
    DelegationType,
    RuntimeStep,
)
from models.schedule import ScheduleStatus, ScheduleType, ScheduleUpdate
from models.skill import SkillOwnerType
from models.tool import ToolExecutionResult, ToolExecutionStatus, ToolIntent
from repositories.agent import AgentRepository
from repositories.schedule import ScheduleRepository
from services.agent import AgentService
from services.agent_run import AgentRunService
from services.approval import ApprovalService
from services.assistant import AssistantService
from services.chat import ChatService
from services.memory import MemoryService
from services.permission import PermissionService
from services.schedule import ScheduleService
from services.skill import SkillService
from services.system_agent import SystemAgentService
from services.tool_authority import BackendToolAuthority, FakeToolExecutor


async def _with_session(test: Callable[[AsyncSession], Awaitable[None]]) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            session.add_all(
                [
                    UserEntity(id=1, telegram_id=101),
                    UserEntity(id=2, telegram_id=202),
                ]
            )
            await session.commit()
            await test(session)
    finally:
        await engine.dispose()


class FakeRuntime:
    def __init__(self, steps: list[RuntimeStep] | None = None, error: Exception | None = None):
        self.steps = list(steps or [])
        self.error = error
        self.contexts = []
        self.requests = []

    async def run(self, context, request):
        self.contexts.append(context.model_copy(deep=True))
        self.requests.append(request.model_copy(deep=True))
        if self.error is not None:
            raise self.error
        return self.steps.pop(0)


class AuthorityApprovalRuntime:
    def __init__(self, session: AsyncSession):
        self._authority = BackendToolAuthority(session, executor=FakeToolExecutor())

    async def begin(self, request, *, runtime_permission_scopes=None):
        return await self._authority.request(
            request,
            flow_id="assistant-integration-test",
            runtime_permission_scopes=runtime_permission_scopes,
        )

    async def resume(self, *, user_id: int, flow_id: str) -> ToolExecutionResult:
        return ToolExecutionResult(
            status=ToolExecutionStatus.FAILED,
            error="Resume is not part of this test",
        )


def _respond(content: str) -> RuntimeStep:
    return RuntimeStep(
        decision=DelegationDecision(type=DelegationType.RESPOND),
        content=content,
    )


class AssistantScheduleIntegrationTests(unittest.TestCase):
    def test_completed_message_uses_bounded_chronological_history_once(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            chats = ChatService(session)
            chat = await chats.create_chat(user_id=1, title="Current")
            other_chat = await chats.create_chat(user_id=1, title="Other")
            foreign_chat = await chats.create_chat(user_id=2, title="Foreign")
            start = datetime.now(timezone.utc) - timedelta(days=2)
            session.add_all(
                [
                    MessageEntity(
                        id=uuid4(),
                        chat_id=chat.id,
                        role="user" if index % 2 == 0 else "assistant",
                        content=f"history-{index:02d}",
                        created_at=start + timedelta(seconds=index),
                    )
                    for index in range(22)
                ]
                + [
                    MessageEntity(
                        id=uuid4(), chat_id=other_chat.id, role="user", content="other-chat"
                    ),
                    MessageEntity(
                        id=uuid4(), chat_id=foreign_chat.id, role="user", content="foreign-user"
                    ),
                ]
            )
            await session.commit()

            primary = await AgentService(session).ensure_primary_agent(user_id=1)
            skills = SkillService(session)
            skill = await skills.create_user_skill(
                user_id=1,
                key="reporting",
                name="Reporting",
                description="Summarize evidence concisely.",
            )
            await skills.assign_to_agent(user_id=1, agent_id=primary.id, skill_id=skill.id)
            await MemoryService(session).put(
                user_id=1,
                scope=MemoryScope.USER_GLOBAL,
                key="preferred_language",
                content="Russian",
            )

            runtime = FakeRuntime([_respond("Итоговый ответ")])
            with self.assertRaises(LookupError):
                await AssistantService(session, runtime=runtime).handle_message(
                    user_id=1,
                    chat_id=foreign_chat.id,
                    text="Cannot access the other user's chat",
                )
            self.assertEqual(
                [item.content for item in await chats.list_messages(foreign_chat.id)],
                ["foreign-user"],
            )
            response = await AssistantService(session, runtime=runtime).handle_message(
                user_id=1,
                chat_id=chat.id,
                text="А что лучше использовать для миграций?",
            )
            self.assertEqual(response.status, AssistantResponseStatus.COMPLETED)
            self.assertEqual(response.message.content, "Итоговый ответ")
            self.assertEqual(runtime.requests[0].message, "А что лучше использовать для миграций?")
            history = runtime.contexts[0].chat_context
            self.assertEqual(len(history), 20)
            self.assertEqual(history[0].content, "history-02")
            self.assertEqual(history[-1].content, "history-21")
            self.assertNotIn(runtime.requests[0].message, [item.content for item in history])
            self.assertNotIn("other-chat", [item.content for item in history])
            self.assertNotIn("foreign-user", [item.content for item in history])
            task_description = DynamicCrewAIFactory._task_description(
                runtime.contexts[0], runtime.requests[0]
            )
            self.assertEqual(task_description.count(runtime.requests[0].message), 1)
            self.assertEqual([item.key for item in runtime.contexts[0].active_skills], ["reporting"])
            self.assertEqual([item.key for item in runtime.contexts[0].memory], ["preferred_language"])

            messages = await chats.list_messages(chat.id)
            self.assertEqual(sum(item.role == "user" and item.content == runtime.requests[0].message for item in messages), 1)
            self.assertEqual(
                sum(item.role == "assistant" and item.content == "Итоговый ответ" for item in messages),
                1,
            )
            runs = (await session.execute(select(AgentRunEntity))).scalars().all()
            self.assertEqual(len(runs), 1)
            user_message = next(item for item in messages if item.content == runtime.requests[0].message)
            self.assertEqual(runs[0].message_id, user_message.id)
            self.assertEqual(runs[0].starting_agent_id, primary.id)
            self.assertEqual(runs[0].chat_id, chat.id)
            self.assertEqual(runs[0].status, AgentRunStatus.COMPLETED)

        asyncio.run(_with_session(scenario))

    def test_waiting_approval_returns_persisted_sanitized_approval_without_answer(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            chat = await ChatService(session).create_chat(user_id=1)
            await AgentService(session).ensure_primary_agent(user_id=1)
            await SystemAgentService(session).ensure_initial_templates()
            await PermissionService(session).grant(
                user_id=1,
                subject_type=PermissionSubjectType.SYSTEM_AGENT,
                subject_id="calendar",
                action_class=ActionClass.EXTERNAL_SIDE_EFFECT,
                resource_scope="calendar:*",
            )
            runtime = FakeRuntime(
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
                            arguments={"title": "Planning", "api_key": "sensitive-value"},
                        ),
                    )
                ]
            )
            response = await AssistantService(
                session,
                runtime=runtime,
                approval_runtime=AuthorityApprovalRuntime(session),
            ).handle_message(user_id=1, chat_id=chat.id, text="Создай встречу")

            self.assertEqual(response.status, AssistantResponseStatus.WAITING_APPROVAL)
            self.assertIsNotNone(response.approval_id)
            self.assertIn("external_action", response.approval_summary)
            self.assertIn("calendar:event", response.approval_summary)
            self.assertIn("[REDACTED]", response.approval_summary)
            self.assertNotIn("sensitive-value", response.approval_summary)
            self.assertIsNone(response.message)
            messages = await ChatService(session).list_messages(chat.id)
            self.assertEqual([item.role for item in messages], ["user"])
            approval = await ApprovalService(session).get(
                user_id=1, approval_id=response.approval_id
            )
            self.assertEqual(approval.run_id, response.run_id)
            self.assertEqual(approval.status.value, "pending")
            self.assertEqual(approval.requesting_subject_type, PermissionSubjectType.SYSTEM_AGENT)
            self.assertEqual(approval.requesting_subject_id, "calendar")
            run = (await session.execute(select(AgentRunEntity))).scalar_one()
            self.assertEqual(run.status, AgentRunStatus.WAITING_APPROVAL)

        asyncio.run(_with_session(scenario))

    def test_failed_and_empty_runtime_results_are_structured_without_error_messages(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            chat = await ChatService(session).create_chat(user_id=1)
            failed = await AssistantService(
                session,
                runtime=FakeRuntime(error=RuntimeError("traceback: private database detail")),
            ).handle_message(user_id=1, chat_id=chat.id, text="Fail safely")
            self.assertEqual(failed.status, AssistantResponseStatus.FAILED)
            self.assertIsNone(failed.message)
            self.assertNotIn("private database detail", failed.error)

            empty = await AssistantService(
                session, runtime=FakeRuntime([_respond("  ")])
            ).handle_message(user_id=1, chat_id=chat.id, text="Empty answer")
            self.assertEqual(empty.status, AssistantResponseStatus.FAILED)
            self.assertIsNone(empty.message)
            messages = await ChatService(session).list_messages(chat.id)
            self.assertEqual([item.role for item in messages], ["user", "user"])
            runs = (await session.execute(select(AgentRunEntity))).scalars().all()
            self.assertEqual({run.status for run in runs}, {AgentRunStatus.FAILED})

        asyncio.run(_with_session(scenario))

    def test_schedule_validation_lifecycle_and_next_run_calculation(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            agents = AgentService(session)
            primary = await agents.ensure_primary_agent(user_id=1)
            foreign = await agents.ensure_primary_agent(user_id=2)
            service = ScheduleService(session, runtime=FakeRuntime([_respond("done")]))
            now = datetime.now(timezone.utc)
            foreign_chat = await ChatService(session).create_chat(user_id=2)

            once_expression = (now + timedelta(hours=1)).isoformat()
            once = await service.create(
                user_id=1,
                agent_id=primary.id,
                name="One time",
                prompt="Run once",
                schedule_type=ScheduleType.ONCE,
                schedule_expression=once_expression,
                timezone="Asia/Qyzylorda",
            )
            self.assertEqual(once.next_run_at, datetime.fromisoformat(once_expression).astimezone(timezone.utc))
            interval = await service.create(
                user_id=1,
                agent_id=primary.id,
                name="Hourly",
                prompt="Run hourly",
                schedule_type=ScheduleType.INTERVAL,
                schedule_expression="3600",
                timezone="UTC",
            )
            self.assertGreaterEqual(interval.next_run_at, now + timedelta(seconds=3590))
            cron = await service.create(
                user_id=1,
                agent_id=primary.id,
                name="Daily",
                prompt="Daily report",
                schedule_type=ScheduleType.CRON,
                schedule_expression="0 9 * * *",
                timezone="Asia/Qyzylorda",
            )
            self.assertEqual(cron.next_run_at.astimezone(ZoneInfo(cron.timezone)).hour, 9)

            revised_interval = await service.update(
                user_id=1,
                schedule_id=interval.id,
                data=ScheduleUpdate(name="Every half hour", schedule_expression="1800"),
            )
            self.assertEqual(revised_interval.name, "Every half hour")
            self.assertGreaterEqual(revised_interval.next_run_at, now + timedelta(seconds=1790))
            interval_executed_at = revised_interval.next_run_at
            interval_after_run = await service.mark_executed(
                user_id=1,
                schedule_id=interval.id,
                executed_at=interval_executed_at,
            )
            self.assertEqual(
                interval_after_run.next_run_at,
                interval_executed_at + timedelta(seconds=1800),
            )

            with self.assertRaises(ValueError):
                await service.create(
                    user_id=1,
                    agent_id=primary.id,
                    name="Bad timezone",
                    prompt="Run",
                    schedule_type=ScheduleType.INTERVAL,
                    schedule_expression="60",
                    timezone="Mars/Olympus",
                )
            for kind, expression in (
                (ScheduleType.INTERVAL, "0"),
                (ScheduleType.INTERVAL, "-10"),
                (ScheduleType.CRON, "0 99 * * *"),
                (ScheduleType.ONCE, "not-a-datetime"),
            ):
                with self.assertRaises(ValueError):
                    await service.create(
                        user_id=1,
                        agent_id=primary.id,
                        name="Invalid",
                        prompt="Run",
                        schedule_type=kind,
                        schedule_expression=expression,
                        timezone="UTC",
                    )
            with self.assertRaises(LookupError):
                await service.create(
                    user_id=1,
                    agent_id=foreign.id,
                    name="Foreign agent",
                    prompt="Run",
                    schedule_type=ScheduleType.INTERVAL,
                    schedule_expression="60",
                    timezone="UTC",
                )
            with self.assertRaises(LookupError):
                await service.create(
                    user_id=1,
                    agent_id=primary.id,
                    chat_id=foreign_chat.id,
                    name="Foreign chat",
                    prompt="Run",
                    schedule_type=ScheduleType.INTERVAL,
                    schedule_expression="60",
                    timezone="UTC",
                )
            with self.assertRaises(LookupError):
                await service.create(
                    user_id=999,
                    agent_id=primary.id,
                    name="Missing user",
                    prompt="Run",
                    schedule_type=ScheduleType.INTERVAL,
                    schedule_expression="60",
                    timezone="UTC",
                )
            inactive = await agents.create_user_agent(
                user_id=1, name="Inactive", role="Worker", goal="Work"
            )
            await AgentRepository(session).update(
                inactive.id, AgentUpdate(status=AgentStatus.DISABLED)
            )
            await session.commit()
            with self.assertRaisesRegex(ValueError, "active"):
                await service.create(
                    user_id=1,
                    agent_id=inactive.id,
                    name="Inactive agent",
                    prompt="Run",
                    schedule_type=ScheduleType.INTERVAL,
                    schedule_expression="60",
                    timezone="UTC",
                )

            with self.assertRaises(LookupError):
                await service.get(user_id=2, schedule_id=once.id)
            self.assertEqual(await service.list(user_id=2), [])
            disabled = await service.disable(user_id=1, schedule_id=once.id)
            self.assertEqual(disabled.status, ScheduleStatus.DISABLED)
            self.assertEqual((await service.disable(user_id=1, schedule_id=once.id)).status, ScheduleStatus.DISABLED)
            enabled = await service.enable(user_id=1, schedule_id=once.id)
            self.assertEqual(enabled.status, ScheduleStatus.ACTIVE)
            completed_once = await service.mark_executed(
                user_id=1,
                schedule_id=once.id,
                executed_at=once.next_run_at + timedelta(seconds=1),
            )
            self.assertIsNone(completed_once.next_run_at)
            self.assertEqual(completed_once.status, ScheduleStatus.DISABLED)
            archived = await service.archive(user_id=1, schedule_id=once.id)
            self.assertEqual(archived.status, ScheduleStatus.ARCHIVED)
            with self.assertRaisesRegex(ValueError, "archived"):
                await service.enable(user_id=1, schedule_id=once.id)

            executed_at = datetime(2026, 9, 21, 8, 59, tzinfo=ZoneInfo("Asia/Qyzylorda"))
            updated_cron = await service.mark_executed(
                user_id=1, schedule_id=cron.id, executed_at=executed_at
            )
            self.assertEqual(
                updated_cron.next_run_at.astimezone(ZoneInfo(cron.timezone)).hour,
                9,
            )
            self.assertEqual(updated_cron.last_run_at, executed_at.astimezone(timezone.utc))

        asyncio.run(_with_session(scenario))

    def test_due_schedule_uses_same_runtime_and_claims_only_one_agent_run(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            primary = await AgentService(session).ensure_primary_agent(user_id=1)
            skill = await SkillService(session).create_user_skill(
                user_id=1, key="daily", name="Daily", description="Prepare daily report"
            )
            await SkillService(session).assign_to_agent(
                user_id=1, agent_id=primary.id, skill_id=skill.id
            )
            await MemoryService(session).put(
                user_id=1,
                scope=MemoryScope.USER_GLOBAL,
                key="language",
                content="Russian",
            )
            runtime = FakeRuntime([_respond("daily result")])
            service = ScheduleService(session, runtime=runtime)
            schedule = await service.create(
                user_id=1,
                agent_id=primary.id,
                name="Daily report",
                prompt="Собери ежедневный отчёт",
                schedule_type=ScheduleType.INTERVAL,
                schedule_expression="3600",
                timezone="UTC",
            )
            now = datetime.now(timezone.utc)
            entity = (
                await session.execute(
                    select(ScheduleEntity).where(ScheduleEntity.id == schedule.id)
                )
            ).scalar_one()
            entity.next_run_at = now - timedelta(seconds=1)
            await session.commit()

            with self.assertRaises(LookupError):
                await service.execute_due(user_id=2, schedule_id=schedule.id, now=now)

            due = await service.get_due(now=now)
            self.assertEqual([item.id for item in due], [schedule.id])
            result = await service.execute_due(user_id=1, schedule_id=schedule.id, now=now)
            self.assertEqual(result.run_status, AgentRunStatus.COMPLETED)
            self.assertEqual(result.runtime_status, AgentRuntimeStatus.COMPLETED)
            self.assertEqual(result.content, "daily result")
            self.assertEqual(runtime.requests[0].message, "Собери ежедневный отчёт")
            self.assertEqual(runtime.requests[0].chat_id, None)
            self.assertEqual(runtime.contexts[0].active_skills[0].key, "daily")
            self.assertEqual(runtime.contexts[0].memory[0].content, "Russian")
            self.assertEqual(result.schedule.last_run_at, now)
            self.assertGreater(result.schedule.next_run_at, now)

            self.assertIsNone(
                await service.execute_due(user_id=1, schedule_id=schedule.id, now=now)
            )
            runs = (await session.execute(select(AgentRunEntity))).scalars().all()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].starting_agent_id, primary.id)
            self.assertEqual(runs[0].status, AgentRunStatus.COMPLETED)
            self.assertEqual(len(runtime.requests), 1)
            self.assertEqual(await ScheduleRepository(session).list_due(now), [])

        asyncio.run(_with_session(scenario))

    def test_due_claim_is_atomic_across_competing_workers(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            primary = await AgentService(session).ensure_primary_agent(user_id=1)
            service = ScheduleService(session, runtime=FakeRuntime([_respond("done")]))
            now = datetime.now(timezone.utc)
            schedule = await service.create(
                user_id=1,
                agent_id=primary.id,
                name="Claim once",
                prompt="Run once",
                schedule_type=ScheduleType.ONCE,
                schedule_expression=(now - timedelta(seconds=1)).isoformat(),
                timezone="UTC",
            )
            repository = ScheduleRepository(session)
            first_token, second_token = uuid4(), uuid4()
            first_claim = await repository.claim_due(
                schedule_id=schedule.id,
                token=first_token,
                now=now,
                expires_at=now + timedelta(minutes=10),
            )
            second_claim = await repository.claim_due(
                schedule_id=schedule.id,
                token=second_token,
                now=now,
                expires_at=now + timedelta(minutes=10),
            )
            self.assertIsNotNone(first_claim)
            self.assertEqual(first_claim.claim_token, first_token)
            self.assertIsNone(second_claim)
            await session.commit()

            finished = await service.mark_executed(
                user_id=1,
                schedule_id=schedule.id,
                claim_token=first_token,
                executed_at=now,
            )
            self.assertEqual(finished.status, ScheduleStatus.DISABLED)
            self.assertIsNone(finished.claim_token)

        asyncio.run(_with_session(scenario))

    def test_failed_once_schedule_is_disabled_after_attempt(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            primary = await AgentService(session).ensure_primary_agent(user_id=1)
            now = datetime.now(timezone.utc)
            schedule = await ScheduleService(session).create(
                user_id=1,
                agent_id=primary.id,
                name="Fail once",
                prompt="Run once",
                schedule_type=ScheduleType.ONCE,
                schedule_expression=(now - timedelta(seconds=1)).isoformat(),
                timezone="UTC",
            )
            runtime = FakeRuntime(error=RuntimeError("scheduled runtime failed"))
            entity = (
                await session.execute(
                    select(ScheduleEntity).where(ScheduleEntity.id == schedule.id)
                )
            ).scalar_one()
            entity.next_run_at = now - timedelta(seconds=1)
            await session.commit()

            result = await ScheduleService(session, runtime=runtime).execute_due(
                user_id=1,
                schedule_id=schedule.id,
                now=now,
            )
            self.assertEqual(result.runtime_status, AgentRuntimeStatus.FAILED)
            self.assertEqual(result.run_status, AgentRunStatus.FAILED)
            self.assertEqual(result.schedule.status, ScheduleStatus.DISABLED)
            self.assertIsNone(result.schedule.next_run_at)

        asyncio.run(_with_session(scenario))
