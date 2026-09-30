import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
import tempfile
import unittest

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from database.base import Base
from database.entities.agent import AgentEntity
from database.entities.agent_connection import AgentConnectionEntity
from database.entities.agent_prompt import AgentPromptVersionEntity
from database.entities.agent_run import AgentRunEntity
from database.entities.agent_run_event import AgentRunEventEntity
from database.entities.approval import ApprovalEntity
from database.entities.chat import ChatEntity
from database.entities.message import MessageEntity
from database.entities.permission import PermissionEntity
from database.entities.system_agent_template import SystemAgentTemplateEntity
from database.entities.user import UserEntity
from database.entities.user_agent_override import UserAgentOverrideEntity
from models.approval import ApprovalStatus
from models.agent import AgentStatus, AgentUpdate
from models.agent_run import AgentRunEventType, AgentRunStatus
from models.permission import ActionClass, PermissionSubjectType
from models.runtime import (
    AgentRuntimeRequest,
    AgentRuntimeStatus,
    DelegationDecision,
    DelegationType,
    RuntimeBudgets,
    RuntimeStep,
    TemporarySubagentStatus,
)
from models.tool import ToolExecutionStatus, ToolIntent, ToolRequest
from services.agent import AgentService
from services.agent_run import AgentRunService
from services.agent_runtime import AgentRuntimeService
from services.approval import ApprovalService
from services.permission import PermissionDeniedError, PermissionService
from services.system_agent import SystemAgentService
from services.tool_authority import BackendToolAuthority, FakeToolExecutor
from repositories.agent import AgentRepository
from agents.assistant.crewai.runtime import CrewAIToolApprovalRuntime


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


class RuntimeAuthorityTests(unittest.TestCase):
    def test_permission_grant_require_revoke_and_tenant_isolation(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            permissions = PermissionService(session)
            granted = await permissions.grant(
                user_id=1,
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id="agent-one",
                action_class=ActionClass.READ,
                resource_scope="information:*",
            )

            assert granted.allowed is True
            assert await permissions.check(
                user_id=1,
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id="agent-one",
                action_class=ActionClass.READ,
                resource="information:weather",
            )
            await permissions.require(
                user_id=1,
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id="agent-one",
                action_class=ActionClass.READ,
                resource="information:weather",
            )

            assert not await permissions.check(
                user_id=2,
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id="agent-one",
                action_class=ActionClass.READ,
                resource="information:weather",
            )
            with self.assertRaises(PermissionDeniedError):
                await permissions.require(
                    user_id=1,
                    subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                    subject_id="agent-one",
                    action_class=ActionClass.WRITE,
                    resource="information:weather",
                )

            revoked = await permissions.revoke(
                user_id=1,
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id="agent-one",
                action_class=ActionClass.READ,
                resource_scope="information:*",
            )
            assert revoked.allowed is False
            assert not await permissions.check(
                user_id=1,
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id="agent-one",
                action_class=ActionClass.READ,
                resource="information:weather",
            )

        asyncio.run(_with_session(scenario))

    def test_primary_delegates_only_to_connected_agent_and_keeps_pinned_prompt(self) -> None:
        class FakeRuntime:
            def __init__(self, steps: list[RuntimeStep]):
                self.steps = steps
                self.active_subjects: list[str] = []
                self.prompt_versions: list[str] = []

            async def run(self, context, request):
                self.active_subjects.append(context.active_agent.subject_id)
                self.prompt_versions.append(str(context.starting_prompt.id))
                return self.steps.pop(0)

        async def scenario(session: AsyncSession) -> None:
            agents = AgentService(session)
            primary = await agents.ensure_primary_agent(user_id=1)
            worker = await agents.create_user_agent(
                user_id=1,
                name="Research",
                role="Researcher",
                goal="Find facts",
            )
            runs = AgentRunService(session)
            run = await runs.create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            pinned_prompt_id = run.prompt_version_id
            await agents.update_prompt(
                user_id=1,
                agent_id=primary.id,
                role="Changed after run creation",
                goal="This version must not replace the pinned version",
            )

            fake = FakeRuntime(
                [
                    RuntimeStep(
                        decision=DelegationDecision(
                            type=DelegationType.DELEGATE_USER_AGENT,
                            target_id=str(worker.id),
                            task_summary="research the answer",
                        )
                    ),
                    RuntimeStep(
                        decision=DelegationDecision(type=DelegationType.RESPOND),
                        content="worker result",
                    ),
                    RuntimeStep(
                        decision=DelegationDecision(type=DelegationType.RESPOND),
                        content="final answer",
                    ),
                ]
            )
            result = await AgentRuntimeService(session, runtime=fake).execute(
                AgentRuntimeRequest(
                    run_id=run.id,
                    user_id=1,
                    agent_id=primary.id,
                    message="Please research",
                )
            )
            assert result.status == AgentRuntimeStatus.COMPLETED
            assert result.content == "final answer"
            assert fake.active_subjects == [str(primary.id), str(worker.id), str(primary.id)]
            assert fake.prompt_versions == [str(pinned_prompt_id)] * 3
            persisted = await runs.get_run(user_id=1, run_id=run.id)
            assert persisted.status == AgentRunStatus.COMPLETED
            assert persisted.prompt_version_id == pinned_prompt_id
            assert persisted.usage["delegations"] == 1
            assert persisted.usage["llm_calls"] == 3

        asyncio.run(_with_session(scenario))

    def test_approval_decision_is_single_use_tenant_scoped_and_resumes_run(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            agent = await AgentService(session).ensure_primary_agent(user_id=1)
            runs = AgentRunService(session)
            run = await runs.create_run(
                user_id=1, starting_agent_id=agent.id, model_name="test-model"
            )
            await runs.start(user_id=1, run_id=run.id)

            approvals = ApprovalService(session)
            approval = await approvals.request(
                user_id=1,
                run_id=run.id,
                requesting_subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                requesting_subject_id=str(agent.id),
                action_name="write_note",
                action_class=ActionClass.WRITE,
                arguments={"text": "safe serializable value"},
            )
            assert approval.status == ApprovalStatus.PENDING
            assert (await runs.get_run(user_id=1, run_id=run.id)).status == AgentRunStatus.WAITING_APPROVAL
            with self.assertRaises(LookupError):
                await approvals.approve(user_id=2, approval_id=approval.id)

            approved = await approvals.approve(
                user_id=1,
                approval_id=approval.id,
                decision_metadata={"source": "test"},
            )
            assert approved.status == ApprovalStatus.APPROVED
            assert approved.decided_at is not None
            assert (await runs.get_run(user_id=1, run_id=run.id)).status == AgentRunStatus.RUNNING
            with self.assertRaisesRegex(ValueError, "already decided"):
                await approvals.reject(user_id=1, approval_id=approval.id)

            events = await runs.list_events(user_id=1, run_id=run.id)
            assert AgentRunEventType.APPROVAL_REQUESTED in {event.event_type for event in events}
            assert AgentRunEventType.APPROVAL_APPROVED in {event.event_type for event in events}

            rejected_run = await runs.create_run(
                user_id=1, starting_agent_id=agent.id, model_name="test-model"
            )
            await runs.start(user_id=1, run_id=rejected_run.id)
            rejected = await approvals.request(
                user_id=1,
                run_id=rejected_run.id,
                requesting_subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                requesting_subject_id=str(agent.id),
                action_name="external_action",
                action_class=ActionClass.EXTERNAL_SIDE_EFFECT,
                arguments={"target": "calendar"},
            )
            rejected = await approvals.reject(user_id=1, approval_id=rejected.id)
            assert rejected.status == ApprovalStatus.REJECTED
            assert (await runs.get_run(user_id=1, run_id=rejected_run.id)).status == AgentRunStatus.RUNNING

        asyncio.run(_with_session(scenario))

    def test_tool_authority_enforces_permission_and_one_action_approval(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            agent = await AgentService(session).ensure_primary_agent(user_id=1)
            runs = AgentRunService(session)
            run = await runs.create_run(
                user_id=1, starting_agent_id=agent.id, model_name="test-model"
            )
            await runs.start(user_id=1, run_id=run.id)
            permissions = PermissionService(session)
            subject_id = str(agent.id)
            await permissions.grant(
                user_id=1,
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id=subject_id,
                action_class=ActionClass.READ,
                resource_scope="information:*",
            )
            await permissions.grant(
                user_id=1,
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id=subject_id,
                action_class=ActionClass.WRITE,
                resource_scope="notes:*",
            )
            executor = FakeToolExecutor()
            tools = BackendToolAuthority(session, executor=executor)

            read = await tools.request(
                ToolRequest(
                    run_id=run.id,
                    user_id=1,
                    requesting_subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                    requesting_subject_id=subject_id,
                    name="read_information",
                    action_class=ActionClass.READ,
                    resource="information:weather",
                    arguments={"query": "weather"},
                )
            )
            assert read.status == ToolExecutionStatus.EXECUTED
            assert read.output == {"query": "weather", "information": "fake result"}

            denied = await tools.request(
                ToolRequest(
                    run_id=run.id,
                    user_id=1,
                    requesting_subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                    requesting_subject_id=subject_id,
                    name="external_action",
                    action_class=ActionClass.EXTERNAL_SIDE_EFFECT,
                    resource="calendar:event",
                    arguments={"title": "blocked"},
                )
            )
            assert denied.status == ToolExecutionStatus.DENIED
            assert executor.external_actions == []
            await permissions.grant(
                user_id=1,
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id=subject_id,
                action_class=ActionClass.EXTERNAL_SIDE_EFFECT,
                resource_scope="calendar:*",
            )

            pending = await tools.request(
                ToolRequest(
                    run_id=run.id,
                    user_id=1,
                    requesting_subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                    requesting_subject_id=subject_id,
                    name="write_note",
                    action_class=ActionClass.WRITE,
                    resource="notes:personal",
                    arguments={"text": "remember this"},
                )
            )
            assert pending.status == ToolExecutionStatus.WAITING_APPROVAL
            assert pending.approval_id is not None
            await ApprovalService(session).approve(
                user_id=1, approval_id=pending.approval_id
            )
            executed = await tools.execute_approved(
                user_id=1, approval_id=pending.approval_id
            )
            assert executed.status == ToolExecutionStatus.EXECUTED
            assert executor.notes == ["remember this"]
            with self.assertRaisesRegex(ValueError, "already executed"):
                await tools.execute_approved(user_id=1, approval_id=pending.approval_id)

            external = await tools.request(
                ToolRequest(
                    run_id=run.id,
                    user_id=1,
                    requesting_subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                    requesting_subject_id=subject_id,
                    name="external_action",
                    action_class=ActionClass.EXTERNAL_SIDE_EFFECT,
                    resource="calendar:event",
                    arguments={"title": "requires approval"},
                )
            )
            assert external.status == ToolExecutionStatus.WAITING_APPROVAL
            await ApprovalService(session).reject(
                user_id=1, approval_id=external.approval_id
            )
            rejected = await tools.execute_approved(
                user_id=1, approval_id=external.approval_id
            )
            assert rejected.status == ToolExecutionStatus.REJECTED
            assert executor.external_actions == []

        asyncio.run(_with_session(scenario))

    def test_crewai_hitl_pauses_and_resumes_one_approved_tool_action(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            agent = await AgentService(session).ensure_primary_agent(user_id=1)
            runs = AgentRunService(session)
            run = await runs.create_run(
                user_id=1, starting_agent_id=agent.id, model_name="test-model"
            )
            await runs.start(user_id=1, run_id=run.id)
            await PermissionService(session).grant(
                user_id=1,
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id=str(agent.id),
                action_class=ActionClass.WRITE,
                resource_scope="notes:*",
            )
            executor = FakeToolExecutor()
            authority = BackendToolAuthority(session, executor=executor)
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
                runtime = CrewAIToolApprovalRuntime(
                    authority=authority,
                    persistence_path=Path(directory) / "flows.db",
                )
                pending = await runtime.begin(
                    ToolRequest(
                        run_id=run.id,
                        user_id=1,
                        requesting_subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                        requesting_subject_id=str(agent.id),
                        name="write_note",
                        action_class=ActionClass.WRITE,
                        resource="notes:personal",
                        arguments={"text": "approved through CrewAI Flow"},
                    )
                )
                assert pending.status == ToolExecutionStatus.WAITING_APPROVAL
                assert pending.approval_id is not None
                assert pending.flow_id is not None
                assert (await runs.get_run(user_id=1, run_id=run.id)).status == AgentRunStatus.WAITING_APPROVAL

                await ApprovalService(session).approve(
                    user_id=1, approval_id=pending.approval_id
                )
                resumed = await runtime.resume(user_id=1, flow_id=pending.flow_id)
                assert resumed.status == ToolExecutionStatus.EXECUTED
                assert executor.notes == ["approved through CrewAI Flow"]
                assert (await runs.get_run(user_id=1, run_id=run.id)).status == AgentRunStatus.RUNNING

        asyncio.run(_with_session(scenario))

    def test_agent_runtime_recovers_approved_action_through_crewai_flow(self) -> None:
        class FakeRuntime:
            def __init__(self, steps: list[RuntimeStep]):
                self.steps = steps

            async def run(self, context, request):
                return self.steps.pop(0)

        async def scenario(session: AsyncSession) -> None:
            agent = await AgentService(session).ensure_primary_agent(user_id=1)
            run = await AgentRunService(session).create_run(
                user_id=1, starting_agent_id=agent.id, model_name="test-model"
            )
            await PermissionService(session).grant(
                user_id=1,
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id=str(agent.id),
                action_class=ActionClass.WRITE,
                resource_scope="notes:*",
            )
            executor = FakeToolExecutor()
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
                persistence_path = Path(directory) / "flows.db"
                first_authority = BackendToolAuthority(session, executor=executor)
                first_service = AgentRuntimeService(
                    session,
                    runtime=FakeRuntime(
                        [
                            RuntimeStep(
                                decision=DelegationDecision(type=DelegationType.RESPOND),
                                tool_intent=ToolIntent(
                                    name="write_note",
                                    action_class=ActionClass.WRITE,
                                    resource="notes:personal",
                                    arguments={"text": "persisted approval"},
                                ),
                            )
                        ]
                    ),
                    tool_executor=executor,
                    approval_runtime=CrewAIToolApprovalRuntime(
                        authority=first_authority,
                        persistence_path=persistence_path,
                    ),
                )
                waiting = await first_service.execute(
                    AgentRuntimeRequest(
                        run_id=run.id,
                        user_id=1,
                        agent_id=agent.id,
                        message="Write this note",
                    )
                )
                assert waiting.status == AgentRuntimeStatus.WAITING_APPROVAL
                approval_id = waiting.context.pending_approval.approval_id
                assert waiting.context.pending_approval.flow_id

                await ApprovalService(session).approve(
                    user_id=1, approval_id=approval_id
                )

                restarted_authority = BackendToolAuthority(session, executor=executor)
                restarted_service = AgentRuntimeService(
                    session,
                    runtime=FakeRuntime(
                        [
                            RuntimeStep(
                                decision=DelegationDecision(type=DelegationType.RESPOND),
                                content="note saved",
                            )
                        ]
                    ),
                    tool_executor=executor,
                    approval_runtime=CrewAIToolApprovalRuntime(
                        authority=restarted_authority,
                        persistence_path=persistence_path,
                    ),
                )
                completed = await restarted_service.resume(user_id=1, run_id=run.id)
                assert completed.status == AgentRuntimeStatus.COMPLETED
                assert completed.content == "note saved"
                assert executor.notes == ["persisted approval"]
                persisted = await AgentRunService(session).get_run(user_id=1, run_id=run.id)
                assert persisted.status == AgentRunStatus.COMPLETED
                assert persisted.usage["tool_calls"] == 1

        asyncio.run(_with_session(scenario))

    def test_primary_direct_response_and_system_agent_delegation(self) -> None:
        class FakeRuntime:
            def __init__(self, steps: list[RuntimeStep]):
                self.steps = steps
                self.active: list[str] = []

            async def run(self, context, request):
                self.active.append(context.active_agent.subject_id)
                return self.steps.pop(0)

        async def scenario(session: AsyncSession) -> None:
            primary = await AgentService(session).ensure_primary_agent(user_id=1)
            runs = AgentRunService(session)
            direct_run = await runs.create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            direct_runtime = FakeRuntime(
                [
                    RuntimeStep(
                        decision=DelegationDecision(type=DelegationType.RESPOND),
                        content="direct answer",
                    )
                ]
            )
            direct_request = AgentRuntimeRequest(
                run_id=direct_run.id,
                user_id=1,
                agent_id=primary.id,
                message="Answer directly",
            )
            direct = await AgentRuntimeService(session, runtime=direct_runtime).execute(
                direct_request
            )
            assert direct.status == AgentRuntimeStatus.COMPLETED
            assert direct.content == "direct answer"
            await SystemAgentService(session).ensure_initial_templates()
            system_run = await runs.create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            system_runtime = FakeRuntime(
                [
                    RuntimeStep(
                        decision=DelegationDecision(
                            type=DelegationType.DELEGATE_SYSTEM_AGENT,
                            target_id="calendar",
                            task_summary="inspect calendar",
                        )
                    ),
                    RuntimeStep(
                        decision=DelegationDecision(type=DelegationType.RESPOND),
                        content="calendar result",
                    ),
                    RuntimeStep(
                        decision=DelegationDecision(type=DelegationType.RESPOND),
                        content="calendar answer",
                    ),
                ]
            )
            delegated = await AgentRuntimeService(session, runtime=system_runtime).execute(
                AgentRuntimeRequest(
                    run_id=system_run.id,
                    user_id=1,
                    agent_id=primary.id,
                    message="Check my calendar",
                )
            )
            assert delegated.status == AgentRuntimeStatus.COMPLETED
            assert system_runtime.active == [str(primary.id), "calendar", str(primary.id)]

        asyncio.run(_with_session(scenario))

    def test_backend_blocks_disconnected_inactive_and_worker_delegation(self) -> None:
        class FakeRuntime:
            def __init__(self, decision: DelegationDecision):
                self.decision = decision

            async def run(self, context, request):
                return RuntimeStep(decision=self.decision)

        async def execute(
            session: AsyncSession,
            *,
            starting_agent_id,
            target_id,
        ):
            run = await AgentRunService(session).create_run(
                user_id=1,
                starting_agent_id=starting_agent_id,
                model_name="test-model",
            )
            return await AgentRuntimeService(
                session,
                runtime=FakeRuntime(
                    DelegationDecision(
                        type=DelegationType.DELEGATE_USER_AGENT,
                        target_id=str(target_id),
                        task_summary="forbidden delegation",
                    )
                ),
            ).execute(
                AgentRuntimeRequest(
                    run_id=run.id,
                    user_id=1,
                    agent_id=starting_agent_id,
                    message="delegate",
                )
            )

        async def scenario(session: AsyncSession) -> None:
            agents = AgentService(session)
            primary = await agents.ensure_primary_agent(user_id=1)
            disconnected = await agents.create_user_agent(
                user_id=1, name="Disconnected", role="Worker", goal="Work"
            )
            await agents.disconnect_agents(
                user_id=1,
                parent_agent_id=primary.id,
                child_agent_id=disconnected.id,
            )
            result = await execute(
                session, starting_agent_id=primary.id, target_id=disconnected.id
            )
            assert result.status == AgentRuntimeStatus.FAILED
            assert "not active and connected" in result.error

            archived = await agents.create_user_agent(
                user_id=1, name="Archived", role="Worker", goal="Work"
            )
            await agents.archive_agent(user_id=1, agent_id=archived.id)
            result = await execute(
                session, starting_agent_id=primary.id, target_id=archived.id
            )
            assert result.status == AgentRuntimeStatus.FAILED

            disabled = await agents.create_user_agent(
                user_id=1, name="Disabled", role="Worker", goal="Work"
            )
            await AgentRepository(session).update(
                disabled.id, AgentUpdate(status=AgentStatus.DISABLED)
            )
            await session.commit()
            result = await execute(
                session, starting_agent_id=primary.id, target_id=disabled.id
            )
            assert result.status == AgentRuntimeStatus.FAILED

            worker = await agents.create_user_agent(
                user_id=1, name="Worker", role="Worker", goal="Work"
            )
            other = await agents.create_user_agent(
                user_id=1, name="Other", role="Worker", goal="Work"
            )
            result = await execute(
                session, starting_agent_id=worker.id, target_id=other.id
            )
            assert result.status == AgentRuntimeStatus.FAILED
            assert "Only the primary agent" in result.error

        asyncio.run(_with_session(scenario))

    def test_temporary_subagent_lifecycle_permissions_and_limits(self) -> None:
        class FakeRuntime:
            def __init__(self, steps: list[RuntimeStep]):
                self.steps = steps

            async def run(self, context, request):
                return self.steps.pop(0)

        def spawn_decision(*, permissions: list[str] | None = None) -> DelegationDecision:
            return DelegationDecision(
                type=DelegationType.CREATE_TEMPORARY_SUBAGENT,
                task_summary="solve specialist task",
                temporary_name="Ephemeral Specialist",
                temporary_role="Specialist",
                temporary_goal="Solve one task",
                requested_permissions=permissions or [],
            )

        async def execute_spawn(
            session: AsyncSession,
            *,
            agent_id,
            decision: DelegationDecision,
            budgets: RuntimeBudgets | None = None,
            finish: bool = False,
        ):
            run = await AgentRunService(session).create_run(
                user_id=1, starting_agent_id=agent_id, model_name="test-model"
            )
            steps = [RuntimeStep(decision=decision)]
            if finish:
                steps.extend(
                    [
                        RuntimeStep(
                            decision=DelegationDecision(type=DelegationType.RESPOND),
                            content="temporary result",
                        ),
                        RuntimeStep(
                            decision=DelegationDecision(type=DelegationType.RESPOND),
                            content="final result",
                        ),
                    ]
                )
            return await AgentRuntimeService(
                session,
                runtime=FakeRuntime(steps),
                budgets=budgets,
            ).execute(
                AgentRuntimeRequest(
                    run_id=run.id,
                    user_id=1,
                    agent_id=agent_id,
                    message="specialist needed",
                )
            )

        async def scenario(session: AsyncSession) -> None:
            agents = AgentService(session)
            primary = await agents.ensure_primary_agent(user_id=1)
            await PermissionService(session).grant(
                user_id=1,
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id=str(primary.id),
                action_class=ActionClass.READ,
                resource_scope="information:*",
            )
            completed = await execute_spawn(
                session,
                agent_id=primary.id,
                decision=spawn_decision(permissions=["read:information:*"]),
                finish=True,
            )
            assert completed.status == AgentRuntimeStatus.COMPLETED
            assert completed.context.usage.subagents == 1
            assert len(completed.context.temporary_subagents) == 1
            assert completed.context.temporary_subagents[0].status == TemporarySubagentStatus.ARCHIVED
            assert [item.id for item in await agents.list_active_agents(1)] == [primary.id]

            worker = await agents.create_user_agent(
                user_id=1, name="No Spawn", role="Worker", goal="Work"
            )
            forbidden = await execute_spawn(
                session,
                agent_id=worker.id,
                decision=spawn_decision(),
            )
            assert forbidden.status == AgentRuntimeStatus.FAILED
            assert "cannot spawn" in forbidden.error

            excessive = await execute_spawn(
                session,
                agent_id=primary.id,
                decision=spawn_decision(permissions=["write:notes:*"]),
            )
            assert excessive.status == AgentRuntimeStatus.FAILED
            assert "exceed parent permissions" in excessive.error

            for budgets, message in (
                (RuntimeBudgets(max_subagents=0), "subagents=0"),
                (RuntimeBudgets(max_subagents_per_parent=0), "per-parent"),
                (RuntimeBudgets(max_subagent_depth=0), "depth"),
            ):
                limited = await execute_spawn(
                    session,
                    agent_id=primary.id,
                    decision=spawn_decision(),
                    budgets=budgets,
                )
                assert limited.status == AgentRuntimeStatus.FAILED
                assert message in limited.error

        asyncio.run(_with_session(scenario))

    def test_backend_enforces_runtime_budgets(self) -> None:
        class FakeRuntime:
            def __init__(self, step: RuntimeStep):
                self.step = step
                self.calls = 0

            async def run(self, context, request):
                self.calls += 1
                return self.step

        async def execute(
            session: AsyncSession,
            *,
            agent_id,
            runtime: FakeRuntime,
            budgets: RuntimeBudgets,
        ):
            run = await AgentRunService(session).create_run(
                user_id=1, starting_agent_id=agent_id, model_name="test-model"
            )
            result = await AgentRuntimeService(
                session, runtime=runtime, budgets=budgets
            ).execute(
                AgentRuntimeRequest(
                    run_id=run.id,
                    user_id=1,
                    agent_id=agent_id,
                    message="exercise budget",
                )
            )
            events = await AgentRunService(session).list_events(user_id=1, run_id=run.id)
            assert AgentRunEventType.BUDGET_EXCEEDED in {event.event_type for event in events}
            return result

        async def scenario(session: AsyncSession) -> None:
            agents = AgentService(session)
            primary = await agents.ensure_primary_agent(user_id=1)
            worker = await agents.create_user_agent(
                user_id=1, name="Budget Worker", role="Worker", goal="Work"
            )
            respond = RuntimeStep(
                decision=DelegationDecision(type=DelegationType.RESPOND),
                content="done",
            )

            zero_llm = FakeRuntime(respond)
            result = await execute(
                session,
                agent_id=primary.id,
                runtime=zero_llm,
                budgets=RuntimeBudgets(max_llm_calls=0),
            )
            assert result.status == AgentRuntimeStatus.FAILED
            assert zero_llm.calls == 0

            result = await execute(
                session,
                agent_id=primary.id,
                runtime=FakeRuntime(
                    RuntimeStep(
                        decision=DelegationDecision(
                            type=DelegationType.DELEGATE_USER_AGENT,
                            target_id=str(worker.id),
                            task_summary="delegate",
                        )
                    )
                ),
                budgets=RuntimeBudgets(max_delegations=0),
            )
            assert "delegations=0" in result.error

            result = await execute(
                session,
                agent_id=primary.id,
                runtime=FakeRuntime(
                    RuntimeStep(
                        decision=DelegationDecision(type=DelegationType.RESPOND),
                        tool_intent=ToolIntent(
                            name="read_information",
                            action_class=ActionClass.READ,
                            resource="information:test",
                        ),
                    )
                ),
                budgets=RuntimeBudgets(max_tool_calls=0),
            )
            assert "tool_calls=0" in result.error

            result = await execute(
                session,
                agent_id=primary.id,
                runtime=FakeRuntime(respond.model_copy(update={"tokens_used": 1})),
                budgets=RuntimeBudgets(max_tokens=0),
            )
            assert "tokens=0" in result.error

            result = await execute(
                session,
                agent_id=primary.id,
                runtime=FakeRuntime(respond.model_copy(update={"retries": 1})),
                budgets=RuntimeBudgets(max_retries=0),
            )
            assert "retries=0" in result.error

            result = await execute(
                session,
                agent_id=primary.id,
                runtime=FakeRuntime(respond),
                budgets=RuntimeBudgets(max_time=-1),
            )
            assert "max_time" in result.error

        asyncio.run(_with_session(scenario))

    def test_acceptance_user_worker_read_tool_flow(self) -> None:
        class FakeRuntime:
            def __init__(self, steps: list[RuntimeStep]):
                self.steps = steps

            async def run(self, context, request):
                return self.steps.pop(0)

        async def scenario(session: AsyncSession) -> None:
            agents = AgentService(session)
            primary = await agents.ensure_primary_agent(user_id=1)
            worker = await agents.create_user_agent(
                user_id=1, name="Reader", role="Reader", goal="Read information"
            )
            await PermissionService(session).grant(
                user_id=1,
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id=str(worker.id),
                action_class=ActionClass.READ,
                resource_scope="information:*",
            )
            run = await AgentRunService(session).create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            result = await AgentRuntimeService(
                session,
                runtime=FakeRuntime(
                    [
                        RuntimeStep(
                            decision=DelegationDecision(
                                type=DelegationType.DELEGATE_USER_AGENT,
                                target_id=str(worker.id),
                                task_summary="read information",
                            )
                        ),
                        RuntimeStep(
                            decision=DelegationDecision(type=DelegationType.RESPOND),
                            tool_intent=ToolIntent(
                                name="read_information",
                                action_class=ActionClass.READ,
                                resource="information:weather",
                                arguments={"query": "weather"},
                            ),
                        ),
                        RuntimeStep(
                            decision=DelegationDecision(type=DelegationType.RESPOND),
                            content="worker read result",
                        ),
                        RuntimeStep(
                            decision=DelegationDecision(type=DelegationType.RESPOND),
                            content="primary synthesis",
                        ),
                    ]
                ),
                tool_executor=FakeToolExecutor(),
            ).execute(
                AgentRuntimeRequest(
                    run_id=run.id,
                    user_id=1,
                    agent_id=primary.id,
                    message="Read and summarize",
                )
            )
            assert result.status == AgentRuntimeStatus.COMPLETED
            assert result.content == "primary synthesis"
            assert result.context.usage.delegations == 1
            assert result.context.usage.tool_calls == 1

        asyncio.run(_with_session(scenario))

    def test_acceptance_system_agent_external_action_flow(self) -> None:
        class FakeRuntime:
            def __init__(self, steps: list[RuntimeStep]):
                self.steps = steps

            async def run(self, context, request):
                return self.steps.pop(0)

        async def scenario(session: AsyncSession) -> None:
            primary = await AgentService(session).ensure_primary_agent(user_id=1)
            await SystemAgentService(session).ensure_initial_templates()
            await PermissionService(session).grant(
                user_id=1,
                subject_type=PermissionSubjectType.SYSTEM_AGENT,
                subject_id="calendar",
                action_class=ActionClass.EXTERNAL_SIDE_EFFECT,
                resource_scope="calendar:*",
            )
            run = await AgentRunService(session).create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            executor = FakeToolExecutor()
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
                persistence_path = Path(directory) / "flows.db"
                first_authority = BackendToolAuthority(session, executor=executor)
                waiting = await AgentRuntimeService(
                    session,
                    runtime=FakeRuntime(
                        [
                            RuntimeStep(
                                decision=DelegationDecision(
                                    type=DelegationType.DELEGATE_SYSTEM_AGENT,
                                    target_id="calendar",
                                    task_summary="create calendar event",
                                )
                            ),
                            RuntimeStep(
                                decision=DelegationDecision(type=DelegationType.RESPOND),
                                tool_intent=ToolIntent(
                                    name="external_action",
                                    action_class=ActionClass.EXTERNAL_SIDE_EFFECT,
                                    resource="calendar:event",
                                    arguments={"title": "Planning"},
                                ),
                            ),
                        ]
                    ),
                    tool_executor=executor,
                    approval_runtime=CrewAIToolApprovalRuntime(
                        authority=first_authority,
                        persistence_path=persistence_path,
                    ),
                ).execute(
                    AgentRuntimeRequest(
                        run_id=run.id,
                        user_id=1,
                        agent_id=primary.id,
                        message="Create a planning event",
                    )
                )
                assert waiting.status == AgentRuntimeStatus.WAITING_APPROVAL
                assert waiting.context.active_agent.subject_id == "calendar"
                await ApprovalService(session).approve(
                    user_id=1,
                    approval_id=waiting.context.pending_approval.approval_id,
                )

                restarted_authority = BackendToolAuthority(session, executor=executor)
                completed = await AgentRuntimeService(
                    session,
                    runtime=FakeRuntime(
                        [
                            RuntimeStep(
                                decision=DelegationDecision(type=DelegationType.RESPOND),
                                content="calendar action completed",
                            ),
                            RuntimeStep(
                                decision=DelegationDecision(type=DelegationType.RESPOND),
                                content="event created",
                            ),
                        ]
                    ),
                    tool_executor=executor,
                    approval_runtime=CrewAIToolApprovalRuntime(
                        authority=restarted_authority,
                        persistence_path=persistence_path,
                    ),
                ).resume(user_id=1, run_id=run.id)
                assert completed.status == AgentRuntimeStatus.COMPLETED
                assert completed.content == "event created"
                assert executor.external_actions == [{"title": "Planning"}]

        asyncio.run(_with_session(scenario))
