import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
import unittest
from uuid import UUID

from pydantic import ValidationError
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
from database.entities.skill import SkillEntity
from database.entities.system_agent_template import SystemAgentTemplateEntity
from database.entities.user import UserEntity
from database.entities.user_agent_override import UserAgentOverrideEntity
from models.agent import AgentKind
from models.agent_run import AgentRunStatus
from models.memory import MemoryCreate, MemoryScope
from models.runtime import (
    AgentRuntimeRequest,
    AgentRuntimeStatus,
    DelegationDecision,
    DelegationType,
    RuntimeAgentKind,
    RuntimeStep,
    RuntimeTemporarySubagent,
)
from models.skill import SkillCreate, SkillOwnerType, SkillStatus
from services.agent import AgentService
from services.agent_run import AgentRunService
from services.agent_runtime import AgentRuntimeService
from services.memory import MemoryService
from services.skill import SkillService
from services.system_agent import SystemAgentService
from repositories.memory import MemoryRepository


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


class RecordingRuntime:
    def __init__(self, steps: list[RuntimeStep]):
        self.steps = steps
        self.contexts = []

    async def run(self, context, request):
        self.contexts.append(context.model_copy(deep=True))
        return self.steps.pop(0)


def _respond(content: str) -> RuntimeStep:
    return RuntimeStep(
        decision=DelegationDecision(type=DelegationType.RESPOND),
        content=content,
    )


async def _configure_information_skills(session: AsyncSession, *, allowed: list[str], default: list[str]) -> None:
    await SystemAgentService(session).ensure_initial_templates()
    template = (
        await session.execute(
            select(SystemAgentTemplateEntity).where(SystemAgentTemplateEntity.key == "information")
        )
    ).scalar_one()
    template.allowed_skills = allowed
    template.default_skills = default
    await session.commit()


class SkillMemoryTests(unittest.TestCase):
    def test_skill_ownership_assignment_and_lifecycle(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            agents = AgentService(session)
            own_agent = await agents.create_user_agent(
                user_id=1, name="Python", role="Python expert", goal="Help with Python"
            )
            other_agent = await agents.create_user_agent(
                user_id=2, name="Other", role="Worker", goal="Work"
            )
            skills = SkillService(session)
            system_skill = await skills.create_system_skill(
                key="shared_key", name="Shared", description="Global capability"
            )
            own_skill = await skills.create_user_skill(
                user_id=1,
                key="shared_key",
                name="Private A",
                description="Only user 1 can see this",
            )
            other_skill = await skills.create_user_skill(
                user_id=2,
                key="shared_key",
                name="Private B",
                description="Only user 2 can see this",
            )

            with self.assertRaises(ValidationError):
                SkillCreate(
                    owner_type=SkillOwnerType.SYSTEM,
                    owner_user_id=1,
                    key="invalid",
                    name="Invalid",
                    description="Invalid ownership",
                )

            available_for_a = await skills.list_available_skills(user_id=1)
            available_for_b = await skills.list_available_skills(user_id=2)
            self.assertEqual(
                {skill.id for skill in available_for_a}, {system_skill.id, own_skill.id}
            )
            self.assertEqual(
                {skill.id for skill in available_for_b}, {system_skill.id, other_skill.id}
            )
            with self.assertRaises(LookupError):
                await skills.get_skill(user_id=2, skill_id=own_skill.id)

            assignment = await skills.assign_to_agent(
                user_id=1, agent_id=own_agent.id, skill_id=own_skill.id
            )
            duplicate = await skills.assign_to_agent(
                user_id=1, agent_id=own_agent.id, skill_id=own_skill.id
            )
            self.assertEqual(assignment.id, duplicate.id)
            self.assertEqual(
                [skill.id for skill in await skills.list_agent_skills(
                    user_id=1, agent_id=own_agent.id
                )],
                [own_skill.id],
            )
            with self.assertRaises(LookupError):
                await skills.assign_to_agent(
                    user_id=1, agent_id=other_agent.id, skill_id=own_skill.id
                )

            disabled = await skills.create_user_skill(
                user_id=1,
                key="disabled",
                name="Disabled",
                description="Inactive capability",
                status=SkillStatus.DISABLED,
            )
            archived = await skills.create_user_skill(
                user_id=1,
                key="archived",
                name="Archived",
                description="Archived capability",
            )
            await skills.archive_skill(user_id=1, skill_id=archived.id)
            for skill in (disabled, archived):
                with self.assertRaisesRegex(ValueError, "inactive"):
                    await skills.assign_to_agent(
                        user_id=1, agent_id=own_agent.id, skill_id=skill.id
                    )
            with self.assertRaises(LookupError):
                await skills.archive_skill(user_id=2, skill_id=own_skill.id)
            with self.assertRaises(LookupError):
                await skills.assign_to_agent(
                    user_id=2, agent_id=other_agent.id, skill_id=own_skill.id
                )

            self.assertTrue(
                await skills.remove_from_agent(
                    user_id=1, agent_id=own_agent.id, skill_id=own_skill.id
                )
            )
            self.assertEqual(
                await skills.list_agent_skills(user_id=1, agent_id=own_agent.id), []
            )

        asyncio.run(_with_session(scenario))

    def test_memory_scope_isolation_expiration_upsert_and_limit(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            agents = AgentService(session)
            primary = await agents.ensure_primary_agent(user_id=1)
            worker = await agents.create_user_agent(
                user_id=1, name="Worker", role="Worker", goal="Work"
            )
            other_user_agent = await agents.ensure_primary_agent(user_id=2)
            runs = AgentRunService(session)
            current_run = await runs.create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            other_run = await runs.create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            foreign_run = await runs.create_run(
                user_id=2, starting_agent_id=other_user_agent.id, model_name="test-model"
            )
            memories = MemoryService(session, max_items=20)

            global_memory = await memories.put(
                user_id=1,
                scope=MemoryScope.USER_GLOBAL,
                key="package_manager",
                content="Prefer uv",
            )
            updated_global = await memories.put(
                user_id=1,
                scope=MemoryScope.USER_GLOBAL,
                key="package_manager",
                content="Prefer uv for new Python projects",
            )
            self.assertEqual(global_memory.id, updated_global.id)
            self.assertEqual(updated_global.content, "Prefer uv for new Python projects")
            await memories.put(
                user_id=1,
                scope=MemoryScope.AGENT_PRIVATE,
                agent_id=primary.id,
                key="primary_rule",
                content="Primary-only memory",
            )
            await memories.put(
                user_id=1,
                scope=MemoryScope.AGENT_PRIVATE,
                agent_id=worker.id,
                key="worker_rule",
                content="Worker-only memory",
            )
            await memories.put(
                user_id=2,
                scope=MemoryScope.USER_GLOBAL,
                key="foreign_global",
                content="User 2 memory",
            )
            foreign_private = await memories.put(
                user_id=2,
                scope=MemoryScope.AGENT_PRIVATE,
                agent_id=other_user_agent.id,
                key="foreign_private",
                content="User 2 private memory",
            )
            await memories.put(
                user_id=1,
                scope=MemoryScope.CREW_SHARED,
                run_id=current_run.id,
                key="crew_note",
                content="Current crew context",
            )
            await memories.put(
                user_id=1,
                scope=MemoryScope.RUN_EPHEMERAL,
                run_id=current_run.id,
                key="scratch",
                content="Current run only",
            )
            await memories.put(
                user_id=1,
                scope=MemoryScope.CREW_SHARED,
                run_id=other_run.id,
                key="other_run_note",
                content="Other run context",
            )
            await memories.put(
                user_id=2,
                scope=MemoryScope.RUN_EPHEMERAL,
                run_id=foreign_run.id,
                key="foreign_run_note",
                content="Other tenant run context",
            )
            expired = await memories.put(
                user_id=1,
                scope=MemoryScope.USER_GLOBAL,
                key="expired",
                content="No longer relevant",
                expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            )

            primary_context = await memories.build_runtime_context(
                user_id=1, agent_id=primary.id, run_id=current_run.id
            )
            primary_items = {(item.scope, item.key, item.content) for item in primary_context}
            self.assertIn((MemoryScope.USER_GLOBAL, "package_manager", updated_global.content), primary_items)
            self.assertIn((MemoryScope.AGENT_PRIVATE, "primary_rule", "Primary-only memory"), primary_items)
            self.assertIn((MemoryScope.CREW_SHARED, "crew_note", "Current crew context"), primary_items)
            self.assertIn((MemoryScope.RUN_EPHEMERAL, "scratch", "Current run only"), primary_items)
            self.assertNotIn((MemoryScope.AGENT_PRIVATE, "worker_rule", "Worker-only memory"), primary_items)
            self.assertNotIn((MemoryScope.USER_GLOBAL, "foreign_global", "User 2 memory"), primary_items)
            self.assertNotIn((MemoryScope.AGENT_PRIVATE, "foreign_private", "User 2 private memory"), primary_items)
            self.assertNotIn((MemoryScope.CREW_SHARED, "other_run_note", "Other run context"), primary_items)
            self.assertNotIn((MemoryScope.RUN_EPHEMERAL, "foreign_run_note", "Other tenant run context"), primary_items)
            self.assertFalse(any(item.key == "expired" for item in primary_context))

            worker_context = await memories.build_runtime_context(
                user_id=1, agent_id=worker.id, run_id=current_run.id
            )
            worker_items = {(item.scope, item.key) for item in worker_context}
            self.assertIn((MemoryScope.AGENT_PRIVATE, "worker_rule"), worker_items)
            self.assertNotIn((MemoryScope.AGENT_PRIVATE, "primary_rule"), worker_items)

            with self.assertRaises(LookupError):
                await memories.get(user_id=2, memory_id=updated_global.id)
            self.assertFalse(any(item.id == updated_global.id for item in await memories.list(user_id=2)))
            self.assertGreaterEqual(await MemoryRepository(session).delete_expired(), 1)
            await session.commit()
            self.assertIsNone(await memories.get(user_id=1, memory_id=expired.id))
            self.assertLessEqual(
                len(
                    await MemoryService(session, max_items=2).build_runtime_context(
                        user_id=1, agent_id=primary.id, run_id=current_run.id
                    )
                ),
                2,
            )

            with self.assertRaises(ValidationError):
                MemoryCreate(
                    user_id=1,
                    scope=MemoryScope.USER_GLOBAL,
                    agent_id=primary.id,
                    key="invalid_scope",
                    content="Invalid scope fields",
                )

        asyncio.run(_with_session(scenario))

    def test_runtime_filters_skills_memory_and_builds_crew_context(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            agents = AgentService(session)
            primary = await agents.ensure_primary_agent(user_id=1)
            worker = await agents.create_user_agent(
                user_id=1, name="Python", role="Python expert", goal="Help with Python"
            )
            foreign_primary = await agents.ensure_primary_agent(user_id=2)
            skills = SkillService(session)
            active = await skills.create_user_skill(
                user_id=1,
                key="python_helper",
                name="Python Helper",
                description="Explain Python code and suggest focused fixes.",
                manifest={
                    "instructions": "Prefer small examples.",
                    "constraints": ["Never claim a tool permission was granted by this skill."],
                },
                required_permissions=["read:information:*"],
            )
            worker_skill = await skills.create_user_skill(
                user_id=1,
                key="worker_only",
                name="Worker Only",
                description="Only assigned to the worker.",
            )
            unassigned = await skills.create_user_skill(
                user_id=1,
                key="unassigned",
                name="Unassigned",
                description="Must not be present in runtime.",
            )
            archived = await skills.create_user_skill(
                user_id=1,
                key="archived",
                name="Archived",
                description="Must not be present in runtime.",
            )
            foreign = await skills.create_user_skill(
                user_id=2,
                key="foreign",
                name="Foreign",
                description="Must not cross tenants.",
            )
            await skills.assign_to_agent(user_id=1, agent_id=primary.id, skill_id=active.id)
            await skills.assign_to_agent(user_id=1, agent_id=worker.id, skill_id=worker_skill.id)
            await skills.assign_to_agent(user_id=1, agent_id=primary.id, skill_id=archived.id)
            await skills.archive_skill(user_id=1, skill_id=archived.id)

            system_skill = await skills.create_system_skill(
                key="web_research",
                name="Web Research",
                description="Research public sources.",
                required_permissions=["read:information:*"],
            )
            allowed_only = await skills.create_system_skill(
                key="allowed_only", name="Allowed Only", description="Not enabled by default."
            )
            await _configure_information_skills(
                session,
                allowed=[system_skill.key, allowed_only.key, foreign.key],
                default=[system_skill.key, "not_allowed"],
            )

            runs = AgentRunService(session)
            run = await runs.create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            other_run = await runs.create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            memories = MemoryService(session)
            await memories.put(
                user_id=1,
                scope=MemoryScope.USER_GLOBAL,
                key="preferred_language",
                content="Russian",
            )
            await memories.put(
                user_id=1,
                scope=MemoryScope.AGENT_PRIVATE,
                agent_id=primary.id,
                key="primary_style",
                content="Use concise answers",
            )
            await memories.put(
                user_id=1,
                scope=MemoryScope.AGENT_PRIVATE,
                agent_id=worker.id,
                key="worker_style",
                content="Worker private memory",
            )
            await memories.put(
                user_id=2,
                scope=MemoryScope.AGENT_PRIVATE,
                agent_id=foreign_primary.id,
                key="foreign_style",
                content="Foreign user memory",
            )
            await memories.put(
                user_id=1,
                scope=MemoryScope.CREW_SHARED,
                run_id=run.id,
                key="crew_goal",
                content="Summarize the source findings",
            )
            await memories.put(
                user_id=1,
                scope=MemoryScope.RUN_EPHEMERAL,
                run_id=run.id,
                key="run_note",
                content="Current run scratch note",
            )
            await memories.put(
                user_id=1,
                scope=MemoryScope.CREW_SHARED,
                run_id=other_run.id,
                key="other_run",
                content="Must not enter this run",
            )
            await memories.put(
                user_id=1,
                scope=MemoryScope.USER_GLOBAL,
                key="expired_note",
                content="Expired",
                expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            )

            runtime = RecordingRuntime([_respond("done")])
            request = AgentRuntimeRequest(
                run_id=run.id, user_id=1, agent_id=primary.id, message="Explain this code"
            )
            result = await AgentRuntimeService(session, runtime=runtime).execute(request)
            self.assertEqual(result.status, AgentRuntimeStatus.COMPLETED)
            context = runtime.contexts[0]
            self.assertEqual([item.key for item in context.active_skills], [active.key])
            self.assertEqual(
                [item.key for item in context.connected_persistent_agents[0].active_skills],
                [worker_skill.key],
            )
            information = next(
                item for item in context.available_system_agents
                if item.identity.subject_id == "information"
            )
            self.assertEqual([item.key for item in information.active_skills], [system_skill.key])
            self.assertNotIn(foreign.key, [item.key for item in information.active_skills])
            self.assertNotIn(archived.key, [item.key for item in context.active_skills])
            self.assertNotIn(unassigned.key, [item.key for item in context.active_skills])

            memory_keys = {(item.scope, item.key) for item in context.memory}
            self.assertIn((MemoryScope.USER_GLOBAL, "preferred_language"), memory_keys)
            self.assertIn((MemoryScope.AGENT_PRIVATE, "primary_style"), memory_keys)
            self.assertIn((MemoryScope.CREW_SHARED, "crew_goal"), memory_keys)
            self.assertIn((MemoryScope.RUN_EPHEMERAL, "run_note"), memory_keys)
            self.assertNotIn((MemoryScope.AGENT_PRIVATE, "worker_style"), memory_keys)
            self.assertNotIn((MemoryScope.AGENT_PRIVATE, "foreign_style"), memory_keys)
            self.assertNotIn((MemoryScope.CREW_SHARED, "other_run"), memory_keys)
            self.assertNotIn((MemoryScope.USER_GLOBAL, "expired_note"), memory_keys)

            task = DynamicCrewAIFactory._task_description(context, request)
            self.assertIn("Python Helper", task)
            self.assertIn("read:information:*", task)
            self.assertIn("Never claim a tool permission", task)
            self.assertIn("preferred_language: Russian", task)
            for excluded in ("Worker private memory", "Foreign user memory", "Must not enter this run"):
                self.assertNotIn(excluded, task)

        asyncio.run(_with_session(scenario))

    def test_system_delegation_and_temporary_subagent_skill_boundaries(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            agents = AgentService(session)
            primary = await agents.ensure_primary_agent(user_id=1)
            skills = SkillService(session)
            inherited = await skills.create_user_skill(
                user_id=1, key="inherited", name="Inherited", description="Parent skill"
            )
            not_assigned = await skills.create_user_skill(
                user_id=1, key="other", name="Other", description="Not assigned"
            )
            system_skill = await skills.create_system_skill(
                key="system_default", name="System Default", description="System capability"
            )
            allowed_only = await skills.create_system_skill(
                key="system_allowed", name="System Allowed", description="Not default"
            )
            await skills.assign_to_agent(user_id=1, agent_id=primary.id, skill_id=inherited.id)
            await _configure_information_skills(
                session,
                allowed=[system_skill.key, allowed_only.key],
                default=[system_skill.key],
            )

            runs = AgentRunService(session)
            system_run = await runs.create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            system_runtime = RecordingRuntime(
                [
                    RuntimeStep(
                        decision=DelegationDecision(
                            type=DelegationType.DELEGATE_SYSTEM_AGENT,
                            target_id="information",
                            task_summary="Research this question",
                        )
                    ),
                    _respond("research result"),
                    _respond("final result"),
                ]
            )
            system_result = await AgentRuntimeService(
                session, runtime=system_runtime
            ).execute(
                AgentRuntimeRequest(
                    run_id=system_run.id,
                    user_id=1,
                    agent_id=primary.id,
                    message="Research this question",
                )
            )
            self.assertEqual(system_result.status, AgentRuntimeStatus.COMPLETED)
            system_context = system_runtime.contexts[1]
            self.assertEqual(system_context.active_agent.kind, RuntimeAgentKind.SYSTEM)
            self.assertEqual([item.key for item in system_context.active_skills], [system_skill.key])
            self.assertNotIn(allowed_only.key, [item.key for item in system_context.active_skills])
            self.assertEqual(await agents.list_active_agents(1), [primary])

            temporary_run = await runs.create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            temporary_runtime = RecordingRuntime(
                [
                    RuntimeStep(
                        decision=DelegationDecision(
                            type=DelegationType.CREATE_TEMPORARY_SUBAGENT,
                            task_summary="Handle one focused step",
                            temporary_name="Temporary helper",
                            temporary_role="Helper",
                            temporary_goal="Complete the focused step",
                        )
                    ),
                    _respond("temporary result"),
                    _respond("final result"),
                ]
            )
            temporary_result = await AgentRuntimeService(
                session, runtime=temporary_runtime
            ).execute(
                AgentRuntimeRequest(
                    run_id=temporary_run.id,
                    user_id=1,
                    agent_id=primary.id,
                    message="Use a temporary helper",
                )
            )
            self.assertEqual(temporary_result.status, AgentRuntimeStatus.COMPLETED)
            temporary_context = temporary_runtime.contexts[1]
            self.assertEqual(temporary_context.active_agent.kind, RuntimeAgentKind.TEMPORARY)
            self.assertEqual([item.key for item in temporary_context.active_skills], [inherited.key])
            self.assertNotIn(not_assigned.key, [item.key for item in temporary_context.active_skills])
            self.assertEqual(
                temporary_context.temporary_subagents[0].active_skills,
                temporary_context.active_skills,
            )
            self.assertFalse(
                any(agent.kind == AgentKind.USER for agent in await agents.list_active_agents(1))
            )

        asyncio.run(_with_session(scenario))
