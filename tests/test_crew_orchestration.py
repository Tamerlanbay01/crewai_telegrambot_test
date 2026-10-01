import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agents.assistant.crewai.factory import DynamicCrewAIFactory
from database.base import Base
from database.bootstrap import _register_entities
from database.entities.user import UserEntity
from models.agent_factory import AgentBlueprint, CrewBlueprint, CrewTaskBlueprint
from models.agent_run import AgentRunStatus
from models.runtime import (
    AgentRuntimeContext, AgentRuntimeRequest, AgentRuntimeStatus, DelegationDecision,
    DelegationType, RuntimeBudgets, RuntimeStep,
)
from services.agent import AgentService
from services.agent_run import AgentRunService
from services.agent_runtime import AgentRuntimeService
from services.assistant import AssistantService
from services.chat import ChatService
from services.crew import CrewService


async def with_session(scenario):
    _register_entities()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            session.add_all([UserEntity(id=1, telegram_id=101), UserEntity(id=2, telegram_id=202)])
            await session.commit()
            await scenario(session)
    finally:
        await engine.dispose()


async def create_crew(session, *, user_id=1, name="Research Crew"):
    return await CrewService(session).create_from_blueprint(
        user_id=user_id, blueprint=CrewBlueprint(
            name=name, purpose="Research and summarize a topic",
            agents=[AgentBlueprint(name="Researcher", role="Researcher", goal="Research")],
            tasks=[CrewTaskBlueprint(
                description="Research the provided topic", expected_output="Research notes", agent_index=0,
            )],
        ),
    )


async def create_context(session, runtime, *, skills=None):
    primary = await AgentService(session).ensure_primary_agent(user_id=1)
    run = await AgentRunService(session).create_run(
        user_id=1, starting_agent_id=primary.id, model_name="fake",
    )
    await AgentRunService(session).start(user_id=1, run_id=run.id)
    request = AgentRuntimeRequest(
        user_id=1, run_id=run.id, agent_id=primary.id, message="Research batteries",
    )
    service = AgentRuntimeService(session, runtime=runtime)
    if skills is not None:
        service._skills = skills
        service._crews._skills = skills
    context = await service._build_context(request, run.prompt_version_id)
    return service, context, request


class RecordingRuntime:
    def __init__(self, target_id, *, error=None):
        self.target_id = str(target_id)
        self.inputs = []
        self.crew_calls = []
        self.error = error

    async def run(self, context, request):
        self.inputs.append(context.current_input)
        if len(self.inputs) == 1:
            return RuntimeStep.model_validate({"decision": {
                "type": "run_crew", "target_id": self.target_id,
                "task_summary": "Research batteries for the user",
            }})
        assert "Crew research notes" in context.current_input
        return RuntimeStep(
            decision=DelegationDecision(type=DelegationType.RESPOND), content="Primary synthesis",
        )

    async def run_crew(self, context, definition, *, task_summary):
        self.crew_calls.append((context.user_id, definition, task_summary))
        if self.error:
            raise self.error
        return {"content": "Crew research notes", "tokens_used": 15, "llm_calls": 1}


def test_active_own_crew_catalog_is_snapshotted_and_only_exposed_to_primary():
    async def scenario(session):
        own = await create_crew(session)
        archived = await create_crew(session, name="Old Crew")
        await CrewService(session).archive(user_id=1, crew_id=archived.id)
        foreign = await create_crew(session, user_id=2, name="Private Crew")
        service, context, request = await create_context(session, SimpleNamespace(run=None))
        assert [item.id for item in context.available_crews] == [own.id]
        prompt = DynamicCrewAIFactory._task_description(context, request)
        assert str(own.id) in prompt and own.purpose in prompt
        assert str(archived.id) not in prompt and str(foreign.id) not in prompt
        await service._persist(context)
        new = await create_crew(session, name="Later Crew")
        persisted = await AgentRunService(session).get_run(user_id=1, run_id=context.run_id)
        pinned = AgentRuntimeContext.model_validate(persisted.checkpoint)
        assert [item.id for item in pinned.available_crews] == [own.id]
        _, fresh, _ = await create_context(session, SimpleNamespace(run=None))
        assert {item.id for item in fresh.available_crews} == {own.id, new.id}
        worker = pinned.connected_persistent_agents[0]
        pinned.active_agent = worker.identity
        pinned.active_skills = worker.active_skills
        worker_prompt = DynamicCrewAIFactory._task_description(pinned, request)
        assert str(own.id) not in worker_prompt and own.purpose not in worker_prompt
        old = context.model_dump(mode="json")
        old.pop("available_crews")
        assert AgentRuntimeContext.model_validate(old).available_crews == []

    asyncio.run(with_session(scenario))


def test_chat_run_crew_returns_to_primary_and_persists_usage():
    async def scenario(session):
        crew = await create_crew(session)
        chat = await ChatService(session).create_chat(user_id=1, title="Crew dialogue")
        runtime = RecordingRuntime(crew.id)
        result = await AssistantService(session, runtime=runtime).handle_message(
            user_id=1, chat_id=chat.id, text="Use Research Crew to research batteries",
        )
        assert result.content == "Primary synthesis"
        assert len(runtime.crew_calls) == 1 and len(runtime.inputs) == 2
        user_id, definition, summary = runtime.crew_calls[0]
        assert user_id == 1 and definition.id == crew.id and summary == "Research batteries for the user"
        run = await AgentRunService(session).get_run(user_id=1, run_id=result.run_id)
        assert run.status == AgentRunStatus.COMPLETED
        assert run.usage["delegations"] == 1 and run.usage["tokens"] == 15
        assert run.usage["llm_calls"] == 3
        assert run.checkpoint["active_agent"]["kind"] == "primary"
        assert run.checkpoint["delegation_stack"] == []

    asyncio.run(with_session(scenario))


@pytest.mark.parametrize("target", ["unknown", "foreign", "archived", "worker", "late", "invalid"])
def test_run_crew_rejects_unavailable_targets_before_kickoff(target):
    async def scenario(session):
        own = await create_crew(session)
        foreign = await create_crew(session, user_id=2)
        runtime = RecordingRuntime(own.id)
        service, context, request = await create_context(session, runtime)
        if target == "unknown":
            runtime.target_id = str(uuid4())
        elif target == "foreign":
            runtime.target_id = str(foreign.id)
        elif target == "archived":
            await CrewService(session).archive(user_id=1, crew_id=own.id)
        elif target == "worker":
            context.active_agent = context.connected_persistent_agents[0].identity
        elif target == "late":
            runtime.target_id = str((await create_crew(session, name="Too Late")).id)
        else:
            runtime.target_id = "not-a-uuid"
        await service._persist(context)
        result = await AgentRuntimeService(session, runtime=runtime).execute(request)
        assert result.status == AgentRuntimeStatus.FAILED
        assert runtime.crew_calls == []

    asyncio.run(with_session(scenario))


def test_forged_catalog_does_not_bypass_crew_ownership_validation():
    async def scenario(session):
        foreign = await create_crew(session, user_id=2)
        runtime = RecordingRuntime(foreign.id)
        service, context, request = await create_context(session, runtime)
        from models.runtime import RuntimeCrewSummary
        context.available_crews.append(RuntimeCrewSummary(
            id=foreign.id, name=foreign.name, purpose=foreign.purpose,
        ))
        result = await service._continue(context, request)
        assert result.status == AgentRuntimeStatus.FAILED
        assert "Crew not found" in result.error and runtime.crew_calls == []

    asyncio.run(with_session(scenario))


def test_actual_crewai_crew_kickoff_uses_native_skills_and_returns_to_primary():
    from crewai import BaseLLM
    from agents.assistant.crewai.runtime import DynamicCrewAIRuntime
    from agents.assistant.crewai.skill_runtime import SkillRuntimeResolver
    from agents.assistant.crewai.tool_runtime import ToolRuntimeResolver
    from models.agent_factory import RequestedPermission
    from models.permission import ActionClass, PermissionSubjectType
    from models.tool import ToolDefinition
    from services.permission import PermissionService
    from services.skill import SkillService
    from services.tool_executor import RegisteredTool, ToolRegistry
    from test_native_skill_runtime import Storage, package
    import json

    async def scenario(session):
        storage = Storage()
        skills = SkillService(session, storage=storage)
        primary = await AgentService(session).ensure_primary_agent(user_id=1)
        await PermissionService(session).grant(
            user_id=1, subject_type=PermissionSubjectType.PERSISTENT_AGENT,
            subject_id=str(primary.id), action_class=ActionClass.READ, resource_scope="information:*",
        )
        skill = await skills.upload_user_skill(
            user_id=1, key="crew-native", name="Crew Native", description="Crew instructions",
            package=package("crew-native", tools=[
                {"id": "read_information", "resource": "information:crew"},
            ]),
        )
        crew = await CrewService(session).create_from_blueprint(user_id=1, blueprint=CrewBlueprint(
            name="Native Research", purpose="Native research",
            agents=[AgentBlueprint(name="Reader", role="Crew Reader", goal="Research",
                                   skill_ids=[skill.id], requested_permissions=[RequestedPermission(
                                       action_class=ActionClass.READ, resource_scope="information:crew",
                                   )])],
            tasks=[CrewTaskBlueprint(description="Research the supplied input", expected_output="Report",
                                     agent_index=0)],
        ))
        owner_loop = asyncio.get_running_loop()
        seen_paths = []
        reads = []

        async def read_information(arguments):
            assert asyncio.get_running_loop() is owner_loop
            assert seen_paths[0].is_dir()
            assert await session.get(UserEntity, 1) is not None
            reads.append(arguments["query"])
            return {"source": "backend", "result": "Crew source"}

        class LLM(BaseLLM):
            def __init__(self):
                super().__init__(model="fake")
                self.primary_calls = 0
                self.reader_calls = 0

            def supports_function_calling(self):
                return False

            def call(self, messages, **kwargs):
                member = kwargs["from_agent"]
                if member.role == "Personal Assistant":
                    if self.primary_calls == 0:
                        self.primary_calls += 1
                        return json.dumps({"decision": {
                            "type": "run_crew", "target_id": str(crew.id),
                            "task_summary": "Research batteries",
                        }})
                    assert "Native crew report" in str(messages)
                    return json.dumps({"decision": {"type": "respond"}, "content": "Primary synthesis"})
                assert "Current crew input: Research batteries" in str(messages)
                if self.reader_calls == 0:
                    seen_paths.append(member.skills[0].path)
                    response = 'Thought: Load skill.\nAction: load_skill\nAction Input: ' + json.dumps({
                        "skill_name": member.skills[0].name,
                    })
                elif self.reader_calls == 1:
                    assert "Follow the documented workflow." in str(messages)
                    response = ('Thought: Read source.\nAction: read_information\n'
                                'Action Input: {"arguments": {"query": "batteries"}}')
                else:
                    assert "Crew source" in str(messages)
                    response = "Thought: Complete.\nFinal Answer: Native crew report"
                self.reader_calls += 1
                return response

        llm = LLM()
        registry = ToolRegistry([RegisteredTool(ToolDefinition(
            name="read_information", action_class=ActionClass.READ, resource_prefix="information:",
        ), read_information)])
        runtime = DynamicCrewAIRuntime(
            factory=DynamicCrewAIFactory(llm=llm), skill_resolver=SkillRuntimeResolver(skills),
            tool_resolver=ToolRuntimeResolver(session, executor=registry),
        )
        service, context, request = await create_context(session, runtime, skills=skills)
        # Existing usage must leave only one backend call for the crew.
        context.budgets.max_tool_calls = 2
        context.usage.tool_calls = 1
        result = await service._continue(context, request)
        assert result.status == AgentRuntimeStatus.COMPLETED and result.content == "Primary synthesis"
        assert reads == ["batteries"] and context.usage.tool_calls == 2
        assert llm.reader_calls == 3 and not seen_paths[0].exists()
        persisted = await CrewService(session).list_tasks(user_id=1, crew_id=crew.id)
        assert persisted[0].description == "Research the supplied input"

    asyncio.run(with_session(scenario))


def test_run_crew_propagates_runtime_failure_without_final_synthesis():
    async def scenario(session):
        crew = await create_crew(session)
        runtime = RecordingRuntime(crew.id, error=RuntimeError("Crew failed"))
        service, context, request = await create_context(session, runtime)
        result = await service._continue(context, request)
        assert result.status == AgentRuntimeStatus.FAILED and result.error == "Crew failed"
        assert len(runtime.inputs) == len(runtime.crew_calls) == 1

    asyncio.run(with_session(scenario))


def test_crew_llm_budget_blocks_forced_final_request_before_provider_call():
    from crewai import BaseLLM
    from agents.assistant.crewai.runtime import DynamicCrewAIRuntime
    from services.agent_runtime import RuntimeBudgetExceededError

    async def scenario(session):
        crew = await create_crew(session)
        definition = await CrewService(session).build_runtime_definition(user_id=1, crew_id=crew.id)

        class LLM(BaseLLM):
            def __init__(self):
                super().__init__(model="fake")
                self.calls = 0

            def supports_function_calling(self):
                return False

            def call(self, messages, **kwargs):
                self.calls += 1
                return 'Thought: Need a tool.\nAction: unknown_tool\nAction Input: {}'

        llm = LLM()
        runtime = DynamicCrewAIRuntime(factory=DynamicCrewAIFactory(llm=llm))
        _, context, _ = await create_context(session, runtime)
        context.budgets.max_llm_calls = 1
        with pytest.raises(RuntimeBudgetExceededError):
            await runtime.run_crew(context, definition, task_summary="Research batteries")
        assert llm.calls == 1
        assert context.usage.llm_calls == 1

    asyncio.run(with_session(scenario))


@pytest.mark.parametrize("budget", [RuntimeBudgets(max_delegations=0), RuntimeBudgets(max_llm_calls=1)])
def test_run_crew_respects_remaining_run_budgets_before_kickoff(budget):
    async def scenario(session):
        crew = await create_crew(session)
        runtime = RecordingRuntime(crew.id)
        service, context, request = await create_context(session, runtime)
        context.budgets = budget
        result = await service._continue(context, request)
        assert result.status == AgentRuntimeStatus.FAILED
        assert runtime.crew_calls == []

    asyncio.run(with_session(scenario))
