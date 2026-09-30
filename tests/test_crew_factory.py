import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from crewai import BaseLLM, Crew as CrewAICrew
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agents.assistant.crewai.factory import DynamicCrewAIFactory
from bot.handlers import crews as crew_handlers
from bot.handlers import messages as message_handlers
from database.base import Base
from database.bootstrap import _register_entities
from database.entities.user import UserEntity
from models.agent import AgentStatus
from models.agent_factory import AgentBlueprint, CrewBlueprint, CrewTaskBlueprint
from models.crew import CrewStatus
from models.memory import MemoryScope
from models.runtime import AgentRuntimeRequest
from services.agent import AgentService
from services.agent_design import AgentDesignService, is_crew_design_request
from services.agent_run import AgentRunService
from services.agent_runtime import AgentRuntimeService
from services.crew import CrewService
from services.memory import MemoryService
from services.permission import PermissionService
from services.skill import SkillService
from models.permission import ActionClass, PermissionSubjectType
from repositories.agent_memory_policy import AgentMemoryPolicyRepository
from bot.keyboards.crews import crew_design_keyboard


class FakeLLM(BaseLLM):
    def call(self, messages, **kwargs):
        return "unused"


def agent(name: str, **kwargs) -> AgentBlueprint:
    return AgentBlueprint(name=name, role=name, goal=f"Do {name}", **kwargs)


def crew_blueprint(**kwargs) -> CrewBlueprint:
    return CrewBlueprint.model_validate({
        "name": "Company Research",
        "purpose": "Research companies",
        "agents": [agent("Researcher").model_dump(mode="json"), agent("Analyst").model_dump(mode="json")],
        "tasks": [
            {"description": "Find companies", "expected_output": "Company list", "agent_index": 0},
            {"description": "Analyze companies", "expected_output": "Report", "agent_index": 1},
        ],
        **kwargs,
    })


async def with_session(test):
    _register_entities()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    @event.listens_for(engine.sync_engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            session.add_all([UserEntity(id=1, telegram_id=101), UserEntity(id=2, telegram_id=202)])
            await session.commit()
            await test(session)
    finally:
        await engine.dispose()


def test_crew_creation_persists_members_and_resolved_task_agent_ids() -> None:
    async def scenario(session):
        service = CrewService(session)
        created = await service.create_from_blueprint(user_id=1, blueprint=crew_blueprint())
        assert created.status == CrewStatus.ACTIVE
        assert created.process.value == "sequential"
        members = await service.list_members(user_id=1, crew_id=created.id)
        tasks = await service.list_tasks(user_id=1, crew_id=created.id)
        assert len(members) == len(tasks) == 2
        assert [task.agent_id for task in tasks] == [member.agent_id for member in members]
        assert not hasattr(tasks[0], "agent_index")
        definition = await service.build_runtime_definition(user_id=1, crew_id=created.id)
        assert definition.id == created.id
        assert [item.agent_id for item in definition.tasks] == [member.agent_id for member in members]
        runtime = DynamicCrewAIFactory(llm=FakeLLM(model="fake"))
        assert isinstance(runtime.build_from_definition(definition), CrewAICrew)

    asyncio.run(with_session(scenario))


def test_foreign_crew_and_foreign_agent_are_unavailable() -> None:
    async def scenario(session):
        service = CrewService(session)
        foreign = await AgentService(session).create_user_agent(
            user_id=2, name="Private", role="Private", goal="Private research",
        )
        blueprint = CrewBlueprint(
            name="Stolen", purpose="Invalid", existing_agent_ids=[foreign.id],
            tasks=[CrewTaskBlueprint(
                description="Read", expected_output="Result", agent_index=0,
            )],
        )
        with pytest.raises(LookupError):
            await service.create_from_blueprint(user_id=1, blueprint=blueprint)
        assert await service.list(user_id=1) == []
        own = await service.create_from_blueprint(user_id=2, blueprint=blueprint)
        assert await service.list(user_id=1) == []
        with pytest.raises(LookupError):
            await service.get(user_id=1, crew_id=own.id)
        with pytest.raises(LookupError):
            await service.build_runtime_definition(user_id=1, crew_id=own.id)
        with pytest.raises(LookupError):
            await service.archive(user_id=1, crew_id=own.id)

    asyncio.run(with_session(scenario))


def test_existing_user_agent_membership_and_required_skill() -> None:
    async def scenario(session):
        existing = await AgentService(session).create_user_agent(
            user_id=1, name="Researcher", role="Researcher", goal="Find companies",
        )
        skill = await SkillService(session).create_user_skill(
            user_id=1, key="company-search", name="Company Search", description="Search companies",
        )
        await SkillService(session).assign_to_agent(
            user_id=1, agent_id=existing.id, skill_id=skill.id,
        )
        blueprint = CrewBlueprint(
            name="Existing Crew", purpose="Research", existing_agent_ids=[existing.id],
            required_skill_ids=[skill.id],
            tasks=[CrewTaskBlueprint(
                description="Find companies", expected_output="List", agent_index=0,
            )],
        )
        design = AgentDesignService(session, runner=SimpleNamespace(
            design_crew=AsyncMock(return_value=blueprint),
        ))
        proposal = await design.design_crew(user_id=1, user_request="Create a crew")
        assert proposal.existing_agent_ids == [existing.id]
        assert await CrewService(session).list(user_id=1) == []
        crew = await CrewService(session).create_from_blueprint(user_id=1, blueprint=proposal)
        members = await CrewService(session).list_members(user_id=1, crew_id=crew.id)
        assert [member.agent_id for member in members] == [existing.id]
        assert [item.skill_id for item in await CrewService(session).list_skills(
            user_id=1, crew_id=crew.id,
        )] == [skill.id]
        definition = await CrewService(session).build_runtime_definition(
            user_id=1, crew_id=crew.id,
        )
        assert definition.required_skill_ids == [skill.id]
        await SkillService(session).archive_skill(user_id=1, skill_id=skill.id)
        with pytest.raises(ValueError, match="required crew skill"):
            await CrewService(session).build_runtime_definition(
                user_id=1, crew_id=crew.id,
            )

    asyncio.run(with_session(scenario))


def test_existing_member_permission_escalation_is_rejected() -> None:
    async def scenario(session):
        existing = await AgentService(session).create_user_agent(
            user_id=1, name="Researcher", role="Researcher", goal="Find companies",
        )
        await PermissionService(session).grant(
            user_id=1,
            subject_type=PermissionSubjectType.PERSISTENT_AGENT,
            subject_id=str(existing.id),
            action_class=ActionClass.READ,
            resource_scope="private:*",
        )
        blueprint = CrewBlueprint(
            name="Escalation", purpose="Invalid", existing_agent_ids=[existing.id],
            tasks=[CrewTaskBlueprint(
                description="Read private data", expected_output="Data", agent_index=0,
            )],
        )
        with pytest.raises(PermissionError, match="exceed primary"):
            await CrewService(session).create_from_blueprint(user_id=1, blueprint=blueprint)
        assert await CrewService(session).list(user_id=1) == []

    asyncio.run(with_session(scenario))


def test_duplicate_existing_agents_and_invalid_process_are_rejected() -> None:
    async def scenario(session):
        existing = await AgentService(session).create_user_agent(
            user_id=1, name="Researcher", role="Researcher", goal="Find companies",
        )
        blueprint = CrewBlueprint(
            name="Duplicate", purpose="Invalid", existing_agent_ids=[existing.id, existing.id],
            tasks=[CrewTaskBlueprint(
                description="Find", expected_output="List", agent_index=0,
            )],
        )
        with pytest.raises(ValueError, match="Duplicate existing"):
            await CrewService(session).create_from_blueprint(user_id=1, blueprint=blueprint)
        with pytest.raises(ValueError):
            CrewBlueprint.model_validate({
                **blueprint.model_dump(mode="json"), "process": "hierarchical",
            })

    asyncio.run(with_session(scenario))


def test_archive_crew_keeps_member_agents_active() -> None:
    async def scenario(session):
        service = CrewService(session)
        created = await service.create_from_blueprint(user_id=1, blueprint=crew_blueprint())
        member_id = (await service.list_members(user_id=1, crew_id=created.id))[0].agent_id
        archived = await service.archive(user_id=1, crew_id=created.id)
        assert archived.status == CrewStatus.ARCHIVED
        assert (await AgentService(session).get_agent(user_id=1, agent_id=member_id)).status == AgentStatus.ACTIVE
        with pytest.raises(ValueError, match="Archived"):
            await service.build_runtime_definition(user_id=1, crew_id=created.id)

    asyncio.run(with_session(scenario))


def test_crew_transaction_rolls_back_agent_creation_on_task_failure() -> None:
    async def scenario(session):
        service = CrewService(session)
        with patch.object(service._crews, "add_task", AsyncMock(side_effect=RuntimeError("storage failure"))):
            with pytest.raises(RuntimeError, match="storage failure"):
                await service.create_from_blueprint(user_id=1, blueprint=crew_blueprint())
        assert await service.list(user_id=1) == []
        assert await AgentService(session).list_active_agents(1) == []

    asyncio.run(with_session(scenario))


def test_missing_capabilities_and_invalid_required_skill_block_crew() -> None:
    async def scenario(session):
        service = CrewService(session)
        with pytest.raises(ValueError, match="missing capabilities"):
            await service.create_from_blueprint(user_id=1, blueprint=crew_blueprint(
                missing_capabilities=["github read"],
            ))
        with pytest.raises(ValueError, match="unavailable"):
            await service.create_from_blueprint(user_id=1, blueprint=crew_blueprint(
                required_skill_ids=[uuid4()],
            ))
        assert await service.list(user_id=1) == []

    asyncio.run(with_session(scenario))


def test_memory_policy_persists_and_filters_agent_runtime_context() -> None:
    async def scenario(session):
        service = AgentService(session)
        created = await service.create_from_blueprint(user_id=1, blueprint=agent(
            "Private Reader", memory_scopes=[MemoryScope.AGENT_PRIVATE],
        ))
        run = await AgentRunService(session).create_run(
            user_id=1, starting_agent_id=created.id, model_name="fake",
        )
        memory = MemoryService(session)
        await memory.put(user_id=1, scope=MemoryScope.USER_GLOBAL, key="global", content="global fact")
        await memory.put(user_id=1, scope=MemoryScope.AGENT_PRIVATE, agent_id=created.id,
                         key="private", content="private fact")
        result = await memory.build_runtime_context(user_id=1, run_id=run.id, agent_id=created.id)
        assert [item.key for item in result] == ["private"]
        runtime = AgentRuntimeService(session, runtime=SimpleNamespace(run=AsyncMock()))
        context = await runtime._build_context(
            AgentRuntimeRequest(run_id=run.id, user_id=1, agent_id=created.id, message="facts"),
            run.prompt_version_id,
        )
        assert context.starting_agent.memory_scopes == [MemoryScope.AGENT_PRIVATE]
        assert [item.key for item in context.memory] == ["private"]

    asyncio.run(with_session(scenario))


def test_disabled_memory_policy_stops_semantic_lookup() -> None:
    async def scenario(session):
        created = await AgentService(session).create_from_blueprint(
            user_id=1, blueprint=agent("Restricted"),
        )
        run = await AgentRunService(session).create_run(
            user_id=1, starting_agent_id=created.id, model_name="fake",
        )
        await AgentMemoryPolicyRepository(session).set_scopes(created.id, [])
        await session.commit()
        embedding = SimpleNamespace(embed=AsyncMock(return_value=[0.1]))
        index = SimpleNamespace(search=AsyncMock(return_value=[]))
        result = await MemoryService(
            session, embedding_provider=embedding, search_index=index,
        ).search_runtime_memory(
            user_id=1, run_id=run.id, agent_id=created.id, query="secret",
        )
        assert result == []
        embedding.embed.assert_not_awaited()
        index.search.assert_not_awaited()

    asyncio.run(with_session(scenario))


def test_missing_capability_preview_disables_creation_button() -> None:
    async def scenario(session):
        blueprint = crew_blueprint(missing_capabilities=["repository access"])
        text = await AgentDesignService(session).crew_preview_text(user_id=1, blueprint=blueprint)
        keyboard = crew_design_keyboard("abc12345", can_create=False)
        assert "repository access" in text
        assert "Creation is blocked" in text
        assert [button.text for button in keyboard.inline_keyboard[0]] == ["Cancel"]

    asyncio.run(with_session(scenario))


def test_design_context_and_budget_instruction_reach_every_factory_role() -> None:
    from agents.factory.crewai.crew import AgentFactoryCrew
    from models.agent_factory import AgentFactoryInput, AvailableSkill, RequestedPermission
    from models.permission import ActionClass

    skill_id = uuid4()
    request = AgentFactoryInput(
        user_request="Create a researcher",
        available_skills=[AvailableSkill(
            id=skill_id, key="search", name="Search", description="Search docs",
        )],
        parent_permissions=[RequestedPermission(
            action_class=ActionClass.READ, resource_scope="docs:*",
        )],
        parent_can_spawn_subagents=True,
    )
    built = AgentFactoryCrew(llm=FakeLLM(model="fake")).build(request)
    for task in built.tasks:
        assert str(skill_id) in task.description
        assert "docs:*" in task.description
        assert "max_agent_iterations" in task.description
        assert "budget_profile must be null" in task.description


def test_crew_routing_and_telegram_preview_confirmation_cancel_stale() -> None:
    class State:
        def __init__(self):
            self.data = {}
            self.current = None

        async def update_data(self, **values):
            self.data.update(values)

        async def set_state(self, value):
            self.current = value

        async def get_data(self):
            return self.data

        async def clear(self):
            self.data.clear()
            self.current = None

    async def scenario():
        assert is_crew_design_request("Создай мне команду для исследования компаний")
        blueprint = crew_blueprint()
        design = SimpleNamespace(
            design_crew=AsyncMock(return_value=blueprint),
            crew_preview_text=AsyncMock(return_value="Crew: Company Research\nMissing capabilities: none"),
        )
        message = SimpleNamespace(
            text="Создай мне команду для исследования компаний",
            from_user=SimpleNamespace(id=101, first_name="Ada"),
            chat=SimpleNamespace(id=501),
            bot=SimpleNamespace(send_chat_action=AsyncMock()),
            answer=AsyncMock(),
        )
        state = State()
        with (
            patch.object(message_handlers, "resolve_internal_user", AsyncMock(return_value=SimpleNamespace(id=1))),
            patch.object(message_handlers, "resolve_telegram_chat", AsyncMock(return_value=SimpleNamespace(id=uuid4()))),
            patch.object(message_handlers, "AgentDesignService", return_value=design),
            patch.object(message_handlers, "AssistantService") as assistant,
        ):
            await message_handlers.handle_text_message(message, object(), state)
        assert "Crew: Company Research" in message.answer.await_args.args[0]
        assert state.data["blueprint"]["name"] == "Company Research"
        assistant.assert_not_called()

        created = SimpleNamespace(name="Company Research")
        service = SimpleNamespace(create_from_blueprint=AsyncMock(return_value=created))
        token = state.data["wizard_id"]
        callback = SimpleNamespace(
            data=f"crew:design:confirm:{token}",
            from_user=message.from_user,
            message=SimpleNamespace(answer=AsyncMock()),
            answer=AsyncMock(),
        )
        with (
            patch.object(crew_handlers, "resolve_internal_user", AsyncMock(return_value=SimpleNamespace(id=1))),
            patch.object(crew_handlers, "CrewService", return_value=service),
        ):
            callback.data = "crew:design:confirm:stale"
            await crew_handlers.decide_crew_design(callback, state, object())
            service.create_from_blueprint.assert_not_awaited()
            callback.data = f"crew:design:confirm:{token}"
            await crew_handlers.decide_crew_design(callback, state, object())
            service.create_from_blueprint.assert_awaited_once()
        assert state.data == {}

        state = State()
        await state.update_data(wizard_id="cancelme", blueprint=blueprint.model_dump(mode="json"))
        callback.data = "crew:design:cancel:cancelme"
        with patch.object(crew_handlers, "CrewService") as forbidden:
            await crew_handlers.decide_crew_design(callback, state, object())
        forbidden.assert_not_called()
        assert state.data == {}

    asyncio.run(scenario())


def test_telegram_confirmation_persists_crew_and_builds_runtime_definition() -> None:
    class State:
        def __init__(self):
            self.data = {}
            self.current = None

        async def update_data(self, **values):
            self.data.update(values)

        async def set_state(self, value):
            self.current = value

        async def get_data(self):
            return self.data

        async def clear(self):
            self.data.clear()
            self.current = None

    async def scenario(session):
        blueprint = crew_blueprint()
        design = AgentDesignService(session, runner=SimpleNamespace(
            design_crew=AsyncMock(return_value=blueprint),
        ))
        user = SimpleNamespace(id=1)
        message = SimpleNamespace(
            text="Создай мне команду для исследования компаний",
            from_user=SimpleNamespace(id=101, first_name="Ada"),
            chat=SimpleNamespace(id=501),
            bot=SimpleNamespace(send_chat_action=AsyncMock()),
            answer=AsyncMock(),
        )
        state = State()
        with (
            patch.object(message_handlers, "resolve_internal_user", AsyncMock(return_value=user)),
            patch.object(message_handlers, "resolve_telegram_chat", AsyncMock(return_value=SimpleNamespace(id=uuid4()))),
            patch.object(message_handlers, "AgentDesignService", return_value=design),
        ):
            await message_handlers.handle_text_message(message, session, state)
        assert await CrewService(session).list(user_id=1) == []
        callback = SimpleNamespace(
            data=f"crew:design:confirm:{state.data['wizard_id']}",
            from_user=message.from_user,
            message=SimpleNamespace(answer=AsyncMock()),
            answer=AsyncMock(),
        )
        with patch.object(crew_handlers, "resolve_internal_user", AsyncMock(return_value=user)):
            await crew_handlers.decide_crew_design(callback, state, session)
        crews = await CrewService(session).list(user_id=1)
        assert len(crews) == 1
        definition = await CrewService(session).build_runtime_definition(
            user_id=1, crew_id=crews[0].id,
        )
        assert len(definition.agents) == len(definition.tasks) == 2
        assert isinstance(
            DynamicCrewAIFactory(llm=FakeLLM(model="fake")).build_from_definition(definition),
            CrewAICrew,
        )

    asyncio.run(with_session(scenario))
