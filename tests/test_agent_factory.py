import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from crewai import Agent as CrewAIAgent, BaseLLM, Crew as CrewAICrew
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agents.assistant.crewai.factory import DynamicCrewAIFactory
from bot.handlers import agents as agent_handlers
from agents.factory.crewai.crew import AgentFactoryCrew
from agents.factory.runner import AgentFactoryRunner, FactoryOutputError
from database.base import Base
from database.bootstrap import _register_entities
from database.entities.user import UserEntity
from models.agent import AgentStatus
from models.crew import RuntimeCrewTask
from models.agent_factory import (
    AgentBlueprint, AgentFactoryInput, CrewBlueprint, CrewDefinition, CrewTaskBlueprint,
    RequestedPermission,
)
from models.memory import MemoryScope
from models.permission import ActionClass, PermissionSubjectType
from models.runtime import RuntimeBudgets
from services.agent import AgentService
from services.agent_design import AgentDesignService, is_agent_design_request
from services.agent_runtime import AgentRuntimeService
from services.permission import PermissionService
from services.skill import SkillService


def blueprint(**updates) -> AgentBlueprint:
    return AgentBlueprint.model_validate({
        "name": "Python Developer",
        "role": "Developer",
        "goal": "Research Python documentation",
        **updates,
    })


def test_runner_parses_structured_output_without_exposing_crewai() -> None:
    async def scenario() -> None:
        expected = blueprint(missing_capabilities=["repository access"])
        built_crew = SimpleNamespace(kickoff_async=AsyncMock(return_value=SimpleNamespace(
            pydantic=None, json_dict=expected.model_dump(mode="json"), raw="",
        )))
        factory = SimpleNamespace(build=lambda request, crew=False: built_crew)
        result = await AgentFactoryRunner(factory=factory).design_agent(
            AgentFactoryInput(user_request="Create a Python agent", parent_can_spawn_subagents=True)
        )
        assert isinstance(result, AgentBlueprint)
        assert result.missing_capabilities == ["repository access"]
        assert result.skill_ids == []

    asyncio.run(scenario())


def test_runner_rejects_malformed_output() -> None:
    async def scenario() -> None:
        crew = SimpleNamespace(kickoff_async=AsyncMock(return_value=SimpleNamespace(
            pydantic=None, json_dict=None, raw="not json",
        )))
        factory = SimpleNamespace(build=lambda request: crew)
        with pytest.raises(FactoryOutputError):
            await AgentFactoryRunner(factory=factory).design_agent(
                AgentFactoryInput(user_request="Create agent", parent_can_spawn_subagents=True)
            )

    asyncio.run(scenario())


def test_runner_propagates_factory_execution_failure() -> None:
    async def scenario() -> None:
        built = SimpleNamespace(kickoff_async=AsyncMock(side_effect=RuntimeError("LLM unavailable")))
        runner = AgentFactoryRunner(factory=SimpleNamespace(build=lambda request: built))
        with pytest.raises(RuntimeError, match="LLM unavailable"):
            await runner.design_agent(AgentFactoryInput(
                user_request="Create an agent", parent_can_spawn_subagents=True,
            ))

    asyncio.run(scenario())


def test_runner_parses_crew_blueprint_and_service_checks_task_indices() -> None:
    async def scenario() -> None:
        proposal = CrewBlueprint(
            name="Research", purpose="Research topics", agents=[blueprint()],
            tasks=[CrewTaskBlueprint(
                description="Research", expected_output="Summary", agent_index=1,
            )],
        )
        built = SimpleNamespace(kickoff_async=AsyncMock(return_value=SimpleNamespace(
            pydantic=proposal, json_dict=None, raw="",
        )))
        runner = AgentFactoryRunner(factory=SimpleNamespace(
            build=lambda request, crew=False: built,
        ))
        parsed = await runner.design_crew(AgentFactoryInput(
            user_request="Create a crew", parent_can_spawn_subagents=True,
        ))
        assert isinstance(parsed, CrewBlueprint)
        service = AgentDesignService.__new__(AgentDesignService)
        service._factory_input = AsyncMock(return_value=AgentFactoryInput(
            user_request="Create a crew", parent_can_spawn_subagents=True,
        ))
        service._runner = SimpleNamespace(design_crew=AsyncMock(return_value=parsed))
        service._agents = SimpleNamespace(validate_blueprint=AsyncMock())
        with pytest.raises(ValueError, match="Crew task"):
            await service.design_crew(user_id=1, user_request="Create a crew")

    asyncio.run(scenario())


def test_design_crew_has_no_repository_imports() -> None:
    root = Path(__file__).resolve().parents[1] / "app" / "agents" / "factory"
    for path in root.rglob("*.py"):
        module = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.ImportFrom) and (node.module or "").startswith("repositories")
            for node in ast.walk(module)
        )


def test_design_crew_uses_three_roles_and_structured_final_task() -> None:
    request = AgentFactoryInput(user_request="Create an agent", parent_can_spawn_subagents=True)
    crew = AgentFactoryCrew(llm=FakeLLM(model="fake")).build(request)
    assert [agent.role for agent in crew.agents] == [
        "AgentDesigner", "SkillResolver", "PolicyReviewer",
    ]
    assert crew.tasks[-1].output_pydantic is AgentBlueprint
    assert all(not agent.tools for agent in crew.agents)
    # Every design/review role must receive the same backend memory policy.
    assert all(
        'Allowed persistent-agent memory_scopes: ["USER_GLOBAL", "AGENT_PRIVATE"]' in task.description
        for task in crew.tasks
    )


def test_backend_rejects_unavailable_skills_and_permissions() -> None:
    async def scenario() -> None:
        service = AgentService.__new__(AgentService)
        service._session = object()
        service._ensure_primary_agent = AsyncMock(return_value=SimpleNamespace(
            id=uuid4(), status=AgentStatus.ACTIVE, can_spawn_subagents=True,
        ))
        available = SimpleNamespace(list_available_skills=AsyncMock(return_value=[]))
        permission = SimpleNamespace(check=AsyncMock(return_value=False))
        with (
            patch("services.agent.SkillService", return_value=available),
            patch("services.agent.PermissionService", return_value=permission),
        ):
            with pytest.raises(ValueError, match="Skill ID"):
                await service.validate_blueprint(user_id=1, blueprint=blueprint(skill_ids=[uuid4()]))
            with pytest.raises(PermissionError, match="parent permissions"):
                await service.validate_blueprint(user_id=1, blueprint=blueprint(
                    requested_permissions=[RequestedPermission(
                        action_class=ActionClass.WRITE, resource_scope="mail:*",
                    )],
                ))
            required_skill_id = uuid4()
            available.list_available_skills.return_value = [SimpleNamespace(
                id=required_skill_id, required_permissions=["write:docs:*"],
            )]
            with pytest.raises(PermissionError, match="Selected skill requires"):
                await service.validate_blueprint(user_id=1, blueprint=blueprint(
                    skill_ids=[required_skill_id],
                ))

    asyncio.run(scenario())


def test_backend_rejects_invalid_connections_and_subagent_escalation() -> None:
    async def scenario() -> None:
        service = AgentService.__new__(AgentService)
        service._session = object()
        service._ensure_primary_agent = AsyncMock(return_value=SimpleNamespace(
            id=uuid4(), status=AgentStatus.ACTIVE, can_spawn_subagents=False,
        ))
        service.get_agent = AsyncMock(side_effect=LookupError("Agent not found"))
        available = SimpleNamespace(list_available_skills=AsyncMock(return_value=[]))
        with patch("services.agent.SkillService", return_value=available):
            with pytest.raises(PermissionError, match="Parent agent"):
                await service.validate_blueprint(user_id=1, blueprint=blueprint(can_spawn_subagents=True))
            with pytest.raises(LookupError, match="Agent not found"):
                await service.validate_blueprint(user_id=1, blueprint=blueprint(
                    connected_agent_ids=[uuid4()],
                ))

    asyncio.run(scenario())


def test_backend_rejects_unsupported_memory_and_custom_budget() -> None:
    async def scenario() -> None:
        service = AgentService.__new__(AgentService)
        service._session = object()
        service._ensure_primary_agent = AsyncMock(return_value=SimpleNamespace(
            id=uuid4(), status=AgentStatus.ACTIVE, can_spawn_subagents=True,
        ))
        with pytest.raises(ValueError, match="memory scope"):
            await service.validate_blueprint(user_id=1, blueprint=blueprint(
                memory_scopes=[MemoryScope.CREW_SHARED],
            ))
        with pytest.raises(ValueError, match="Custom agent budgets"):
            await service.validate_blueprint(user_id=1, blueprint=blueprint(
                budget_profile=RuntimeBudgets(max_llm_calls=1),
            ))

    asyncio.run(scenario())


def test_natural_language_agent_request_routing() -> None:
    assert is_agent_design_request("Создай мне отдельного Python-агента для разработки")
    assert is_agent_design_request("Create an agent for research")
    assert is_agent_design_request("Мне нужен отдельный помощник для почты")
    assert not is_agent_design_request("Создай команду агентов для анализа")
    assert not is_agent_design_request("Explain Python agents")


def test_cancel_design_discards_blueprint_without_persistence() -> None:
    async def scenario() -> None:
        callback = SimpleNamespace(
            data="agent:design:cancel:abc12345",
            answer=AsyncMock(),
            message=SimpleNamespace(answer=AsyncMock()),
        )
        state = SimpleNamespace(
            get_data=AsyncMock(return_value={
                "wizard_id": "abc12345", "blueprint": blueprint().model_dump(mode="json"),
            }),
            clear=AsyncMock(),
        )
        with patch.object(agent_handlers, "AgentDesignService") as service:
            await agent_handlers.decide_agent_design(callback, state, object())
        state.clear.assert_awaited_once()
        service.assert_not_called()
        assert "cancelled" in callback.message.answer.await_args.args[0]

    asyncio.run(scenario())


class FakeLLM(BaseLLM):
    def call(self, messages, **kwargs):
        return "unused"


def test_runtime_factories_build_real_crewai_objects_without_llm_call() -> None:
    async def scenario() -> None:
        _register_entities()
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        try:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            async with session_factory() as session:
                session.add(UserEntity(id=1, telegram_id=123))
                await session.commit()
                service = AgentService(session)
                agent = await service.create_from_blueprint(user_id=1, blueprint=blueprint())
                prompt = await service.get_current_prompt(user_id=1, agent_id=agent.id)
                definition = AgentRuntimeService._persistent_definition(agent, prompt)
                factory = DynamicCrewAIFactory(llm=FakeLLM(model="fake"))
                runtime_agent = factory.build_agent(definition, budgets=RuntimeBudgets())
                assert isinstance(runtime_agent, CrewAIAgent)
                assert runtime_agent.role == "Developer"
                crew = factory.build_from_definition(CrewDefinition(
                    name="Python research",
                    agents=[definition],
                    tasks=[RuntimeCrewTask(
                        description="Research Python docs",
                        expected_output="Summary",
                        agent_id=agent.id,
                    )],
                ))
                assert isinstance(crew, CrewAICrew)
                assert len(crew.agents) == len(crew.tasks) == 1
        finally:
            await engine.dispose()

    asyncio.run(scenario())


def test_design_preview_and_confirmation_persist_only_after_acceptance() -> None:
    async def scenario() -> None:
        _register_entities()
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        try:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            async with session_factory() as session:
                session.add(UserEntity(id=1, telegram_id=123))
                await session.commit()
                agents = AgentService(session)
                primary = await agents.ensure_primary_agent(user_id=1)
                skill = await SkillService(session).create_user_skill(
                    user_id=1, key="python-docs", name="Python Docs", description="Read documentation",
                )
                await PermissionService(session).grant(
                    user_id=1, subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                    subject_id=str(primary.id), action_class=ActionClass.READ,
                    resource_scope="docs:*",
                )
                proposal = blueprint(
                    skill_ids=[skill.id],
                    missing_capabilities=["repository access"],
                    requested_permissions=[RequestedPermission(
                        action_class=ActionClass.READ, resource_scope="docs:python",
                    )],
                    can_spawn_subagents=True,
                )
                fake_runner = SimpleNamespace(design_agent=AsyncMock(return_value=proposal))
                design = AgentDesignService(session, runner=fake_runner)
                preview = await design.design_agent(user_id=1, user_request="Создай Python агента")
                assert "repository access" in await design.preview_text(user_id=1, blueprint=preview)
                assert len(await agents.list_active_agents(1)) == 1
                result = await design.create_agent(user_id=1, blueprint=preview)
                assert result.name == "Python Developer"
                assigned = await SkillService(session).list_agent_skills(user_id=1, agent_id=result.id)
                assert [item.id for item in assigned] == [skill.id]
                assert await PermissionService(session).check(
                    user_id=1, subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                    subject_id=str(result.id), action_class=ActionClass.READ,
                    resource="docs:python",
                )
                assert result.can_spawn_subagents
        finally:
            await engine.dispose()

    asyncio.run(scenario())
