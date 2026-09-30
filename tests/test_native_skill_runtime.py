import asyncio
import json
import sys
import threading
from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from crewai import BaseLLM
from crewai.skills.models import METADATA, Skill as NativeSkill
from crewai.skills.tool import LoadSkillTool
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agents.assistant.crewai.factory import DynamicCrewAIFactory
from agents.assistant.crewai.runtime import DynamicCrewAIRuntime
from database.base import Base
from database.bootstrap import _register_entities
from database.entities.skill import SkillEntity
from database.entities.user import UserEntity
from models.agent_factory import CrewDefinition
from models.crew import RuntimeCrewTask
from models.permission import ActionClass, PermissionSubjectType
from models.runtime import (
    AgentRuntimeRequest, DelegationDecision, DelegationType, RuntimeBudgets, RuntimeStep,
)
from models.skill import SkillPackage, SkillPackageFile, SkillStatus
from models.tool import ToolExecutionStatus, ToolRequest
from models.tool import ToolDefinition
from services.tool_executor import RegisteredTool, ToolRegistry
from services.skill_execution import SkillExecutionService
from integrations.sandbox.fake import FakeSkillSandbox
from models.skill import SkillExecutionResult, SkillExecutionStatus
from services.approval import ApprovalService
from core.config import config
from integrations.llm.factory import create_crewai_llm


def test_no_auth_custom_llm_does_not_inherit_an_unrelated_openai_key(monkeypatch) -> None:
    monkeypatch.setattr(config.llm, "base_url", "http://localhost:8000/v1")
    monkeypatch.setattr(config.llm, "api_key", "")
    monkeypatch.setattr(config.llm, "model", "openai/local-model")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-test-key")
    llm = create_crewai_llm()
    assert llm.api_key == "not-required"
    assert llm.base_url == "http://localhost:8000/v1"
from services.agent import AgentService
from services.agent_run import AgentRunService
from services.agent_runtime import AgentRuntimeService
from services.assistant import AssistantService
from services.schedule import ScheduleService
from services.permission import PermissionService
from services.skill import SkillService
from services.skill_package import SkillPackageValidationError
from agents.assistant.crewai.skill_runtime import SkillRuntimeResolver
from services.tool_authority import BackendToolAuthority, FakeToolExecutor
from agents.assistant.crewai.tool_runtime import ToolCallBudget, ToolRuntimeResolver


def test_production_composition_never_silently_selects_fake_executor() -> None:
    async def scenario(session):
        assistant = AssistantService(session)
        schedule = ScheduleService(session)
        authority = BackendToolAuthority(session)
        assert assistant._tool_executor is not None
        assert schedule._tool_executor is not None
        assert not isinstance(assistant._tool_executor, FakeToolExecutor)
        assert not isinstance(schedule._tool_executor, FakeToolExecutor)
        assert not isinstance(authority._executor, FakeToolExecutor)
        assert authority._executor.definition("read_information") is None
        with pytest.raises(RuntimeError, match="ToolExecutor is not configured"):
            ToolRuntimeResolver(session)

    asyncio.run(with_session(scenario))


def test_archived_executable_skill_continues_after_approval_but_disabled_is_denied() -> None:
    async def scenario(session):
        storage = Storage()
        agent, _, skills = await create_agent_with_skill(session, storage)
        executable = package("executable")
        executable.manifest.update(runtime="python", entrypoint="src/untrusted_module.py")
        skill = await skills.upload_user_skill(
            user_id=1, key="executable", name="Executable", description="Executable",
            package=executable,
        )
        await skills.assign_to_agent(user_id=1, agent_id=agent.id, skill_id=skill.id)
        run = await AgentRunService(session).create_run(
            user_id=1, starting_agent_id=agent.id, model_name="fake",
        )
        await AgentRunService(session).start(user_id=1, run_id=run.id)
        coordinator = AgentRuntimeService(session, runtime=SimpleNamespace(run=None))
        coordinator._skills = skills
        context = await coordinator._build_context(AgentRuntimeRequest(
            run_id=run.id, user_id=1, agent_id=agent.id, message="Execute",
        ), run.prompt_version_id)
        await coordinator._persist(context)
        await PermissionService(session).grant(
            user_id=1, subject_type=PermissionSubjectType.PERSISTENT_AGENT,
            subject_id=str(agent.id), action_class=ActionClass.EXECUTE,
            resource_scope="skill:*",
        )
        sandbox = FakeSkillSandbox(result=SkillExecutionResult(
            status=SkillExecutionStatus.SUCCEEDED, exit_code=0, output={"done": True},
        ))
        authority = BackendToolAuthority(session, skill_execution_service=SkillExecutionService(
            session, skill_service=skills, sandbox=sandbox,
        ))
        request = ToolRequest(
            run_id=run.id, user_id=1,
            requesting_subject_type=PermissionSubjectType.PERSISTENT_AGENT,
            requesting_subject_id=str(agent.id), name="execute_skill",
            action_class=ActionClass.EXECUTE, resource=f"skill:{skill.id}:v1",
            arguments={"skill_key": skill.key, "arguments": {}},
        )
        pending = await authority.request(request, runtime_skill_catalog=context.active_skills)
        assert pending.status == ToolExecutionStatus.WAITING_APPROVAL
        await skills.archive_skill(user_id=1, skill_id=skill.id)
        await ApprovalService(session).approve(user_id=1, approval_id=pending.approval_id)
        result = await authority.execute_approved(
            user_id=1, approval_id=pending.approval_id,
            runtime_skill_catalog=context.active_skills,
        )
        assert result.status == ToolExecutionStatus.EXECUTED
        await skills.disable_skill(user_id=1, skill_id=skill.id)
        denied = await authority.request(request, runtime_skill_catalog=context.active_skills)
        assert denied.status == ToolExecutionStatus.DENIED

    asyncio.run(with_session(scenario))


def test_archived_skill_only_remains_available_to_its_persisted_run_pin() -> None:
    async def scenario(session):
        storage = Storage()
        agent, skill, skills = await create_agent_with_skill(
            session, storage, tools=[
                {"id": "read_information", "resource": "information:companies"},
            ],
        )
        run = await AgentRunService(session).create_run(
            user_id=1, starting_agent_id=agent.id, model_name="fake",
        )
        await AgentRunService(session).start(user_id=1, run_id=run.id)
        request = AgentRuntimeRequest(
            run_id=run.id, user_id=1, agent_id=agent.id, message="Research",
        )
        coordinator = AgentRuntimeService(session, runtime=SimpleNamespace(run=None))
        coordinator._skills = skills
        context = await coordinator._build_context(request, run.prompt_version_id)
        await coordinator._persist(context)
        pinned = context.starting_agent.active_skills[0]
        await skills.archive_skill(user_id=1, skill_id=skill.id)
        assert await skills.resolve_agent_skills(user_id=1, agent_id=agent.id) == []
        with pytest.raises(ValueError, match="inactive"):
            await skills.assign_to_agent(user_id=1, agent_id=agent.id, skill_id=skill.id)
        resolver = SkillRuntimeResolver(skills)
        async with resolver.materialize_for_agent(
            user_id=1, agent_id=agent.id, definitions=[pinned], run_id=run.id,
        ) as native:
            assert len(native) == 1
        with pytest.raises(SkillPackageValidationError):
            async with resolver.materialize_for_agent(
                user_id=1, agent_id=agent.id, definitions=[pinned], run_id=uuid4(),
            ):
                pass
        await PermissionService(session).grant(
            user_id=1, subject_type=PermissionSubjectType.PERSISTENT_AGENT,
            subject_id=str(agent.id), action_class=ActionClass.READ,
            resource_scope="information:*",
        )
        tools = await ToolRuntimeResolver(session, executor=FakeToolExecutor()).resolve(
            run_id=run.id, user_id=1,
            subject_type=PermissionSubjectType.PERSISTENT_AGENT,
            subject_id=str(agent.id), skills=[pinned],
        )
        assert len(tools) == 1
        await skills.disable_skill(user_id=1, skill_id=skill.id)
        with pytest.raises(SkillPackageValidationError):
            async with resolver.materialize_for_agent(
                user_id=1, agent_id=agent.id, definitions=[pinned], run_id=run.id,
            ):
                pass
        assert json.loads(await tools[0]._arun({"query": "revoked"}))["status"] == "denied"

    asyncio.run(with_session(scenario))


class FakeLLM(BaseLLM):
    def call(self, messages, **kwargs):
        return "unused"


class Storage:
    def __init__(self):
        self.objects: dict[str, dict[str, bytes]] = {}
        self.checksums: dict[str, str] = {}

    async def upload_package(self, *, prefix: str, files: Sequence[SkillPackageFile], checksum: str):
        self.objects[prefix] = {item.path: item.content for item in files}
        self.checksums[prefix] = checksum

    async def get_file(self, *, prefix: str, path: str, max_bytes: int) -> bytes:
        data = self.objects[prefix][path]
        assert len(data) <= max_bytes
        return data

    async def list_files(self, *, prefix: str) -> list[str]:
        return sorted(self.objects[prefix])

    async def get_package_checksum(self, *, prefix: str) -> str | None:
        return self.checksums.get(prefix)

    async def exists(self, *, prefix: str) -> bool:
        return prefix in self.objects

    async def delete_version(self, *, prefix: str) -> None:
        self.objects.pop(prefix, None)
        self.checksums.pop(prefix, None)


def package(
    key: str, *, version: int = 1, tools=None,
    body: str = "Follow the documented workflow.", resources: bool = False,
):
    files = [SkillPackageFile(
        path="src/untrusted_module.py",
        content=b"raise RuntimeError('must never import')\n",
    )]
    if resources:
        files.append(SkillPackageFile(path="resources/guide.md", content=b"Trusted guide bytes"))
    return SkillPackage(
        skill_md=body,
        manifest={
            "key": key,
            "version": version,
            "entrypoint": None,
            "description": f"Guidance for {key}",
            "files": [item.path for item in files],
            "tools": tools or [],
        },
        files=files,
    )


async def with_session(test):
    _register_entities()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            session.add_all([UserEntity(id=1, telegram_id=101), UserEntity(id=2, telegram_id=202)])
            await session.commit()
            await test(session)
    finally:
        await engine.dispose()


async def create_agent_with_skill(session, storage, *, user_id=1, key="python-helper", version=1,
                                  system=False, tools=None, resources=False):
    agents = AgentService(session)
    agent = await agents.create_user_agent(
        user_id=user_id, name=f"Agent {key}", role="Researcher", goal="Research safely",
    )
    skills = SkillService(session, storage=storage)
    if system:
        skill = await skills.create_system_skill(
            key=key, name=key, description=f"Guidance for {key}",
            version=version, package=package(key, version=version, tools=tools, resources=resources),
        )
    else:
        skill = await skills.upload_user_skill(
            user_id=user_id, key=key, name=key, description=f"Guidance for {key}",
            version=version, package=package(key, version=version, tools=tools, resources=resources),
        )
    await skills.assign_to_agent(user_id=user_id, agent_id=agent.id, skill_id=skill.id)
    return agent, skill, skills


def test_native_loader_receives_verified_skill_and_cleans_runtime_directory() -> None:
    async def scenario(session):
        storage = Storage()
        agent, skill, skills = await create_agent_with_skill(session, storage, resources=True)
        definitions = await skills.resolve_agent_skills(user_id=1, agent_id=agent.id)
        resolver = SkillRuntimeResolver(skills)
        async with resolver.materialize_for_agent(
            user_id=1, agent_id=agent.id, definitions=definitions, run_id=uuid4(),
        ) as native:
            assert len(native) == 1 and isinstance(native[0], NativeSkill)
            assert native[0].disclosure_level == METADATA
            path = native[0].path
            assert (path / "SKILL.md").is_file()
            assert (path / "src" / "untrusted_module.py").is_file()
            assert (path / "references" / "guide.md").read_bytes() == b"Trusted guide bytes"
            assert "untrusted_module" not in sys.modules
            definition = AgentRuntimeService._persistent_definition(
                agent, await AgentService(session).get_current_prompt(user_id=1, agent_id=agent.id),
                active_skills=definitions,
            )
            crewai_agent = DynamicCrewAIFactory(llm=FakeLLM(model="fake")).build_agent(
                definition, budgets=RuntimeBudgets(), native_skills=native,
            )
            assert crewai_agent.skills == native
            crewai_agent.set_skills()
            assert len(crewai_agent.skills) == 1
            assert isinstance(crewai_agent._add_skill_loader_tool([])[0], LoadSkillTool)
        assert not path.exists()

    asyncio.run(with_session(scenario))


def test_materialization_rejects_colliding_resource_alias(tmp_path) -> None:
    from models.runtime import RuntimeSkillDefinition
    from models.skill import PreparedSkillPackage

    prepared = PreparedSkillPackage(
        manifest={}, checksum="0" * 64,
        files=(
            SkillPackageFile(path="SKILL.md", content=b"Instructions"),
            SkillPackageFile(path="resources/guide.md", content=b"Resource"),
            SkillPackageFile(path="references/guide.md", content=b"Reference"),
        ),
    )
    root = tmp_path / "skill"
    with pytest.raises(SkillPackageValidationError, match="alias collides"):
        SkillRuntimeResolver._write_package(
            skill_dir=root, prepared=prepared,
            definition=RuntimeSkillDefinition(
                id=uuid4(), key="reference", name="Reference", description="Guide", version=1,
            ),
        )
    assert not root.exists()


def test_multiple_system_and_empty_skill_resolution() -> None:
    async def scenario(session):
        storage = Storage()
        agent, first, skills = await create_agent_with_skill(session, storage, key="first")
        second = await skills.create_system_skill(
            key="system-help", name="system-help", description="System guidance",
            package=package("system-help"),
        )
        await skills.assign_to_agent(user_id=1, agent_id=agent.id, skill_id=second.id)
        definitions = await skills.resolve_agent_skills(user_id=1, agent_id=agent.id)
        async with SkillRuntimeResolver(skills).materialize_for_agent(
            user_id=1, agent_id=agent.id, definitions=definitions,
        ) as native:
            assert len(native) == 2
            assert len({item.name for item in native}) == 2
        async with SkillRuntimeResolver(skills).materialize_for_agent(
            user_id=1, agent_id=agent.id, definitions=[],
        ) as native:
            assert native == []

    asyncio.run(with_session(scenario))


def test_persistent_crew_runtime_keeps_distinct_native_agent_skills_alive() -> None:
    async def scenario(session):
        storage = Storage()
        first, _, skills = await create_agent_with_skill(session, storage, key="research")
        second, _, _ = await create_agent_with_skill(session, storage, key="analysis")
        first_skills = await skills.resolve_agent_skills(user_id=1, agent_id=first.id)
        second_skills = await skills.resolve_agent_skills(user_id=1, agent_id=second.id)
        agent_service = AgentService(session)
        definitions = [
            AgentRuntimeService._persistent_definition(
                member,
                await agent_service.get_current_prompt(user_id=1, agent_id=member.id),
                active_skills=assigned,
            ) for member, assigned in ((first, first_skills), (second, second_skills))
        ]
        definition = CrewDefinition(
            user_id=1, name="Research Crew", agents=definitions,
            tasks=[RuntimeCrewTask(
                agent_id=first.id, description="Research", expected_output="Notes",
            ), RuntimeCrewTask(
                agent_id=second.id, description="Analyze", expected_output="Report",
            )],
        )
        runtime = DynamicCrewAIRuntime(
            factory=DynamicCrewAIFactory(llm=FakeLLM(model="fake")),
            skill_resolver=SkillRuntimeResolver(skills),
        )
        async with runtime.prepared_crew(definition, user_id=1) as crew:
            assert len(crew.agents) == 2
            assert all(len(member.skills) == 1 for member in crew.agents)
            assert crew.agents[0].skills[0].name != crew.agents[1].skills[0].name
            paths = [member.skills[0].path for member in crew.agents]
            assert all(path.is_dir() for path in paths)
        assert all(not path.exists() for path in paths)

    asyncio.run(with_session(scenario))


def test_async_runtime_keeps_native_skill_files_until_kickoff_completes() -> None:
    async def scenario(session):
        storage = Storage()
        agent, _, skills = await create_agent_with_skill(session, storage)
        run = await AgentRunService(session).create_run(
            user_id=1, starting_agent_id=agent.id, model_name="fake",
        )
        request = AgentRuntimeRequest(
            run_id=run.id, user_id=1, agent_id=agent.id, message="Research",
        )
        coordinator = AgentRuntimeService(session, runtime=SimpleNamespace(run=None))
        coordinator._skills = skills
        context = await coordinator._build_context(request, run.prompt_version_id)
        seen_paths = []
        output = SimpleNamespace(
            pydantic=RuntimeStep(
                decision=DelegationDecision(type=DelegationType.RESPOND), content="Done",
            ), json_dict=None, raw="", token_usage=None,
        )

        class Factory:
            def build(self, _context, _request, *, native_skills, resolved_tools):
                assert len(native_skills) == 1
                assert native_skills[0].path.is_dir()
                seen_paths.append(native_skills[0].path)
                return SimpleNamespace(akickoff=AsyncMock(return_value=output))

        runtime = DynamicCrewAIRuntime(
            factory=Factory(), skill_resolver=SkillRuntimeResolver(skills),
        )
        result = await runtime.run(context, request)
        assert result.content == "Done"
        assert seen_paths and not seen_paths[0].exists()

    asyncio.run(with_session(scenario))


def test_cross_tenant_archive_checksum_path_and_missing_skill_fail_closed() -> None:
    async def scenario(session):
        storage = Storage()
        agent, skill, skills = await create_agent_with_skill(session, storage)
        definition = (await skills.resolve_agent_skills(user_id=1, agent_id=agent.id))[0]
        resolver = SkillRuntimeResolver(skills)
        with pytest.raises(LookupError):
            async with resolver.materialize_for_agent(
                user_id=2, agent_id=None, definitions=[definition],
            ):
                pass
        prefix = next(iter(storage.objects))
        original = storage.checksums[prefix]
        storage.checksums[prefix] = "0" * 64
        with pytest.raises(SkillPackageValidationError, match="checksum"):
            async with resolver.materialize_for_agent(
                user_id=1, agent_id=agent.id, definitions=[definition],
            ):
                pass
        storage.checksums[prefix] = original
        storage.objects[prefix]["../secret"] = b"no"
        with pytest.raises(SkillPackageValidationError, match="path|paths"):
            async with resolver.materialize_for_agent(
                user_id=1, agent_id=agent.id, definitions=[definition],
            ):
                pass
        del storage.objects[prefix]["../secret"]
        removed = storage.objects[prefix].pop("SKILL.md")
        with pytest.raises(SkillPackageValidationError, match="SKILL.md"):
            async with resolver.materialize_for_agent(
                user_id=1, agent_id=agent.id, definitions=[definition],
            ):
                pass
        storage.objects[prefix]["SKILL.md"] = removed
        entity = await session.get(SkillEntity, skill.id)
        assert entity is not None
        entity.storage_uri = "s3://forged/other-user/"
        await session.commit()
        with pytest.raises(SkillPackageValidationError, match="metadata changed"):
            async with resolver.materialize_for_agent(
                user_id=1, agent_id=agent.id, definitions=[definition],
            ):
                pass
        entity.storage_uri = definition.storage_uri
        entity.status = SkillStatus.ARCHIVED
        await session.commit()
        with pytest.raises(SkillPackageValidationError, match="no longer active"):
            async with resolver.materialize_for_agent(
                user_id=1, agent_id=agent.id, definitions=[definition],
            ):
                pass

    asyncio.run(with_session(scenario))


def test_pinned_skill_version_survives_new_assignment_in_same_run() -> None:
    async def scenario(session):
        storage = Storage()
        agent, first, skills = await create_agent_with_skill(session, storage, key="versioned", version=1)
        run = await AgentRunService(session).create_run(
            user_id=1, starting_agent_id=agent.id, model_name="fake",
        )
        await AgentRunService(session).start(user_id=1, run_id=run.id)
        runtime = AgentRuntimeService(session, runtime=SimpleNamespace(run=None))
        runtime._skills = skills
        request = AgentRuntimeRequest(
            run_id=run.id, user_id=1, agent_id=agent.id, message="research",
        )
        context = await runtime._build_context(request, run.prompt_version_id)
        await runtime._persist(context)
        pinned = context.starting_agent.active_skills[0]
        second = await skills.upload_user_skill(
            user_id=1, key="versioned", name="versioned", description="v2 guidance",
            version=2, package=package("versioned", version=2),
        )
        await skills.remove_from_agent(user_id=1, agent_id=agent.id, skill_id=first.id)
        await skills.assign_to_agent(user_id=1, agent_id=agent.id, skill_id=second.id)
        async with SkillRuntimeResolver(skills).materialize_for_agent(
            user_id=1, agent_id=agent.id, definitions=[pinned], run_id=run.id,
        ) as native:
            assert len(native) == 1
            assert native[0].path.is_dir()
        assert pinned.version == 1

    asyncio.run(with_session(scenario))


def test_declared_read_tool_is_native_and_write_remains_approval_gated() -> None:
    async def scenario(session):
        storage = Storage()
        declarations = [
            {"id": "read_information", "resource": "information:companies"},
            {"id": "write_note", "resource": "notes:research"},
        ]
        agent, skill, skills = await create_agent_with_skill(
            session, storage, key="research-tools", tools=declarations,
        )
        run = await AgentRunService(session).create_run(
            user_id=1, starting_agent_id=agent.id, model_name="fake",
        )
        await AgentRunService(session).start(user_id=1, run_id=run.id)
        definitions = await skills.resolve_agent_skills(user_id=1, agent_id=agent.id)
        resolver = ToolRuntimeResolver(session, executor=FakeToolExecutor())
        assert await resolver.resolve(
            run_id=run.id, user_id=1,
            subject_type=PermissionSubjectType.PERSISTENT_AGENT,
            subject_id=str(agent.id), skills=definitions,
        ) == []
        permissions = PermissionService(session)
        await permissions.grant(
            user_id=1, subject_type=PermissionSubjectType.PERSISTENT_AGENT,
            subject_id=str(agent.id), action_class=ActionClass.READ,
            resource_scope="information:*",
        )
        await permissions.grant(
            user_id=1, subject_type=PermissionSubjectType.PERSISTENT_AGENT,
            subject_id=str(agent.id), action_class=ActionClass.WRITE,
            resource_scope="notes:*",
        )
        tools = await resolver.resolve(
            run_id=run.id, user_id=1,
            subject_type=PermissionSubjectType.PERSISTENT_AGENT,
            subject_id=str(agent.id), skills=definitions,
        )
        assert [tool.name for tool in tools] == ["read_information"]
        runtime_definition = AgentRuntimeService._persistent_definition(
            agent, await AgentService(session).get_current_prompt(user_id=1, agent_id=agent.id),
            active_skills=definitions,
        )
        native_agent = DynamicCrewAIFactory(llm=FakeLLM(model="fake")).build_agent(
            runtime_definition, budgets=RuntimeBudgets(), resolved_tools=tools,
        )
        assert [tool.name for tool in native_agent.tools] == ["read_information"]
        response = await tools[0].to_structured_tool().ainvoke(
            input={"arguments": {"query": "acme"}}
        )
        assert '"status": "executed"' in response
        budget = ToolCallBudget(1)
        bounded = await resolver.resolve(
            run_id=run.id, user_id=1,
            subject_type=PermissionSubjectType.PERSISTENT_AGENT,
            subject_id=str(agent.id), skills=definitions, budget=budget,
        )
        await bounded[0].to_structured_tool().ainvoke(input={"arguments": {"query": "one"}})
        with pytest.raises(PermissionError, match="budget"):
            await bounded[0].to_structured_tool().ainvoke(input={"arguments": {"query": "two"}})
        assert budget.used == 1
        assert not tools[0].cache_function({}, "cached output")
        await permissions.revoke(
            user_id=1, subject_type=PermissionSubjectType.PERSISTENT_AGENT,
            subject_id=str(agent.id), action_class=ActionClass.READ,
            resource_scope="information:*",
        )
        response = await tools[0].to_structured_tool().ainvoke(
            input={"arguments": {"query": "acme"}}
        )
        assert json.loads(response)["status"] == "denied"
        assert json.loads(response)["output"] is None
        inherited = await resolver.resolve(
            run_id=run.id, user_id=1,
            subject_type=PermissionSubjectType.TEMPORARY_SUBAGENT,
            subject_id=str(uuid4()), skills=definitions,
            runtime_permission_scopes=["read:information:*"],
        )
        assert [tool.name for tool in inherited] == ["read_information"]
        response = await inherited[0].to_structured_tool().ainvoke(
            input={"arguments": {"query": "inherited"}}
        )
        assert json.loads(response)["status"] == "executed"
        assert await resolver.resolve(
            run_id=run.id, user_id=1,
            subject_type=PermissionSubjectType.TEMPORARY_SUBAGENT,
            subject_id=str(uuid4()), skills=definitions,
            runtime_permission_scopes=["read:information:other"],
        ) == []
        authority = BackendToolAuthority(session, executor=FakeToolExecutor())
        result = await authority.request(ToolRequest(
            run_id=run.id, user_id=1,
            requesting_subject_type=PermissionSubjectType.PERSISTENT_AGENT,
            requesting_subject_id=str(agent.id), name="write_note",
            action_class=ActionClass.WRITE, resource="notes:research",
            arguments={"text": "draft"},
        ))
        assert result.status == ToolExecutionStatus.WAITING_APPROVAL
        forged = definitions[0].model_copy(deep=True)
        forged.declared_tools[0].id = "unknown"
        with pytest.raises(ValueError, match="Declared tools differ|metadata changed"):
            await resolver.resolve(
                run_id=run.id, user_id=1,
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id=str(agent.id), skills=[forged],
            )
        unknown = await skills.upload_user_skill(
            user_id=1, key="unknown-tool", name="Unknown Tool", description="Unknown",
            package=package("unknown-tool", tools=[
                {"id": "does_not_exist", "resource": "information:companies"},
            ]),
        )
        await skills.assign_to_agent(user_id=1, agent_id=agent.id, skill_id=unknown.id)
        unknown_definition = next(
            item for item in await skills.resolve_agent_skills(user_id=1, agent_id=agent.id)
            if item.id == unknown.id
        )
        with pytest.raises(ValueError, match="Unknown"):
            await resolver.resolve(
                run_id=run.id, user_id=1,
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id=str(agent.id), skills=[unknown_definition],
            )

    asyncio.run(with_session(scenario))


@pytest.mark.parametrize("native_function_calling", [False, True])
def test_real_native_async_kickoff_loads_skill_and_calls_backend_in_owner_loop(native_function_calling) -> None:
    async def scenario(session):
        owner_loop = asyncio.get_running_loop()
        storage = Storage()
        agent, _, skills = await create_agent_with_skill(
            session, storage, tools=[
                {"id": "read_information", "resource": "information:companies"},
            ],
        )
        run = await AgentRunService(session).create_run(
            user_id=1, starting_agent_id=agent.id, model_name="fake",
        )
        await AgentRunService(session).start(user_id=1, run_id=run.id)
        await PermissionService(session).grant(
            user_id=1, subject_type=PermissionSubjectType.PERSISTENT_AGENT,
            subject_id=str(agent.id), action_class=ActionClass.READ,
            resource_scope="information:*",
        )
        request = AgentRuntimeRequest(
            run_id=run.id, user_id=1, agent_id=agent.id, message="Research",
        )
        coordinator = AgentRuntimeService(session, runtime=SimpleNamespace(run=None))
        coordinator._skills = skills
        context = await coordinator._build_context(request, run.prompt_version_id)
        paths = []
        calls = []

        async def read_information(arguments):
            assert asyncio.get_running_loop() is owner_loop
            assert paths[0].is_dir()
            # Execute a real query on the backend-owned async session.
            user = await session.get(UserEntity, 1)
            assert user is not None
            calls.append("read_information")
            return {"query": arguments["query"], "user_exists": True}

        class ScriptedLLM(FakeLLM):
            def __init__(self):
                super().__init__(model="fake")
                self.turn = 0

            def call(self, messages, **kwargs):
                # AgentExecutor invokes inference in a worker thread; all
                # backend AsyncSession work still belongs to owner_loop.
                return self._reply(messages, **kwargs)

            def supports_function_calling(self):
                return native_function_calling

            async def acall(self, messages, **kwargs):
                return self._reply(messages, **kwargs)

            def _reply(self, messages, **kwargs):
                member = kwargs["from_agent"]
                if self.turn == 0:
                    paths.append(member.skills[0].path)
                    reply = (
                        "Thought: Load the relevant instructions.\nAction: load_skill\n"
                        "Action Input: " + json.dumps({"skill_name": member.skills[0].name})
                    )
                elif self.turn == 1:
                    assert "Follow the documented workflow." in str(messages)
                    reply = (
                        'Thought: Read company information.\nAction: read_information\n'
                        'Action Input: {"arguments": {"query": "acme"}}'
                    )
                else:
                    assert '"status": "executed"' in str(messages)
                    reply = (
                        'Thought: Research complete.\nFinal Answer: '
                        '{"decision": {"type": "respond"}, "content": "Done"}'
                    )
                if native_function_calling:
                    if self.turn < 2:
                        name = "load_skill" if self.turn == 0 else "read_information"
                        arguments = ({"skill_name": member.skills[0].name} if self.turn == 0
                                     else {"arguments": {"query": "acme"}})
                        reply = [{"id": f"call-{self.turn}", "type": "function", "function": {
                            "name": name, "arguments": json.dumps(arguments),
                        }}]
                    else:
                        reply = '{"decision": {"type": "respond"}, "content": "Done"}'
                self.turn += 1
                return reply

        runtime = DynamicCrewAIRuntime(
            factory=DynamicCrewAIFactory(llm=ScriptedLLM()),
            skill_resolver=SkillRuntimeResolver(skills),
            tool_resolver=ToolRuntimeResolver(session, executor=ToolRegistry([
                RegisteredTool(ToolDefinition(
                    name="read_information", action_class=ActionClass.READ,
                    resource_prefix="information:",
                ), read_information),
            ])),
        )
        result = await runtime.run(context, request)
        assert result.content == "Done"
        assert calls == ["read_information"]
        assert context.usage.tool_calls == 1
        assert not paths[0].exists()

    asyncio.run(with_session(scenario))


def test_cancel_drains_crewai_worker_before_removing_materialized_files() -> None:
    async def scenario(session):
        storage = Storage()
        agent, _, skills = await create_agent_with_skill(session, storage)
        run = await AgentRunService(session).create_run(
            user_id=1, starting_agent_id=agent.id, model_name="fake",
        )
        request = AgentRuntimeRequest(
            run_id=run.id, user_id=1, agent_id=agent.id, message="Research",
        )
        coordinator = AgentRuntimeService(session, runtime=SimpleNamespace(run=None))
        coordinator._skills = skills
        context = await coordinator._build_context(request, run.prompt_version_id)
        owner_loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        paths = []
        worker_finished = []

        class Factory:
            def build(self, _context, _request, *, native_skills, resolved_tools):
                paths.append(native_skills[0].path)

                def worker():
                    owner_loop.call_soon_threadsafe(started.set)
                    assert release.wait(timeout=5)
                    assert paths[0].is_dir()
                    worker_finished.append(True)
                    return SimpleNamespace(pydantic=RuntimeStep(
                        decision=DelegationDecision(type=DelegationType.RESPOND), content="Done",
                    ), token_usage=None)

                async def kickoff():
                    return await asyncio.to_thread(worker)

                return SimpleNamespace(akickoff=kickoff)

        runtime = DynamicCrewAIRuntime(factory=Factory(), skill_resolver=SkillRuntimeResolver(skills))
        running = asyncio.create_task(runtime.run(context, request))
        await asyncio.wait_for(started.wait(), timeout=5)
        running.cancel()
        await asyncio.sleep(0)
        assert paths[0].exists()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert worker_finished == [True]
        assert not paths[0].exists()

    asyncio.run(with_session(scenario))


@pytest.mark.parametrize("error", [RuntimeError("kickoff failed"), asyncio.CancelledError()])
def test_kickoff_failure_cleans_skills_and_preserves_tool_usage(error) -> None:
    async def scenario(session):
        storage = Storage()
        agent, _, skills = await create_agent_with_skill(session, storage)
        run = await AgentRunService(session).create_run(
            user_id=1, starting_agent_id=agent.id, model_name="fake",
        )
        request = AgentRuntimeRequest(
            run_id=run.id, user_id=1, agent_id=agent.id, message="Research",
        )
        coordinator = AgentRuntimeService(session, runtime=SimpleNamespace(run=None))
        coordinator._skills = skills
        context = await coordinator._build_context(request, run.prompt_version_id)
        paths = []
        budget = None

        async def resolve(**kwargs):
            nonlocal budget
            budget = kwargs["budget"]
            return []

        class Factory:
            def build(self, _context, _request, *, native_skills, resolved_tools):
                paths.append(native_skills[0].path)

                async def kickoff():
                    assert paths[0].is_dir()
                    budget.consume()
                    raise error

                return SimpleNamespace(akickoff=kickoff)

        runtime = DynamicCrewAIRuntime(
            factory=Factory(), skill_resolver=SkillRuntimeResolver(skills),
            tool_resolver=SimpleNamespace(resolve=resolve),
        )
        with pytest.raises(type(error)):
            await runtime.run(context, request)
        assert context.usage.tool_calls == 1
        assert not paths[0].exists()

    asyncio.run(with_session(scenario))
