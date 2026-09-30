import asyncio
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
import unittest
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import SandboxConfig, config
from database.base import Base
from database.entities.agent import AgentEntity
from database.entities.agent_prompt import AgentPromptVersionEntity
from database.entities.agent_run import AgentRunEntity
from database.entities.agent_run_event import AgentRunEventEntity
from database.entities.agent_skill import AgentSkillEntity
from database.entities.approval import ApprovalEntity
from database.entities.permission import PermissionEntity
from database.entities.skill import SkillEntity
from database.entities.skill_execution import SkillExecutionEntity
from database.entities.user import UserEntity
from integrations.sandbox.docker import DockerSkillSandbox
from integrations.sandbox.fake import FakeSkillSandbox
from integrations.storage.skill_storage import SkillStorageConflictError, SkillStorageError
from models.agent_run import AgentRunEventType
from models.permission import ActionClass, PermissionSubjectType
from models.runtime import RuntimeSkillDefinition
from models.skill import (
    MaterializedSkillPackage,
    SandboxLimits,
    SkillExecutionResult,
    SkillExecutionRequest,
    SkillExecutionStatus,
    SkillPackage,
    SkillPackageFile,
)
from services.agent import AgentService
from services.agent_run import AgentRunService
from services.approval import ApprovalService
from services.permission import PermissionService
from services.skill import SkillService
from services.skill_execution import SkillExecutionService
from services.skill_package import SkillPackageService, SkillPackageValidationError
from services.tool_authority import BackendToolAuthority


class FakeExecutionStorage:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, bytes]] = {}
        self.checksums: dict[str, str] = {}

    async def upload_package(
        self, *, prefix: str, files: Sequence[SkillPackageFile], checksum: str
    ) -> None:
        if prefix in self.checksums and self.checksums[prefix] != checksum:
            raise SkillStorageConflictError("immutable package changed")
        self.objects[prefix] = {item.path: item.content for item in files}
        self.checksums[prefix] = checksum

    async def get_file(self, *, prefix: str, path: str, max_bytes: int) -> bytes:
        try:
            value = self.objects[prefix][path]
        except KeyError as exc:
            raise SkillStorageError("object missing") from exc
        if len(value) > max_bytes:
            raise SkillStorageError("object too large")
        return value

    async def list_files(self, *, prefix: str) -> list[str]:
        return sorted(self.objects.get(prefix, {}))

    async def delete_version(self, *, prefix: str) -> None:
        self.objects.pop(prefix, None)
        self.checksums.pop(prefix, None)

    async def exists(self, *, prefix: str) -> bool:
        return prefix in self.objects

    async def get_package_checksum(self, *, prefix: str) -> str | None:
        return self.checksums.get(prefix)


def executable_package(*, key: str = "python_calculator", version: int = 1) -> SkillPackage:
    return SkillPackage(
        skill_md="Execute the calculator when arithmetic is required.",
        manifest={
            "key": key,
            "version": version,
            "runtime": "python",
            "entrypoint": "src/main.py",
            "files": ["src/main.py"],
        },
        files=[SkillPackageFile(path="src/main.py", content=b"print('safe')\n")],
    )


async def with_session(test: Callable[[AsyncSession], Awaitable[None]]) -> None:
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


async def prepare_skill(
    session: AsyncSession, *, required_permissions: list[str] | None = None
):
    storage = FakeExecutionStorage()
    skills = SkillService(session, storage=storage)
    agent = await AgentService(session).ensure_primary_agent(user_id=1)
    skill = await skills.upload_user_skill(
        user_id=1,
        key="python_calculator",
        name="Python calculator",
        description="Arithmetic helper",
        package=executable_package(),
        required_permissions=required_permissions,
    )
    await skills.assign_to_agent(user_id=1, agent_id=agent.id, skill_id=skill.id)
    run = await AgentRunService(session).create_run(
        user_id=1,
        starting_agent_id=agent.id,
        model_name="test",
    )
    await AgentRunService(session).start(user_id=1, run_id=run.id)
    catalog = await skills.resolve_agent_skills(user_id=1, agent_id=agent.id)
    return storage, skills, agent, skill, run, catalog


class SkillExecutionTests(unittest.TestCase):
    def test_executable_manifest_requires_python_entrypoint_under_src(self) -> None:
        service = SkillPackageService(storage_provider=FakeExecutionStorage, config=config.s3)
        with self.assertRaises(SkillPackageValidationError):
            service.validate(
                package=SkillPackage(
                    skill_md="x",
                    manifest={
                        "key": "bad",
                        "version": 1,
                        "runtime": "python",
                        "entrypoint": "main.py",
                    },
                ),
                key="bad",
                version=1,
            )

    def test_docker_command_has_isolation_and_allowlisted_environment(self) -> None:
        sandbox = DockerSkillSandbox(image="python:3.12.7-slim")
        package = MaterializedSkillPackage(
            skill_id=uuid4(),
            key="calculator",
            version=1,
            root_path="C:/tmp/skill",
            work_path="C:/tmp/work",
            output_path="C:/tmp/output",
            entrypoint="src/main.py",
            checksum="a" * 64,
        )
        command = sandbox.build_command(
            package=package,
            entrypoint=package.entrypoint,
            limits=SandboxLimits(
                timeout_seconds=3,
                memory_mb=64,
                cpus=0.25,
                pids_limit=8,
                max_stdout_bytes=32,
                max_stderr_bytes=32,
            ),
            container_name="skill-test",
            environment={
                "SKILL_RUN_ID": "run",
                "DATABASE_URL": "secret",
            },
        )
        self.assertIn("--network", command)
        self.assertIn("none", command)
        self.assertIn("--read-only", command)
        self.assertIn("--cap-drop", command)
        self.assertIn("ALL", command)
        self.assertIn("--security-opt", command)
        self.assertIn("no-new-privileges", command)
        self.assertIn("--pids-limit", command)
        self.assertIn("--memory", command)
        self.assertIn("--cpus", command)
        self.assertIn("--tmpfs", command)
        self.assertNotIn("DATABASE_URL", command)
        self.assertIn("--user", command)
        self.assertIn("65532:65532", command)

    def test_success_is_materialized_audited_and_idempotent(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            storage, skills, agent, skill, run, catalog = await prepare_skill(session)
            sandbox = FakeSkillSandbox(
                SkillExecutionResult(
                    status=SkillExecutionStatus.SUCCEEDED,
                    stdout='{"ok":true,"result":{"value":4}}',
                    output={"value": 4},
                )
            )
            service = SkillExecutionService(session, skill_service=skills, sandbox=sandbox)
            request = {
                "run_id": run.id,
                "user_id": 1,
                "agent_id": agent.id,
                "skill_id": skill.id,
                "skill_key": skill.key,
                "skill_version": skill.version,
                "requesting_subject_type": PermissionSubjectType.PERSISTENT_AGENT,
                "requesting_subject_id": str(agent.id),
                "arguments": {"expression": "2 + 2"},
            }
            first = await service.execute(
                SkillExecutionRequest(**request), runtime_skill_catalog=catalog
            )
            second = await service.execute(
                SkillExecutionRequest(**request), runtime_skill_catalog=catalog
            )
            self.assertEqual(first.status, SkillExecutionStatus.SUCCEEDED)
            self.assertEqual(first.output, {"value": 4})
            self.assertEqual(second.status, SkillExecutionStatus.SUCCEEDED)
            self.assertEqual(len(sandbox.calls), 1)
            self.assertFalse(Path(sandbox.calls[0]["package"].root_path).exists())
            persisted = (await session.execute(select(SkillExecutionEntity))).scalars().all()
            self.assertEqual(len(persisted), 1)
            self.assertEqual(persisted[0].status, SkillExecutionStatus.SUCCEEDED)
            events = await AgentRunService(session).list_events(user_id=1, run_id=run.id)
            self.assertCountEqual(
                [event.event_type for event in events if "skill_execution" in event.event_type.value],
                [AgentRunEventType.SKILL_EXECUTION_STARTED, AgentRunEventType.SKILL_EXECUTION_COMPLETED],
            )

        asyncio.run(with_session(scenario))

    def test_checksum_mismatch_never_calls_sandbox(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            storage, skills, agent, skill, run, catalog = await prepare_skill(session)
            prefix = "users/1/python_calculator/v1/"
            storage.checksums[prefix] = "b" * 64
            sandbox = FakeSkillSandbox()
            service = SkillExecutionService(session, skill_service=skills, sandbox=sandbox)
            result = await service.execute(
                SkillExecutionRequest(
                    run_id=run.id,
                    user_id=1,
                    agent_id=agent.id,
                    skill_id=skill.id,
                    skill_key=skill.key,
                    skill_version=skill.version,
                    requesting_subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                    requesting_subject_id=str(agent.id),
                ),
                runtime_skill_catalog=catalog,
            )
            self.assertEqual(result.status, SkillExecutionStatus.SANDBOX_ERROR)
            self.assertEqual(sandbox.calls, [])

        asyncio.run(with_session(scenario))

    def test_instructional_skill_is_denied_before_sandbox(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            storage = FakeExecutionStorage()
            skills = SkillService(session, storage=storage)
            agent = await AgentService(session).ensure_primary_agent(user_id=1)
            skill = await skills.upload_user_skill(
                user_id=1,
                key="instructional",
                name="Instructional",
                description="Docs only",
                package=SkillPackage(
                    skill_md="Instructions only",
                    manifest={
                        "key": "instructional",
                        "version": 1,
                        "entrypoint": None,
                    },
                ),
            )
            await skills.assign_to_agent(user_id=1, agent_id=agent.id, skill_id=skill.id)
            run = await AgentRunService(session).create_run(
                user_id=1, starting_agent_id=agent.id, model_name="test"
            )
            await AgentRunService(session).start(user_id=1, run_id=run.id)
            catalog = await skills.resolve_agent_skills(user_id=1, agent_id=agent.id)
            sandbox = FakeSkillSandbox()
            result = await SkillExecutionService(
                session, skill_service=skills, sandbox=sandbox
            ).execute(
                SkillExecutionRequest(
                    run_id=run.id,
                    user_id=1,
                    agent_id=agent.id,
                    skill_id=skill.id,
                    skill_key=skill.key,
                    skill_version=skill.version,
                    requesting_subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                    requesting_subject_id=str(agent.id),
                ),
                runtime_skill_catalog=catalog,
            )
            self.assertEqual(result.status, SkillExecutionStatus.DENIED)
            self.assertEqual(sandbox.calls, [])

        asyncio.run(with_session(scenario))

    def test_archived_skill_is_not_executable(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            _storage, skills, agent, skill, run, catalog = await prepare_skill(session)
            await skills.archive_skill(user_id=1, skill_id=skill.id)
            sandbox = FakeSkillSandbox()
            result = await SkillExecutionService(
                session, skill_service=skills, sandbox=sandbox
            ).execute(
                SkillExecutionRequest(
                    run_id=run.id,
                    user_id=1,
                    agent_id=agent.id,
                    skill_id=skill.id,
                    skill_key=skill.key,
                    skill_version=skill.version,
                    requesting_subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                    requesting_subject_id=str(agent.id),
                ),
                runtime_skill_catalog=catalog,
            )
            self.assertEqual(result.status, SkillExecutionStatus.DENIED)
            self.assertEqual(sandbox.calls, [])

        asyncio.run(with_session(scenario))

    def test_timeout_and_output_limits_are_persisted_as_bounded_results(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            _storage, skills, agent, skill, run, catalog = await prepare_skill(session)
            sandbox = FakeSkillSandbox(
                SkillExecutionResult(
                    status=SkillExecutionStatus.TIMED_OUT,
                    stdout="x" * 100,
                    stderr="e" * 100,
                    error="Skill execution timed out",
                )
            )
            sandbox_config = SandboxConfig(
                python_image="python:3.12.7-slim",
                timeout_seconds=1,
                memory_mb=64,
                cpus=0.25,
                pids_limit=8,
                max_stdout_bytes=16,
                max_stderr_bytes=12,
            )
            result = await SkillExecutionService(
                session,
                skill_service=skills,
                sandbox=sandbox,
                sandbox_config=sandbox_config,
            ).execute(
                SkillExecutionRequest(
                    run_id=run.id,
                    user_id=1,
                    agent_id=agent.id,
                    skill_id=skill.id,
                    skill_key=skill.key,
                    skill_version=skill.version,
                    requesting_subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                    requesting_subject_id=str(agent.id),
                ),
                runtime_skill_catalog=catalog,
            )
            self.assertEqual(result.status, SkillExecutionStatus.TIMED_OUT)
            self.assertTrue(result.truncated)
            self.assertLessEqual(len(result.stdout.encode()), 16)
            self.assertLessEqual(len(result.stderr.encode()), 12)
            persisted = (await session.execute(select(SkillExecutionEntity))).scalar_one()
            self.assertEqual(persisted.status, SkillExecutionStatus.TIMED_OUT)
            self.assertLessEqual(len(persisted.stdout_preview.encode()), 16)
            self.assertLessEqual(len(persisted.stderr_preview.encode()), 12)

        asyncio.run(with_session(scenario))

    def test_skill_required_permission_is_not_granted_implicitly(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            _storage, skills, agent, skill, run, catalog = await prepare_skill(
                session, required_permissions=["read:files:input"]
            )
            sandbox = FakeSkillSandbox()
            authority = BackendToolAuthority(
                session,
                skill_execution_service=SkillExecutionService(
                    session, skill_service=skills, sandbox=sandbox
                ),
            )
            await PermissionService(session).grant(
                user_id=1,
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id=str(agent.id),
                action_class=ActionClass.EXECUTE,
                resource_scope="skill:*",
            )
            from models.tool import ToolRequest

            result = await authority.request(
                ToolRequest(
                    run_id=run.id,
                    user_id=1,
                    requesting_subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                    requesting_subject_id=str(agent.id),
                    name="execute_skill",
                    action_class=ActionClass.EXECUTE,
                    resource=f"skill:{skill.id}:v1",
                    arguments={"skill_key": skill.key, "arguments": {}},
                ),
                runtime_skill_catalog=catalog,
            )
            self.assertEqual(result.status.value, "denied")
            self.assertEqual(sandbox.calls, [])

        asyncio.run(with_session(scenario))

    def test_authority_resolves_assignment_checks_permission_and_approval(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            _storage, skills, agent, skill, run, catalog = await prepare_skill(session)
            sandbox = FakeSkillSandbox(
                SkillExecutionResult(status=SkillExecutionStatus.SUCCEEDED, output={"value": 4})
            )
            execution = SkillExecutionService(session, skill_service=skills, sandbox=sandbox)
            authority = BackendToolAuthority(session, skill_execution_service=execution)
            await PermissionService(session).grant(
                user_id=1,
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id=str(agent.id),
                action_class=ActionClass.EXECUTE,
                resource_scope="skill:*",
            )
            from models.tool import ToolRequest

            request = ToolRequest(
                run_id=run.id,
                user_id=1,
                requesting_subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                requesting_subject_id=str(agent.id),
                name="execute_skill",
                action_class=ActionClass.EXECUTE,
                resource=f"skill:{skill.id}:v{skill.version}",
                arguments={"skill_key": skill.key, "arguments": {"expression": "2 + 2"}},
            )
            pending = await authority.request(request, runtime_skill_catalog=catalog)
            self.assertEqual(pending.status.value, "waiting_approval")
            approval = await ApprovalService(session).get(user_id=1, approval_id=pending.approval_id)
            await ApprovalService(session).approve(user_id=1, approval_id=approval.id)
            result = await authority.execute_approved(
                user_id=1,
                approval_id=approval.id,
                runtime_skill_catalog=catalog,
            )
            self.assertEqual(result.status.value, "executed")
            self.assertEqual(result.output, {"value": 4})
            self.assertEqual(len(sandbox.calls), 1)
            events = await AgentRunService(session).list_events(user_id=1, run_id=run.id)
            self.assertIn(
                AgentRunEventType.SKILL_EXECUTION_REQUESTED,
                [event.event_type for event in events],
            )

        asyncio.run(with_session(scenario))

    def test_unassigned_skill_is_denied_before_sandbox(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            storage = FakeExecutionStorage()
            skills = SkillService(session, storage=storage)
            agent = await AgentService(session).ensure_primary_agent(user_id=1)
            skill = await skills.upload_user_skill(
                user_id=1,
                key="unassigned",
                name="Unassigned",
                description="x",
                package=executable_package(key="unassigned"),
            )
            run = await AgentRunService(session).create_run(
                user_id=1, starting_agent_id=agent.id, model_name="test"
            )
            await AgentRunService(session).start(user_id=1, run_id=run.id)
            catalog = [
                RuntimeSkillDefinition(
                    id=skill.id,
                    key=skill.key,
                    name=skill.name,
                    description=skill.description,
                    version=skill.version,
                    storage_uri=skill.storage_uri,
                    package_checksum=skill.package_checksum,
                    runtime="python",
                    entrypoint="src/main.py",
                )
            ]
            sandbox = FakeSkillSandbox()
            authority = BackendToolAuthority(
                session,
                skill_execution_service=SkillExecutionService(
                    session, skill_service=skills, sandbox=sandbox
                ),
            )
            from models.tool import ToolRequest

            result = await authority.request(
                ToolRequest(
                    run_id=run.id,
                    user_id=1,
                    requesting_subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                    requesting_subject_id=str(agent.id),
                    name="execute_skill",
                    action_class=ActionClass.EXECUTE,
                    resource=f"skill:{skill.id}:v1",
                    arguments={"skill_key": "unassigned", "arguments": {}},
                ),
                runtime_skill_catalog=catalog,
            )
            self.assertEqual(result.status.value, "denied")
            self.assertEqual(sandbox.calls, [])

        asyncio.run(with_session(scenario))


if __name__ == "__main__":
    unittest.main()
