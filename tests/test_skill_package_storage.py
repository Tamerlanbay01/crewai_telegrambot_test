from __future__ import annotations

import asyncio
import unittest
from collections.abc import Awaitable, Callable, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from database.base import Base
from database.entities.agent import AgentEntity
from database.entities.agent_prompt import AgentPromptVersionEntity
from database.entities.agent_skill import AgentSkillEntity
from database.entities.skill import SkillEntity
from database.entities.user import UserEntity
from core.config import config
from integrations.storage.s3 import S3SkillStorage
from integrations.storage.skill_storage import SkillStorageConflictError, SkillStorageError
from models.skill import PreparedSkillPackage, SkillPackage, SkillPackageFile
from services.agent import AgentService
from services.skill import SkillPackagePersistenceError, SkillService
from services.skill_package import SkillPackageService, SkillPackageValidationError


class FakeSkillStorage:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, bytes]] = {}
        self.checksums: dict[str, str] = {}
        self.fail_upload = False

    async def upload_package(
        self, *, prefix: str, files: Sequence[SkillPackageFile], checksum: str
    ) -> None:
        if self.fail_upload:
            raise SkillStorageError("storage unavailable")
        existing = self.checksums.get(prefix)
        if existing is not None:
            if existing == checksum:
                return
            raise SkillStorageConflictError("immutable prefix already exists")
        if prefix in self.objects:
            raise SkillStorageConflictError("orphaned prefix already exists")
        self.objects[prefix] = {item.path: item.content for item in files}
        self.checksums[prefix] = checksum

    async def get_file(self, *, prefix: str, path: str, max_bytes: int) -> bytes:
        try:
            content = self.objects[prefix][path]
        except KeyError as exc:
            raise SkillStorageError("object missing") from exc
        if len(content) > max_bytes:
            raise SkillStorageError("object exceeds limit")
        return content

    async def list_files(self, *, prefix: str) -> list[str]:
        return sorted(self.objects.get(prefix, {}))

    async def delete_version(self, *, prefix: str) -> None:
        self.objects.pop(prefix, None)
        self.checksums.pop(prefix, None)

    async def exists(self, *, prefix: str) -> bool:
        return prefix in self.objects

    async def get_package_checksum(self, *, prefix: str) -> str | None:
        return self.checksums.get(prefix)


class FakeS3NotFoundError(Exception):
    def __init__(self) -> None:
        super().__init__("not found")
        self.response = {"Error": {"Code": "404"}}


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.fail_once_on_key: str | None = None
        self.delete_calls = 0

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        if Key not in self.objects:
            raise FakeS3NotFoundError()
        return {"Metadata": self.metadata.get(Key, {})}

    def list_objects_v2(self, *, Bucket: str, Prefix: str, MaxKeys: int | None = None) -> dict[str, object]:
        keys = sorted(key for key in self.objects if key.startswith(Prefix))
        if MaxKeys is not None:
            keys = keys[:MaxKeys]
        return {"Contents": [{"Key": key} for key in keys], "IsTruncated": False}

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
        Metadata: dict[str, str],
    ) -> None:
        self.objects[Key] = Body
        self.metadata[Key] = Metadata
        if Key == self.fail_once_on_key:
            self.fail_once_on_key = None
            raise RuntimeError("connection dropped after object write")

    def delete_objects(self, *, Bucket: str, Delete: dict[str, object]) -> None:
        self.delete_calls += 1
        for item in Delete["Objects"]:  # type: ignore[index]
            key = item["Key"]  # type: ignore[index]
            self.objects.pop(key, None)
            self.metadata.pop(key, None)


def package(*, key: str = "python_helper", version: int = 1, skill_md: str = "Use Python safely.") -> SkillPackage:
    return SkillPackage(
        skill_md=skill_md,
        manifest={
            "key": key,
            "version": version,
            "entrypoint": None,
            "description": "Helpful Python guidance",
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
            session.add_all([UserEntity(id=1, telegram_id=101), UserEntity(id=2, telegram_id=202)])
            await session.commit()
            await test(session)
    finally:
        await engine.dispose()


class SkillPackageStorageTests(unittest.TestCase):
    def test_prepared_package_is_a_pydantic_model(self) -> None:
        prepared = SkillPackageService(storage_provider=FakeSkillStorage, config=config.s3).validate(
            package=package(), key="python_helper", version=1
        )
        self.assertIsInstance(prepared, PreparedSkillPackage)

    def test_validation_rejects_required_files_manifest_mismatch_and_unsafe_paths(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            service = SkillService(session, storage=FakeSkillStorage())
            with self.assertRaises(SkillPackageValidationError):
                await service.upload_user_skill(
                    user_id=1, key="python_helper", name="Python", description="x",
                    package=SkillPackage(manifest={"key": "python_helper", "version": 1}),
                )

            original_file_limit = config.s3.max_skill_file_bytes
            original_package_limit = config.s3.max_skill_package_bytes
            try:
                config.s3.max_skill_file_bytes = 3
                with self.assertRaises(SkillPackageValidationError):
                    await service.upload_user_skill(
                        user_id=1, key="oversized", name="Oversized", description="x",
                        package=package(key="oversized"),
                    )
                config.s3.max_skill_file_bytes = original_file_limit
                config.s3.max_skill_package_bytes = 10
                with self.assertRaises(SkillPackageValidationError):
                    await service.upload_user_skill(
                        user_id=1, key="too_large", name="Too large", description="x",
                        package=package(key="too_large"),
                    )
            finally:
                config.s3.max_skill_file_bytes = original_file_limit
                config.s3.max_skill_package_bytes = original_package_limit
            with self.assertRaises(SkillPackageValidationError):
                await service.upload_user_skill(
                    user_id=1, key="python_helper", name="Python", description="x",
                    package=SkillPackage(
                        files=[
                            SkillPackageFile(path="SKILL.md", content=b"x"),
                            SkillPackageFile(path="manifest.json", content=b"not-json"),
                        ]
                    ),
                )
            with self.assertRaises(SkillPackageValidationError):
                await service.upload_user_skill(
                    user_id=1, key="python_helper", name="Python", description="x",
                    package=SkillPackage(
                        skill_md="x",
                        manifest={"key": "other", "version": 1},
                    ),
                )
            with self.assertRaises(SkillPackageValidationError):
                await service.upload_user_skill(
                    user_id=1, key="python_helper", name="Python", description="x",
                    package=SkillPackage(
                        skill_md="x",
                        manifest={"key": "python_helper", "version": 1},
                        files=[SkillPackageFile(path="../secret", content=b"no")],
                    ),
                )
            with self.assertRaises(SkillPackageValidationError):
                await service.upload_user_skill(
                    user_id=1, key="python_helper", name="Python", description="x",
                    package=SkillPackage(
                        skill_md="x",
                        manifest={
                            "key": "python_helper",
                            "version": 1,
                            "files": ["src/a.py"],
                        },
                        files=[SkillPackageFile(path="src/b.py", content=b"print('b')\n")],
                    ),
                )

        asyncio.run(with_session(scenario))

    def test_s3_retries_after_partial_upload_by_cleaning_incomplete_prefix(self) -> None:
        async def scenario() -> None:
            client = FakeS3Client()
            prefix = "users/1/python_helper/v1/"
            client.fail_once_on_key = f"{prefix}src/main.py"
            storage = object.__new__(S3SkillStorage)
            storage._bucket = "skills"
            storage._client = client
            files = [
                SkillPackageFile(path="SKILL.md", content=b"instructions"),
                SkillPackageFile(path="manifest.json", content=b"{}"),
                SkillPackageFile(path="src/main.py", content=b"print('safe')\n"),
            ]

            with self.assertRaises(SkillStorageError):
                await storage.upload_package(prefix=prefix, files=files, checksum="a" * 64)
            self.assertNotIn(f"{prefix}manifest.json", client.objects)
            self.assertTrue(client.objects)

            await storage.upload_package(prefix=prefix, files=files, checksum="a" * 64)
            self.assertEqual(
                sorted(key.removeprefix(prefix) for key in client.objects),
                ["SKILL.md", "manifest.json", "src/main.py"],
            )
            self.assertEqual(await storage.get_package_checksum(prefix=prefix), "a" * 64)
            self.assertEqual(client.delete_calls, 1)

            await storage.upload_package(prefix=prefix, files=files, checksum="a" * 64)
            self.assertEqual(client.delete_calls, 1)

            with self.assertRaises(SkillStorageConflictError):
                await storage.upload_package(prefix=prefix, files=files, checksum="b" * 64)
            self.assertEqual(client.delete_calls, 1)

        asyncio.run(scenario())

    def test_upload_is_versioned_idempotent_and_tenant_scoped(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            storage = FakeSkillStorage()
            service = SkillService(session, storage=storage)
            first = await service.upload_user_skill(
                user_id=1, key="python_helper", name="Python", description="x", package=package()
            )
            self.assertEqual(first.storage_uri, "s3://agent-skills/users/1/python_helper/v1/")
            self.assertIsNotNone(first.package_checksum)
            retry = await service.upload_user_skill(
                user_id=1, key="python_helper", name="Changed name", description="changed", package=package()
            )
            self.assertEqual(retry.id, first.id)
            with self.assertRaises(ValueError):
                await service.upload_user_skill(
                    user_id=1, key="python_helper", name="Python", description="x",
                    package=package(skill_md="Different content"),
                )
            second = await service.upload_user_skill(
                user_id=1, key="python_helper", name="Python", description="x",
                version=2, package=package(version=2),
            )
            other = await service.upload_user_skill(
                user_id=2, key="python_helper", name="Python", description="x", package=package()
            )
            self.assertEqual(second.storage_uri, "s3://agent-skills/users/1/python_helper/v2/")
            self.assertEqual(other.storage_uri, "s3://agent-skills/users/2/python_helper/v1/")
            self.assertEqual(await service.list_package_files(user_id=1, skill_id=first.id), [
                "SKILL.md", "manifest.json", "src/main.py"
            ])
            self.assertEqual(await service.get_package_file(user_id=1, skill_id=first.id, path="SKILL.md"), b"Use Python safely.")
            with self.assertRaises(LookupError):
                await service.get_package_file(user_id=2, skill_id=first.id, path="SKILL.md")
            self.assertTrue(await service.verify_skill_storage(user_id=1, skill_id=first.id))

        asyncio.run(with_session(scenario))

    def test_storage_failure_creates_no_metadata_and_db_failure_leaves_inspectable_orphan(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            storage = FakeSkillStorage()
            storage.fail_upload = True
            service = SkillService(session, storage=storage)
            with self.assertRaises(SkillStorageError):
                await service.upload_user_skill(
                    user_id=1, key="python_helper", name="Python", description="x", package=package()
                )
            self.assertEqual((await session.execute(select(SkillEntity))).scalars().all(), [])

            storage.fail_upload = False

            async def fail_create(_data):
                raise RuntimeError("database unavailable")

            service._skills.create = fail_create  # type: ignore[method-assign]
            with self.assertRaises(SkillPackagePersistenceError):
                await service.upload_user_skill(
                    user_id=1, key="orphan", name="Orphan", description="x", package=package(key="orphan")
                )
            self.assertIn("users/1/orphan/v1/", storage.objects)

        asyncio.run(with_session(scenario))

    def test_runtime_loads_skill_md_and_archive_preserves_then_allows_privileged_delete(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            storage = FakeSkillStorage()
            service = SkillService(session, storage=storage)
            agent = await AgentService(session).ensure_primary_agent(user_id=1)
            skill = await service.upload_user_skill(
                user_id=1, key="python_helper", name="Python", description="x", package=package()
            )
            await service.assign_to_agent(user_id=1, agent_id=agent.id, skill_id=skill.id)
            runtime = await service.resolve_agent_skills(user_id=1, agent_id=agent.id)
            self.assertEqual(runtime[0].instructions, "Use Python safely.")
            self.assertEqual(runtime[0].storage_uri, skill.storage_uri)
            archived = await service.archive_skill(user_id=1, skill_id=skill.id)
            self.assertEqual(archived.status.value, "ARCHIVED")
            prefix = "users/1/python_helper/v1/"
            self.assertIn(prefix, storage.objects)
            await service.delete_skill_version_storage(skill_id=skill.id)
            self.assertNotIn(prefix, storage.objects)

        asyncio.run(with_session(scenario))

    def test_system_prefix_and_missing_runtime_package_are_safe(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            storage = FakeSkillStorage()
            service = SkillService(session, storage=storage)
            system = await service.create_system_skill(
                key="web_research", name="Web research", description="x",
                package=package(key="web_research"),
            )
            self.assertEqual(system.storage_uri, "s3://agent-skills/system/web_research/v1/")
            self.assertEqual(
                await service.list_package_files(user_id=2, skill_id=system.id),
                ["SKILL.md", "manifest.json", "src/main.py"],
            )

            agent = await AgentService(session).ensure_primary_agent(user_id=1)
            broken = await service.upload_user_skill(
                user_id=1, key="broken", name="Broken", description="x", package=package(key="broken")
            )
            await service.assign_to_agent(user_id=1, agent_id=agent.id, skill_id=broken.id)
            storage.objects.pop("users/1/broken/v1/")
            storage.checksums.pop("users/1/broken/v1/")
            self.assertEqual(await service.resolve_agent_skills(user_id=1, agent_id=agent.id), [])

        asyncio.run(with_session(scenario))
