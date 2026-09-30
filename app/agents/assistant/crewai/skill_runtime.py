"""Materialize backend-verified skill packages for CrewAI's native loader."""

import json
import re
import tempfile
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

from crewai.skills import discover_skills
from crewai.skills.models import Skill as NativeSkill

from models.runtime import RuntimeSkillDefinition
from models.skill import PreparedSkillPackage
from services.skill import SkillService
from services.skill_package import SkillPackageValidationError, SkillPackageService


class SkillRuntimeResolver:
    """No execution or commits: only pin checks, bytes verification, and native discovery."""

    def __init__(self, skills: SkillService) -> None:
        self._skills = skills

    @asynccontextmanager
    async def materialize_for_agent(
        self,
        *,
        user_id: int,
        agent_id: UUID | None,
        definitions: Sequence[RuntimeSkillDefinition],
        run_id: UUID | None = None,
    ) -> AsyncIterator[list[NativeSkill]]:
        with tempfile.TemporaryDirectory(prefix="crewai-native-skills-") as temporary:
            root = Path(temporary) / str(user_id) / str(run_id or uuid4()) / "skills"
            native = await self._materialize(
                root=root, user_id=user_id, agent_id=agent_id, definitions=definitions,
                run_id=run_id,
            )
            yield native

    @asynccontextmanager
    async def materialize_for_crew(
        self,
        *,
        user_id: int,
        agents: Sequence[tuple[UUID, Sequence[RuntimeSkillDefinition]]],
        run_id: UUID | None = None,
    ) -> AsyncIterator[dict[UUID, list[NativeSkill]]]:
        with tempfile.TemporaryDirectory(prefix="crewai-native-skills-") as temporary:
            root = Path(temporary) / str(user_id) / str(run_id or uuid4()) / "skills"
            resolved = {}
            for agent_id, definitions in agents:
                resolved[agent_id] = await self._materialize(
                    root=root / str(agent_id),
                    user_id=user_id,
                    agent_id=agent_id,
                    definitions=definitions,
                    run_id=run_id,
                )
            yield resolved

    async def _materialize(
        self,
        *,
        root: Path,
        user_id: int,
        agent_id: UUID | None,
        definitions: Sequence[RuntimeSkillDefinition],
        run_id: UUID | None,
    ) -> list[NativeSkill]:
        native: list[NativeSkill] = []
        seen: set[UUID] = set()
        for definition in definitions:
            if definition.storage_uri is None:
                if definition.id is not None:
                    await self._skills.get_runtime_skill(
                        user_id=user_id, definition=definition,
                        agent_id=agent_id, run_id=run_id,
                    )
                continue  # Legacy metadata-only skill instructions remain in the runtime prompt.
            if definition.id is None or definition.id in seen:
                raise SkillPackageValidationError("Runtime skill identity is missing or duplicated")
            seen.add(definition.id)
            prepared = await self._skills.get_verified_runtime_package(
                user_id=user_id, definition=definition, agent_id=agent_id,
                run_id=run_id,
            )
            parent = root / str(definition.id) / f"v{definition.version}"
            skill_dir = parent / self._slug(definition)
            self._write_package(skill_dir=skill_dir, prepared=prepared, definition=definition)
            discovered = discover_skills(parent)
            if len(discovered) != 1 or discovered[0].path != skill_dir:
                raise SkillPackageValidationError("CrewAI could not discover the verified skill")
            native.extend(discovered)
        return native

    @staticmethod
    def _slug(definition: RuntimeSkillDefinition) -> str:
        assert definition.id is not None
        stem = re.sub(r"[^a-z0-9]+", "-", definition.key.lower()).strip("-")[:24].strip("-")
        return f"{stem or 'skill'}-{definition.id.hex}"

    @staticmethod
    def _write_package(
        *, skill_dir: Path, prepared: PreparedSkillPackage,
        definition: RuntimeSkillDefinition,
    ) -> None:
        root = skill_dir.resolve()
        seen: set[str] = set()
        skill_md: bytes | None = None
        paths = {item.path.casefold() for item in prepared.files}
        for item in prepared.files:
            relative = PurePosixPath(item.path)
            if relative.parts and relative.parts[0] == "resources":
                alias = PurePosixPath("references", *relative.parts[1:]).as_posix()
                if alias.casefold() in paths:
                    raise SkillPackageValidationError("Resource alias collides with a package file")
        for item in prepared.files:
            SkillPackageService.validate_path(item.path)
            relative = PurePosixPath(item.path)
            if relative.as_posix() != item.path or item.path.casefold() in seen:
                raise SkillPackageValidationError("Package path is ambiguous")
            seen.add(item.path.casefold())
            if item.path == "SKILL.md":
                skill_md = item.content
                continue
            target = skill_dir.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(root):
                raise SkillPackageValidationError("Package path escapes materialization root")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.content)
            if relative.parts[0] == "resources":
                # CrewAI catalogs references/; retain the original path for SKILL.md links.
                alias = skill_dir.joinpath("references", *relative.parts[1:])
                alias.parent.mkdir(parents=True, exist_ok=True)
                alias.write_bytes(item.content)
        if skill_md is None:
            raise SkillPackageValidationError("Verified package is missing SKILL.md")
        try:
            body = SkillRuntimeResolver._body(skill_md.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise SkillPackageValidationError("SKILL.md must be UTF-8") from exc
        description = (definition.description or definition.name or "").strip() or "Skill instructions"
        frontmatter = (
            "---\n"
            f"name: {SkillRuntimeResolver._slug(definition)}\n"
            f"description: {json.dumps(description[:1024], ensure_ascii=False)}\n"
            "---\n"
        )
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(frontmatter + body, encoding="utf-8")

    @staticmethod
    def _body(text: str) -> str:
        lines = text.splitlines(keepends=True)
        if lines and lines[0].strip() == "---":
            for index, line in enumerate(lines[1:], 1):
                if line.strip() == "---":
                    return "".join(lines[index + 1:])
            raise SkillPackageValidationError("SKILL.md frontmatter is incomplete")
        return text
