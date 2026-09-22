"""Execution boundary used by the skill execution service."""

from collections.abc import Mapping
from typing import Protocol

from models.skill import MaterializedSkillPackage, SandboxLimits, SkillExecutionResult


class SkillSandbox(Protocol):
    async def execute(
        self,
        *,
        package: MaterializedSkillPackage,
        entrypoint: str,
        arguments: dict[str, object],
        limits: SandboxLimits,
        environment: Mapping[str, str] | None = None,
    ) -> SkillExecutionResult:
        """Execute a verified package without exposing host execution."""
