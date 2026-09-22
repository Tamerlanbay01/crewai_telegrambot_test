"""Deterministic sandbox double for unit tests."""

from collections.abc import Mapping

from models.skill import MaterializedSkillPackage, SandboxLimits, SkillExecutionResult


class FakeSkillSandbox:
    def __init__(self, result: SkillExecutionResult | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.requests = self.calls
        self.result = result or SkillExecutionResult(status="SUCCEEDED", output={"ok": True})

    async def execute(
        self,
        *,
        package: MaterializedSkillPackage,
        entrypoint: str,
        arguments: dict[str, object],
        limits: SandboxLimits,
        environment: Mapping[str, str] | None = None,
    ) -> SkillExecutionResult:
        self.calls.append(
            {
                "package": package,
                "entrypoint": entrypoint,
                "arguments": dict(arguments),
                "limits": limits,
                "environment": dict(environment or {}),
            }
        )
        return self.result
