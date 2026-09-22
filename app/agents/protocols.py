from typing import Protocol

from models.runtime import (
    AgentRuntimeContext,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    RuntimeStep,
    RuntimeSkillDefinition,
)
from models.tool import ToolExecutionResult, ToolRequest


class AgentRuntime(Protocol):
    async def run(
        self,
        request: AgentRuntimeRequest,
    ) -> AgentRuntimeResult:
        ...


class AgentExecutionRuntime(Protocol):
    async def run(
        self,
        context: AgentRuntimeContext,
        request: AgentRuntimeRequest,
    ) -> RuntimeStep:
        ...


class ToolApprovalRuntime(Protocol):
    async def begin(
        self,
        request: ToolRequest,
        *,
        runtime_permission_scopes: list[str] | None = None,
        runtime_skill_catalog: list[RuntimeSkillDefinition] | None = None,
    ) -> ToolExecutionResult:
        ...

    async def resume(self, *, user_id: int, flow_id: str) -> ToolExecutionResult:
        ...
