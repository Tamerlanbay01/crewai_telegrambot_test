from typing import Protocol
from models.agent_factory import CrewDefinition

from models.runtime import (
    AgentRuntimeContext,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    RuntimeStep,
    RuntimeSkillDefinition,
    RuntimeCrewResult,
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

    async def run_crew(
        self, context: AgentRuntimeContext, definition: CrewDefinition,
        *, task_summary: str,
    ) -> RuntimeCrewResult:
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
