from typing import Protocol

from models.runtime import (
    AgentRuntimeRequest,
    AgentRuntimeResult,
)


class AssistantAgent(Protocol):
    async def run(
        self,
        *,
        message: str,
        history: list[AgentRuntimeRequest],
    ) -> AgentRuntimeResult:
        ...