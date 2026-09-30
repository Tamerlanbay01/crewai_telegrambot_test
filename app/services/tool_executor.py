"""Explicit production tool registration; unknown tools fail closed."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from models.tool import ToolDefinition


class ToolExecutor(Protocol):
    def definition(self, name: str) -> ToolDefinition | None: ...

    async def execute(self, name: str, arguments: dict[str, object]) -> object: ...


ToolHandler = Callable[[dict[str, object]], Awaitable[object]]


class RegisteredTool:
    """A backend tool definition paired with its installed async handler."""

    def __init__(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        self.definition = definition
        self.handler = handler


class ToolRegistry:
    """Dispatch only backend-installed handlers, never skill package code."""

    def __init__(self, tools: Sequence[RegisteredTool] = ()) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        for tool in tools:
            name = tool.definition.name
            if name == "execute_skill":
                raise ValueError("execute_skill is owned by BackendToolAuthority")
            if name in self._tools:
                raise ValueError(f"Duplicate registered tool: {name}")
            self._tools[name] = RegisteredTool(tool.definition.model_copy(deep=True), tool.handler)

    def definition(self, name: str) -> ToolDefinition | None:
        registered = self._tools.get(name)
        return registered.definition.model_copy(deep=True) if registered else None

    async def execute(self, name: str, arguments: dict[str, object]) -> object:
        registered = self._tools.get(name)
        if registered is None:
            raise LookupError(f"ToolExecutor is not configured for tool: {name}")
        return await registered.handler(dict(arguments))


def create_tool_executor() -> ToolRegistry:
    """Production composition seam for real connectors as they are installed.

    No external READ/WRITE connectors are installed yet. execute_skill remains
    available through BackendToolAuthority and its configured sandbox service.
    """
    return ToolRegistry()
