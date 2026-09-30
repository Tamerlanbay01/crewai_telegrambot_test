"""Expose declared read tools through backend authority without granting new rights."""

import asyncio
import json
from uuid import UUID

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from models.permission import ActionClass, PermissionSubjectType
from models.runtime import RuntimeSkillDefinition, RuntimeToolDeclaration
from models.tool import ToolExecutionStatus, ToolRequest
from services.permission import PermissionService
from services.skill import SkillService
from services.skill_package import SkillPackageValidationError
from services.tool_authority import BackendToolAuthority, ToolExecutor


class BackendToolArguments(BaseModel):
    arguments: dict[str, object] = Field(default_factory=dict)


def never_cache_backend_result(_arguments, _result) -> bool:
    """Permissions and budgets must be checked for every backend invocation."""
    return False


class ToolCallBudget:
    def __init__(self, limit: int) -> None:
        self.limit = max(0, limit)
        self.used = 0

    def consume(self) -> None:
        if self.used >= self.limit:
            raise PermissionError("Runtime tool-call budget exceeded")
        self.used += 1


class BackendReadTool(BaseTool):
    """The native CrewAI surface for an already-authorized backend READ action."""

    name: str = "backend_read"
    description: str = "Read through the backend authority."
    args_schema: type[BaseModel] = BackendToolArguments

    _authority: BackendToolAuthority = PrivateAttr()
    _request: ToolRequest = PrivateAttr()
    _budget: ToolCallBudget | None = PrivateAttr(default=None)
    _execution_lock: asyncio.Lock = PrivateAttr()
    _runtime_permission_scopes: list[str] = PrivateAttr(default_factory=list)
    _skills: SkillService = PrivateAttr()
    _skill: RuntimeSkillDefinition = PrivateAttr()
    _owner_loop: asyncio.AbstractEventLoop = PrivateAttr()

    def __init__(
        self, *, authority: BackendToolAuthority, request: ToolRequest,
        skill_service: SkillService, skill: RuntimeSkillDefinition,
        budget: ToolCallBudget | None = None,
        execution_lock: asyncio.Lock | None = None,
        runtime_permission_scopes: list[str] | None = None,
    ) -> None:
        super().__init__(
            name=request.name,
            description=f"Read {request.resource} through backend authority",
            cache_function=never_cache_backend_result,
        )
        self._authority = authority
        self._request = request
        self._skills = skill_service
        self._skill = skill.model_copy(deep=True)
        self._owner_loop = asyncio.get_running_loop()
        self._budget = budget
        self._execution_lock = execution_lock or asyncio.Lock()
        self._runtime_permission_scopes = list(runtime_permission_scopes or [])

    def _run(self, arguments: dict[str, object] | None = None) -> str:
        """CrewAI's sync worker thread waits while backend DB work uses its loop."""
        try:
            caller_loop = asyncio.get_running_loop()
        except RuntimeError:
            caller_loop = None
        if caller_loop is self._owner_loop:
            raise RuntimeError("Sync tool dispatch cannot block its backend owner loop")
        if self._owner_loop.is_closed():
            raise RuntimeError("Backend tool owner loop is closed")
        pending = asyncio.run_coroutine_threadsafe(self._execute(arguments), self._owner_loop)
        return pending.result()

    async def _arun(self, arguments: dict[str, object] | None = None) -> str:
        if asyncio.get_running_loop() is self._owner_loop:
            return await self._execute(arguments)
        pending = asyncio.run_coroutine_threadsafe(self._execute(arguments), self._owner_loop)
        return await asyncio.wrap_future(pending)

    async def _execute(self, arguments: dict[str, object] | None) -> str:
        # All native tools share the resolver's AsyncSession; serialize DB work.
        async with self._execution_lock:
            if self._budget is not None:
                self._budget.consume()
            try:
                await self._skills.get_runtime_skill(
                    user_id=self._request.user_id, definition=self._skill,
                    agent_id=(UUID(self._request.requesting_subject_id)
                              if self._request.requesting_subject_type == PermissionSubjectType.PERSISTENT_AGENT
                              else None),
                    run_id=self._request.run_id,
                )
            except (LookupError, SkillPackageValidationError):
                return json.dumps({"status": "denied", "output": None,
                                   "error": "Skill is no longer available to this run"})
            result = await self._authority.request(
                self._request.model_copy(update={"arguments": arguments or {}}),
                runtime_permission_scopes=self._runtime_permission_scopes,
            )
        return json.dumps(
            {"status": result.status.value,
             "output": result.output if result.status == ToolExecutionStatus.EXECUTED else None,
             "error": result.error},
            ensure_ascii=False, default=str,
        )


class ToolRuntimeResolver:
    """Fail on unknown declarations, filter by backend permission, expose READ only."""

    def __init__(
        self, session, *, executor: ToolExecutor | None = None,
    ) -> None:
        if executor is None:
            raise RuntimeError("ToolExecutor is not configured")
        selected = executor
        self._executor = selected
        self._permissions = PermissionService(session)
        self._authority = BackendToolAuthority(session, executor=selected)
        self._skills = SkillService(session)
        self._execution_lock = asyncio.Lock()

    async def resolve(
        self,
        *,
        run_id: UUID,
        user_id: int,
        subject_type: PermissionSubjectType,
        subject_id: str,
        skills: list[RuntimeSkillDefinition],
        budget: ToolCallBudget | None = None,
        runtime_permission_scopes: list[str] | None = None,
    ) -> list[BaseTool]:
        tools: list[BaseTool] = []
        seen: set[str] = set()
        for skill in skills:
            if skill.declared_tools:
                if skill.id is None:
                    raise ValueError("Declared tool has no backend skill identity")
                persisted = await self._skills.get_runtime_skill(
                    user_id=user_id, definition=skill,
                    agent_id=UUID(subject_id) if subject_type == PermissionSubjectType.PERSISTENT_AGENT else None,
                    run_id=run_id,
                )
                actual = [RuntimeToolDeclaration.model_validate(item)
                          for item in persisted.manifest.get("tools", [])]
                if actual != skill.declared_tools:
                    raise ValueError("Declared tools differ from backend skill manifest")
            for declared in skill.declared_tools:
                definition = self._executor.definition(declared.id)
                if definition is None or not declared.resource.startswith(definition.resource_prefix):
                    raise ValueError(f"Unknown or invalid declared tool: {declared.id}")
                if declared.id in seen:
                    raise ValueError(f"Duplicate declared tool: {declared.id}")
                seen.add(declared.id)
                if subject_type == PermissionSubjectType.TEMPORARY_SUBAGENT:
                    allowed = BackendToolAuthority._runtime_permission_allows(
                        runtime_permission_scopes or [], definition.action_class,
                        declared.resource,
                    )
                else:
                    allowed = await self._permissions.check(
                        user_id=user_id,
                        subject_type=subject_type,
                        subject_id=subject_id,
                        action_class=definition.action_class,
                        resource=declared.resource,
                    )
                if not allowed or definition.action_class != ActionClass.READ:
                    # Side effects continue through RuntimeStep -> approval checkpoint.
                    continue
                tools.append(BackendReadTool(
                    authority=self._authority,
                    skill_service=self._skills,
                    skill=skill,
                    budget=budget,
                    execution_lock=self._execution_lock,
                    runtime_permission_scopes=runtime_permission_scopes,
                    request=ToolRequest(
                        run_id=run_id,
                        user_id=user_id,
                        requesting_subject_type=subject_type,
                        requesting_subject_id=subject_id,
                        name=declared.id,
                        action_class=definition.action_class,
                        resource=declared.resource,
                    ),
                ))
        return tools
