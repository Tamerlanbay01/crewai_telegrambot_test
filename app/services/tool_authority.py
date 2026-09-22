"""Backend authority gate for every CrewAI-requested tool action."""

from datetime import datetime, timezone
from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from models.approval import ApprovalStatus
from models.agent_run import AgentRunEventCreate, AgentRunEventType, AgentRunStatus
from models.permission import ActionClass
from models.runtime import RuntimeSkillDefinition
from models.skill import (
    ExecutableSkillDefinition,
    SkillExecutionRequest,
    SkillExecutionResult,
    SkillExecutionStatus,
)
from models.tool import (
    ToolDefinition,
    ToolExecutionResult,
    ToolExecutionStatus,
    ToolRequest,
)
from repositories.agent_run import AgentRunRepository
from repositories.approval import ApprovalRepository
from services.agent_run import AgentRunService
from services.approval import ApprovalService
from services.permission import PermissionDeniedError, PermissionService
from services.skill import SkillExecutionDeniedError
from services.skill_execution import SkillExecutionService


SKILL_TOOL_NAME = "execute_skill"
SKILL_TOOL_RESOURCE_PREFIX = "skill:"
SKILL_TOOL_DEFINITION = ToolDefinition(
    name=SKILL_TOOL_NAME,
    action_class=ActionClass.EXECUTE,
    resource_prefix=SKILL_TOOL_RESOURCE_PREFIX,
)


class ToolExecutor(Protocol):
    def definition(self, name: str) -> ToolDefinition | None: ...

    async def execute(self, name: str, arguments: dict[str, object]) -> object: ...


class FakeToolExecutor:
    """Deterministic integration boundary used until real connectors exist."""

    DEFINITIONS = {
        "read_information": ToolDefinition(
            name="read_information",
            action_class=ActionClass.READ,
            resource_prefix="information:",
        ),
        "write_note": ToolDefinition(
            name="write_note",
            action_class=ActionClass.WRITE,
            resource_prefix="notes:",
        ),
        "external_action": ToolDefinition(
            name="external_action",
            action_class=ActionClass.EXTERNAL_SIDE_EFFECT,
            resource_prefix="calendar:",
        ),
        SKILL_TOOL_NAME: SKILL_TOOL_DEFINITION,
    }

    def __init__(self) -> None:
        self.notes: list[str] = []
        self.external_actions: list[dict[str, object]] = []

    def definition(self, name: str) -> ToolDefinition | None:
        return self.DEFINITIONS.get(name)

    async def execute(self, name: str, arguments: dict[str, object]) -> object:
        if name == "read_information":
            return {
                "query": str(arguments.get("query", "")),
                "information": "fake result",
            }
        if name == "write_note":
            text = str(arguments.get("text", ""))
            self.notes.append(text)
            return {"written": True, "text": text}
        if name == "external_action":
            self.external_actions.append(dict(arguments))
            return {"executed": True}
        raise LookupError(f"Unknown tool: {name}")


class BackendToolAuthority:
    def __init__(
        self,
        session: AsyncSession,
        *,
        executor: ToolExecutor | None = None,
        skill_execution_service: SkillExecutionService | None = None,
    ):
        self._session = session
        self._executor = executor or FakeToolExecutor()
        self._permissions = PermissionService(session)
        self._approvals = ApprovalService(session)
        self._approval_repository = ApprovalRepository(session)
        self._runs = AgentRunRepository(session)
        self._run_service = AgentRunService(session)
        self._skill_execution = skill_execution_service or SkillExecutionService(session)

    async def request(
        self,
        request: ToolRequest,
        *,
        runtime_permission_scopes: list[str] | None = None,
        flow_id: str | None = None,
        runtime_skill_catalog: Sequence[RuntimeSkillDefinition] = (),
    ) -> ToolExecutionResult:
        run = await self._run_service.get_run(user_id=request.user_id, run_id=request.run_id)
        if run.status != AgentRunStatus.RUNNING:
            raise ValueError("Tools can only be requested by a running run")
        definition = self._require_definition(request)
        effective_request = request
        skill_definition: ExecutableSkillDefinition | None = None
        if request.name == SKILL_TOOL_NAME:
            try:
                effective_request, skill_definition = await self._prepare_skill_request(
                    request, runtime_skill_catalog
                )
            except SkillExecutionDeniedError as exc:
                await self._append_event(
                    run_id=request.run_id,
                    event_type=AgentRunEventType.TOOL_REQUESTED,
                    payload=self._audit_payload(request),
                )
                await self._append_event(
                    run_id=request.run_id,
                    event_type=AgentRunEventType.SKILL_EXECUTION_REQUESTED,
                    payload=self._audit_payload(request),
                )
                await self._append_event(
                    run_id=request.run_id,
                    event_type=AgentRunEventType.SKILL_EXECUTION_FAILED,
                    payload={**self._audit_payload(request), "error": str(exc)},
                    commit=True,
                )
                return ToolExecutionResult(
                    status=ToolExecutionStatus.DENIED,
                    error=str(exc),
                )
        await self._append_event(
            run_id=effective_request.run_id,
            event_type=AgentRunEventType.TOOL_REQUESTED,
            payload=self._audit_payload(effective_request),
        )
        if skill_definition is not None:
            await self._append_event(
                run_id=effective_request.run_id,
                event_type=AgentRunEventType.SKILL_EXECUTION_REQUESTED,
                payload=self._audit_payload(effective_request),
            )
        try:
            if effective_request.requesting_subject_type.value == "temporary_subagent":
                if not self._runtime_permission_allows(
                    runtime_permission_scopes or [],
                    definition.action_class,
                    effective_request.resource,
                ):
                    raise PermissionDeniedError(
                        "Temporary subagent permission is outside its inherited runtime scope"
                    )
                if skill_definition is not None:
                    self._require_runtime_skill_permissions(
                        runtime_permission_scopes or [], skill_definition
                    )
            else:
                await self._permissions.require(
                    user_id=effective_request.user_id,
                    subject_type=effective_request.requesting_subject_type,
                    subject_id=effective_request.requesting_subject_id,
                    action_class=definition.action_class,
                    resource=effective_request.resource,
                )
                if skill_definition is not None:
                    await self._require_skill_permissions(effective_request, skill_definition)
        except PermissionDeniedError as exc:
            await self._append_event(
                run_id=effective_request.run_id,
                event_type=AgentRunEventType.PERMISSION_DENIED,
                payload={**self._audit_payload(effective_request), "error": str(exc)},
                commit=True,
            )
            if skill_definition is not None:
                await self._append_event(
                    run_id=effective_request.run_id,
                    event_type=AgentRunEventType.SKILL_EXECUTION_FAILED,
                    payload={**self._audit_payload(effective_request), "error": str(exc)},
                    commit=True,
                )
            return ToolExecutionResult(
                status=ToolExecutionStatus.DENIED,
                error=str(exc),
            )

        if definition.action_class != ActionClass.READ:
            approval = await self._approvals.request(
                user_id=effective_request.user_id,
                run_id=effective_request.run_id,
                flow_id=flow_id,
                requesting_subject_type=effective_request.requesting_subject_type,
                requesting_subject_id=effective_request.requesting_subject_id,
                action_name=effective_request.name,
                action_class=definition.action_class,
                resource=effective_request.resource,
                arguments=effective_request.arguments,
            )
            return ToolExecutionResult(
                status=ToolExecutionStatus.WAITING_APPROVAL,
                approval_id=approval.id,
            )

        return await self._execute(
            effective_request,
            runtime_skill_catalog=runtime_skill_catalog,
            skill_definition=skill_definition,
        )

    async def execute_approved(
        self,
        *,
        user_id: int,
        approval_id: UUID,
        runtime_permission_scopes: list[str] | None = None,
        runtime_skill_catalog: Sequence[RuntimeSkillDefinition] = (),
    ) -> ToolExecutionResult:
        approval = await self._approvals.get(user_id=user_id, approval_id=approval_id)
        if approval.status in {
            ApprovalStatus.REJECTED,
            ApprovalStatus.EXPIRED,
            ApprovalStatus.CANCELLED,
        }:
            return ToolExecutionResult(
                status=ToolExecutionStatus.REJECTED,
                error="Action was rejected by the user",
                approval_id=approval.id,
            )
        if approval.status != ApprovalStatus.APPROVED:
            raise ValueError(f"Approval is not approved: {approval.status.value}")
        run = await self._run_service.get_run(user_id=user_id, run_id=approval.run_id)
        if run.status != AgentRunStatus.RUNNING:
            raise ValueError("Approved tool action requires a running AgentRun")
        if approval.decision_metadata.get("executed_at"):
            raise ValueError("Approved action was already executed")
        request = ToolRequest(
            run_id=approval.run_id,
            user_id=approval.user_id,
            requesting_subject_type=approval.requesting_subject_type,
            requesting_subject_id=approval.requesting_subject_id,
            name=approval.action_name,
            action_class=approval.action_class,
            resource=approval.resource,
            arguments=approval.arguments,
        )
        definition = self._require_definition(request)
        effective_request = request
        skill_definition: ExecutableSkillDefinition | None = None
        if request.name == SKILL_TOOL_NAME:
            try:
                effective_request, skill_definition = await self._prepare_skill_request(
                    request, runtime_skill_catalog
                )
            except SkillExecutionDeniedError as exc:
                return ToolExecutionResult(
                    status=ToolExecutionStatus.DENIED,
                    error=str(exc),
                    approval_id=approval.id,
                )
        if approval.requesting_subject_type.value == "temporary_subagent":
            if not self._runtime_permission_allows(
                runtime_permission_scopes or [],
                definition.action_class,
                effective_request.resource,
            ):
                raise PermissionDeniedError(
                    "Temporary subagent permission is outside its inherited runtime scope"
                )
            if skill_definition is not None:
                self._require_runtime_skill_permissions(
                    runtime_permission_scopes or [], skill_definition
                )
        else:
            await self._permissions.require(
                user_id=approval.user_id,
                subject_type=approval.requesting_subject_type,
                subject_id=approval.requesting_subject_id,
                action_class=definition.action_class,
                resource=effective_request.resource,
            )
            if skill_definition is not None:
                await self._require_skill_permissions(effective_request, skill_definition)
        result = await self._execute(
            effective_request,
            runtime_skill_catalog=runtime_skill_catalog,
            skill_definition=skill_definition,
        )
        metadata = dict(approval.decision_metadata)
        metadata["executed_at"] = datetime.now(timezone.utc).isoformat()
        await self._approval_repository.update_decision_metadata(
            approval_id=approval.id, decision_metadata=metadata
        )
        await self._session.commit()
        return result.model_copy(update={"approval_id": approval.id})

    async def get_approval(self, *, user_id: int, approval_id: UUID):
        return await self._approvals.get(user_id=user_id, approval_id=approval_id)

    def _require_definition(self, request: ToolRequest) -> ToolDefinition:
        if request.name == SKILL_TOOL_NAME:
            if request.action_class not in {
                ActionClass.EXECUTE,
                ActionClass.EXTERNAL_SIDE_EFFECT,
            }:
                raise ValueError("Skill execution has an invalid action class")
            definition = ToolDefinition(
                name=SKILL_TOOL_NAME,
                action_class=request.action_class,
                resource_prefix=SKILL_TOOL_RESOURCE_PREFIX,
            )
            if not request.resource.startswith(definition.resource_prefix):
                raise ValueError("Tool resource is outside its backend-defined scope")
            return definition
        definition = self._executor.definition(request.name)
        if definition is None:
            raise LookupError(f"Unknown tool: {request.name}")
        if definition.action_class != request.action_class:
            raise ValueError("Tool action class does not match backend definition")
        if not request.resource.startswith(definition.resource_prefix):
            raise ValueError("Tool resource is outside its backend-defined scope")
        return definition

    @staticmethod
    def _runtime_permission_allows(
        scopes: list[str], action_class: ActionClass, resource: str
    ) -> bool:
        for encoded in scopes:
            action, separator, scope = encoded.partition(":")
            if not separator or action != action_class.value:
                continue
            if scope == "*" or scope == resource:
                return True
            if scope.endswith("*") and resource.startswith(scope[:-1]):
                return True
        return False

    async def _execute(
        self,
        request: ToolRequest,
        *,
        runtime_skill_catalog: Sequence[RuntimeSkillDefinition] = (),
        skill_definition: ExecutableSkillDefinition | None = None,
    ) -> ToolExecutionResult:
        if request.name == SKILL_TOOL_NAME:
            assert skill_definition is not None
            skill_request = SkillExecutionRequest(
                run_id=request.run_id,
                user_id=request.user_id,
                agent_id=(
                    UUID(request.requesting_subject_id)
                    if request.requesting_subject_type.value == "persistent_agent"
                    else None
                ),
                skill_id=skill_definition.skill_id,
                skill_key=skill_definition.key,
                skill_version=skill_definition.version,
                requesting_subject_type=request.requesting_subject_type,
                requesting_subject_id=request.requesting_subject_id,
                arguments=request.arguments["arguments"],
            )
            result = await self._skill_execution.execute(
                skill_request,
                runtime_skill_catalog=runtime_skill_catalog,
            )
            return self._skill_tool_result(request, result)
        try:
            output = await self._executor.execute(request.name, request.arguments)
        except Exception as exc:
            return ToolExecutionResult(status=ToolExecutionStatus.FAILED, error=str(exc))
        await self._append_event(
            run_id=request.run_id,
            event_type=AgentRunEventType.TOOL_EXECUTED,
            payload=self._audit_payload(request),
            commit=True,
        )
        return ToolExecutionResult(status=ToolExecutionStatus.EXECUTED, output=output)

    async def _prepare_skill_request(
        self,
        request: ToolRequest,
        runtime_skill_catalog: Sequence[RuntimeSkillDefinition],
    ) -> tuple[ToolRequest, ExecutableSkillDefinition]:
        raw_skill_key = request.arguments.get("skill_key")
        raw_arguments = request.arguments.get("arguments")
        if not isinstance(raw_skill_key, str) or not raw_skill_key.strip():
            raise SkillExecutionDeniedError("Skill key is required")
        if not isinstance(raw_arguments, dict):
            raise SkillExecutionDeniedError("Skill arguments must be a JSON object")
        skill_id, version = self._parse_skill_resource(request.resource)
        definition = await self._skill_execution.resolve_executable_skill(
            user_id=request.user_id,
            requesting_subject_type=request.requesting_subject_type,
            requesting_subject_id=request.requesting_subject_id,
            skill_key=raw_skill_key,
            skill_version=version,
            skill_id=skill_id,
            runtime_skill_catalog=runtime_skill_catalog,
        )
        if definition.action_class != request.action_class:
            raise SkillExecutionDeniedError("Skill action class does not match its manifest")
        canonical_resource = f"skill:{definition.skill_id}:v{definition.version}"
        return (
            request.model_copy(
                update={
                    "resource": canonical_resource,
                    "arguments": {
                        "skill_key": definition.key,
                        "arguments": dict(raw_arguments),
                    },
                }
            ),
            definition,
        )

    async def _require_skill_permissions(
        self, request: ToolRequest, definition: ExecutableSkillDefinition
    ) -> None:
        for encoded in definition.required_permissions:
            action, separator, resource = encoded.partition(":")
            if not separator or not resource:
                raise PermissionDeniedError("Skill required permission is invalid")
            try:
                action_class = ActionClass(action)
            except ValueError as exc:
                raise PermissionDeniedError("Skill required permission is invalid") from exc
            await self._permissions.require(
                user_id=request.user_id,
                subject_type=request.requesting_subject_type,
                subject_id=request.requesting_subject_id,
                action_class=action_class,
                resource=resource,
            )

    @staticmethod
    def _require_runtime_skill_permissions(
        scopes: list[str], definition: ExecutableSkillDefinition
    ) -> None:
        for encoded in definition.required_permissions:
            action, separator, resource = encoded.partition(":")
            if not separator or not BackendToolAuthority._runtime_scope_matches(
                scopes, action, resource
            ):
                raise PermissionDeniedError("Skill required permission is outside runtime scope")

    @staticmethod
    def _runtime_scope_matches(scopes: list[str], action: str, resource: str) -> bool:
        for encoded in scopes:
            scope_action, separator, scope = encoded.partition(":")
            if not separator or scope_action != action:
                continue
            if scope == "*" or scope == resource:
                return True
            if scope.endswith("*") and resource.startswith(scope[:-1]):
                return True
        return False

    @staticmethod
    def _parse_skill_resource(resource: str) -> tuple[UUID | None, int | None]:
        if not resource.startswith(SKILL_TOOL_RESOURCE_PREFIX):
            return None, None
        value = resource.removeprefix(SKILL_TOOL_RESOURCE_PREFIX)
        if ":v" not in value:
            return None, None
        raw_id, raw_version = value.rsplit(":v", 1)
        try:
            version = int(raw_version)
        except ValueError:
            return None, None
        try:
            return UUID(raw_id), version
        except ValueError:
            return None, version

    @staticmethod
    def _skill_tool_result(
        request: ToolRequest, result: SkillExecutionResult
    ) -> ToolExecutionResult:
        if result.status == SkillExecutionStatus.SUCCEEDED:
            return ToolExecutionResult(
                status=ToolExecutionStatus.EXECUTED,
                output=result.output,
            )
        if result.status == SkillExecutionStatus.DENIED:
            return ToolExecutionResult(
                status=ToolExecutionStatus.DENIED,
                error=result.error or "Skill execution denied",
            )
        if result.status == SkillExecutionStatus.CANCELLED:
            return ToolExecutionResult(
                status=ToolExecutionStatus.FAILED,
                error="Skill execution cancelled",
            )
        return ToolExecutionResult(
            status=ToolExecutionStatus.FAILED,
            error=result.error or "Skill execution failed",
        )

    async def _append_event(
        self,
        *,
        run_id: UUID,
        event_type: AgentRunEventType,
        payload: dict[str, object],
        commit: bool = False,
    ) -> None:
        await self._runs.append_event(
            AgentRunEventCreate(run_id=run_id, event_type=event_type, payload=payload)
        )
        if commit:
            await self._session.commit()

    @staticmethod
    def _audit_payload(request: ToolRequest) -> dict[str, object]:
        return {
            "tool": request.name,
            "action_class": request.action_class.value,
            "resource": request.resource,
            "subject_type": request.requesting_subject_type.value,
            "subject_id": request.requesting_subject_id,
        }
