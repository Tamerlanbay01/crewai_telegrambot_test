"""Backend authority gate for every CrewAI-requested tool action."""

from datetime import datetime, timezone
from typing import Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from models.approval import ApprovalStatus
from models.agent_run import AgentRunEventCreate, AgentRunEventType, AgentRunStatus
from models.permission import ActionClass
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
    def __init__(self, session: AsyncSession, *, executor: ToolExecutor | None = None):
        self._session = session
        self._executor = executor or FakeToolExecutor()
        self._permissions = PermissionService(session)
        self._approvals = ApprovalService(session)
        self._approval_repository = ApprovalRepository(session)
        self._runs = AgentRunRepository(session)
        self._run_service = AgentRunService(session)

    async def request(
        self,
        request: ToolRequest,
        *,
        runtime_permission_scopes: list[str] | None = None,
        flow_id: str | None = None,
    ) -> ToolExecutionResult:
        run = await self._run_service.get_run(user_id=request.user_id, run_id=request.run_id)
        if run.status != AgentRunStatus.RUNNING:
            raise ValueError("Tools can only be requested by a running run")
        definition = self._require_definition(request)
        await self._append_event(
            run_id=request.run_id,
            event_type=AgentRunEventType.TOOL_REQUESTED,
            payload=self._audit_payload(request),
        )
        try:
            if request.requesting_subject_type.value == "temporary_subagent":
                if not self._runtime_permission_allows(
                    runtime_permission_scopes or [],
                    definition.action_class,
                    request.resource,
                ):
                    raise PermissionDeniedError(
                        "Temporary subagent permission is outside its inherited runtime scope"
                    )
            else:
                await self._permissions.require(
                    user_id=request.user_id,
                    subject_type=request.requesting_subject_type,
                    subject_id=request.requesting_subject_id,
                    action_class=definition.action_class,
                    resource=request.resource,
                )
        except PermissionDeniedError as exc:
            await self._append_event(
                run_id=request.run_id,
                event_type=AgentRunEventType.PERMISSION_DENIED,
                payload={**self._audit_payload(request), "error": str(exc)},
                commit=True,
            )
            return ToolExecutionResult(
                status=ToolExecutionStatus.DENIED,
                error=str(exc),
            )

        if definition.action_class != ActionClass.READ:
            approval = await self._approvals.request(
                user_id=request.user_id,
                run_id=request.run_id,
                flow_id=flow_id,
                requesting_subject_type=request.requesting_subject_type,
                requesting_subject_id=request.requesting_subject_id,
                action_name=request.name,
                action_class=definition.action_class,
                resource=request.resource,
                arguments=request.arguments,
            )
            return ToolExecutionResult(
                status=ToolExecutionStatus.WAITING_APPROVAL,
                approval_id=approval.id,
            )

        return await self._execute(request)

    async def execute_approved(
        self,
        *,
        user_id: int,
        approval_id: UUID,
        runtime_permission_scopes: list[str] | None = None,
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
        self._require_definition(request)
        if approval.requesting_subject_type.value == "temporary_subagent":
            if not self._runtime_permission_allows(
                runtime_permission_scopes or [],
                approval.action_class,
                approval.resource,
            ):
                raise PermissionDeniedError(
                    "Temporary subagent permission is outside its inherited runtime scope"
                )
        else:
            await self._permissions.require(
                user_id=approval.user_id,
                subject_type=approval.requesting_subject_type,
                subject_id=approval.requesting_subject_id,
                action_class=approval.action_class,
                resource=approval.resource,
            )
        result = await self._execute(request)
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

    async def _execute(self, request: ToolRequest) -> ToolExecutionResult:
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
