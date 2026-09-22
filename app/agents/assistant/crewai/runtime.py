"""CrewAI execution adapters, including non-blocking HITL tool approval."""

from pathlib import Path
from typing import Any, ClassVar
from uuid import UUID

from pydantic import PrivateAttr

from crewai.flow import (
    Flow,
    HumanFeedbackPending,
    HumanFeedbackProvider,
    PendingFeedbackContext,
    human_feedback,
    listen,
    start,
)
from crewai.flow.persistence import SQLiteFlowPersistence
from agents.assistant.crewai.factory import DynamicCrewAIFactory
from models.runtime import (
    AgentRuntimeContext,
    AgentRuntimeRequest,
    RuntimeStep,
    RuntimeSkillDefinition,
    ToolApprovalFlowState,
)
from models.tool import ToolExecutionResult, ToolExecutionStatus, ToolRequest
from services.tool_authority import BackendToolAuthority


class BackendApprovalProvider(HumanFeedbackProvider):
    """Pause CrewAI and expose backend approval identifiers to the caller."""

    def request_feedback(
        self,
        context: PendingFeedbackContext,
        flow: Flow[Any],
    ) -> str:
        output = context.method_output
        if isinstance(output, ToolExecutionResult):
            result = output
        else:
            result = ToolExecutionResult.model_validate(output)
        if result.status != ToolExecutionStatus.WAITING_APPROVAL:
            return ""
        raise HumanFeedbackPending(
            context=context,
            callback_info={
                "flow_id": context.flow_id,
                "approval_id": str(result.approval_id),
                "run_id": str(flow.state.run_id),
            },
        )


class ToolApprovalFlow(Flow[ToolApprovalFlowState]):
    """One CrewAI HITL checkpoint for exactly one backend tool action."""

    _skip_auto_memory: ClassVar[bool] = True
    _authority: BackendToolAuthority = PrivateAttr()

    def __init__(
        self,
        *,
        authority: BackendToolAuthority,
        request: ToolRequest | None = None,
        runtime_permission_scopes: list[str] | None = None,
        runtime_skill_catalog: list[RuntimeSkillDefinition] | None = None,
        persistence: SQLiteFlowPersistence | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            persistence=persistence,
            tracing=False,
            suppress_flow_events=True,
            **kwargs,
        )
        self._authority = authority
        if request is not None:
            self.state.request = request.model_dump(mode="json")
            self.state.run_id = str(request.run_id)
            self.state.user_id = request.user_id
            self.state.runtime_permission_scopes = runtime_permission_scopes or []
            self.state.runtime_skill_catalog = runtime_skill_catalog or []

    @start()
    @human_feedback(
        message="Approve this concrete tool action?",
        provider=BackendApprovalProvider(),
    )
    async def request_approval(self) -> ToolExecutionResult:
        request = ToolRequest.model_validate(self.state.request)
        if self.state.runtime_skill_catalog:
            result = await self._authority.request(
                request,
                flow_id=self.flow_id,
                runtime_permission_scopes=self.state.runtime_permission_scopes,
                runtime_skill_catalog=self.state.runtime_skill_catalog,
            )
        else:
            result = await self._authority.request(
                request,
                flow_id=self.flow_id,
                runtime_permission_scopes=self.state.runtime_permission_scopes,
            )
        if result.approval_id is not None:
            self.state.approval_id = str(result.approval_id)
        return result

    @listen(request_approval)
    async def continue_after_decision(self) -> ToolExecutionResult:
        if self.state.approval_id is None:
            result = self.last_human_feedback
            if result is None:
                return ToolExecutionResult(
                    status=ToolExecutionStatus.FAILED,
                    error="CrewAI approval flow completed without a result",
                )
            return ToolExecutionResult.model_validate(result.output)
        if self.state.runtime_skill_catalog:
            return await self._authority.execute_approved(
                user_id=self.state.user_id,
                approval_id=UUID(self.state.approval_id),
                runtime_permission_scopes=self.state.runtime_permission_scopes,
                runtime_skill_catalog=self.state.runtime_skill_catalog,
            )
        return await self._authority.execute_approved(
            user_id=self.state.user_id,
            approval_id=UUID(self.state.approval_id),
            runtime_permission_scopes=self.state.runtime_permission_scopes,
        )


class CrewAIToolApprovalRuntime:
    """Start and resume CrewAI's persisted async human-feedback flow."""

    def __init__(
        self,
        *,
        authority: BackendToolAuthority,
        persistence_path: Path,
    ) -> None:
        persistence_path.parent.mkdir(parents=True, exist_ok=True)
        self._authority = authority
        self._persistence = SQLiteFlowPersistence(str(persistence_path))

    async def begin(
        self,
        request: ToolRequest,
        *,
        runtime_permission_scopes: list[str] | None = None,
        runtime_skill_catalog: list[RuntimeSkillDefinition] | None = None,
    ) -> ToolExecutionResult:
        flow = ToolApprovalFlow(
            authority=self._authority,
            request=request,
            runtime_permission_scopes=runtime_permission_scopes,
            runtime_skill_catalog=runtime_skill_catalog,
            persistence=self._persistence,
        )
        result = await flow.kickoff_async()
        if isinstance(result, HumanFeedbackPending):
            approval_id = result.callback_info.get("approval_id")
            return ToolExecutionResult(
                status=ToolExecutionStatus.WAITING_APPROVAL,
                approval_id=UUID(str(approval_id)),
                flow_id=result.context.flow_id,
            )
        return ToolExecutionResult.model_validate(result)

    async def resume(self, *, user_id: int, flow_id: str) -> ToolExecutionResult:
        flow = ToolApprovalFlow.from_pending(
            flow_id,
            self._persistence,
            authority=self._authority,
        )
        approval_id = flow.state.approval_id
        if approval_id is None:
            raise ValueError("CrewAI flow has no backend approval id")
        approval = await self._authority.get_approval(
            user_id=user_id,
            approval_id=UUID(approval_id),
        )
        result = await flow.resume_async(approval.status.value)
        if isinstance(result, HumanFeedbackPending):
            raise RuntimeError("One-action approval flow unexpectedly paused twice")
        return ToolExecutionResult.model_validate(result)


class DynamicCrewAIRuntime:
    """Execute a dynamically built CrewAI crew and map it to RuntimeStep."""

    def __init__(self, *, factory: DynamicCrewAIFactory | None = None) -> None:
        self._factory = factory or DynamicCrewAIFactory()

    async def run(
        self,
        context: AgentRuntimeContext,
        request: AgentRuntimeRequest,
    ) -> RuntimeStep:
        crew = self._factory.build(context, request)
        output = await crew.kickoff_async()
        if output.pydantic is not None:
            step = RuntimeStep.model_validate(output.pydantic)
        elif output.json_dict is not None:
            step = RuntimeStep.model_validate(output.json_dict)
        else:
            step = RuntimeStep.model_validate_json(output.raw)

        usage = getattr(output, "token_usage", None)
        total_tokens = 0
        if usage is not None:
            if isinstance(usage, dict):
                total_tokens = int(usage.get("total_tokens", 0) or 0)
            else:
                total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        if total_tokens:
            step = step.model_copy(update={"tokens_used": total_tokens})
        return step
