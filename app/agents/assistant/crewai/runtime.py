"""CrewAI execution adapters, including non-blocking HITL tool approval."""

from pathlib import Path
import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
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
from models.agent_factory import CrewDefinition
from models.runtime import RuntimeAgentKind
from agents.assistant.crewai.skill_runtime import SkillRuntimeResolver
from agents.assistant.crewai.tool_runtime import ToolCallBudget, ToolRuntimeResolver
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

    def __init__(
        self, *, factory: DynamicCrewAIFactory | None = None,
        skill_resolver: SkillRuntimeResolver | None = None,
        tool_resolver: ToolRuntimeResolver | None = None,
    ) -> None:
        self._factory = factory or DynamicCrewAIFactory()
        self._skill_resolver = skill_resolver
        self._tool_resolver = tool_resolver

    async def run(
        self,
        context: AgentRuntimeContext,
        request: AgentRuntimeRequest,
    ) -> RuntimeStep:
        tool_budget = ToolCallBudget(context.budgets.max_tool_calls - context.usage.tool_calls)
        resolved_tools = (
            await self._tool_resolver.resolve(
                run_id=context.run_id,
                user_id=context.user_id,
                subject_type=context.active_agent.subject_type,
                subject_id=context.active_agent.subject_id,
                skills=context.active_skills,
                budget=tool_budget,
                runtime_permission_scopes=next(
                    (item.permissions for item in context.temporary_subagents
                     if item.identity == context.active_agent),
                    [],
                ),
            ) if self._tool_resolver is not None else []
        )
        try:
            if self._skill_resolver is None:
                if any(skill.storage_uri for skill in context.active_skills):
                    raise RuntimeError("Native Skill resolver is not configured")
                crew = self._factory.build(context, request, resolved_tools=resolved_tools)
                output = await self._kickoff(crew)
            else:
                agent_id = (
                    UUID(context.active_agent.subject_id)
                    if context.active_agent.kind in {RuntimeAgentKind.PRIMARY, RuntimeAgentKind.USER}
                    else None
                )
                async with self._skill_resolver.materialize_for_agent(
                    user_id=context.user_id,
                    agent_id=agent_id,
                    definitions=context.active_skills,
                    run_id=context.run_id,
                ) as native_skills:
                    crew = self._factory.build(
                        context, request, native_skills=native_skills,
                        resolved_tools=resolved_tools,
                    )
                    output = await self._kickoff(crew)
        finally:
            context.usage.tool_calls += tool_budget.used
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

    @staticmethod
    async def _kickoff(crew):
        """Drain CrewAI worker threads before materialized files are removed."""
        pending = asyncio.create_task(crew.akickoff())
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            try:
                await pending
            finally:
                raise

    @asynccontextmanager
    async def prepared_crew(
        self, definition: CrewDefinition, *, user_id: int, run_id: UUID | None = None,
    ) -> AsyncIterator[Any]:
        """Keep materialized native skills alive for the caller's CrewAI kickoff."""
        if definition.user_id != user_id:
            raise PermissionError("Crew runtime definition is outside this user")
        if self._skill_resolver is None:
            if any(skill.storage_uri for agent in definition.agents for skill in agent.active_skills):
                raise RuntimeError("Native Skill resolver is not configured")
            yield self._factory.build_from_definition(definition)
            return
        agents = [
            (UUID(agent.identity.subject_id), agent.active_skills)
            for agent in definition.agents
        ]
        async with self._skill_resolver.materialize_for_crew(
            user_id=user_id, agents=agents, run_id=run_id,
        ) as native_skills:
            resolved_tools = {}
            tool_budget = ToolCallBudget(definition.budgets.max_tool_calls)
            if run_id is not None and self._tool_resolver is not None:
                for agent in definition.agents:
                    agent_id = UUID(agent.identity.subject_id)
                    resolved_tools[agent_id] = await self._tool_resolver.resolve(
                        run_id=run_id,
                        user_id=user_id,
                        subject_type=agent.identity.subject_type,
                        subject_id=agent.identity.subject_id,
                        skills=agent.active_skills,
                        budget=tool_budget,
                    )
            yield self._factory.build_from_definition(
                definition, native_skills_by_agent=native_skills,
                resolved_tools_by_agent=resolved_tools,
            )
