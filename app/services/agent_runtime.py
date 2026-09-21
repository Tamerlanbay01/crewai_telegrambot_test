"""Central backend coordinator for persisted, authorized agent execution."""

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from agents.protocols import AgentExecutionRuntime, ToolApprovalRuntime
from core.config import config
from models.agent import Agent, AgentKind, AgentStatus
from models.agent_prompt import AgentPromptVersion
from models.agent_run import AgentRunEventCreate, AgentRunEventType, AgentRunStatus
from models.permission import PermissionSubjectType
from models.runtime import (
    AgentRuntimeContext,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    AgentRuntimeStatus,
    DelegationDecision,
    DelegationRecord,
    DelegationType,
    PendingApprovalCheckpoint,
    RuntimeAgentDefinition,
    RuntimeAgentIdentity,
    RuntimeAgentKind,
    RuntimeBudgets,
    RuntimeChatMessage,
    RuntimeSkillDefinition,
    RuntimeTemporarySubagent,
    TemporarySubagentStatus,
)
from models.tool import ToolExecutionStatus, ToolRequest
from repositories.agent import AgentRepository
from repositories.agent_connection import AgentConnectionRepository
from repositories.agent_prompt import AgentPromptRepository
from repositories.agent_run import AgentRunRepository
from repositories.message import MessageRepository
from services.agent_run import AgentRunService
from services.memory import MemoryService
from services.skill import SkillService
from services.system_agent import SystemAgentService
from services.tool_authority import BackendToolAuthority, ToolExecutor
from services.permission import PermissionService


class RuntimeBudgetExceededError(RuntimeError):
    pass


class AgentRuntimeService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        runtime: AgentExecutionRuntime,
        tool_executor: ToolExecutor | None = None,
        approval_runtime: ToolApprovalRuntime | None = None,
        approval_persistence_path: Path | None = None,
        budgets: RuntimeBudgets | None = None,
    ):
        self._session = session
        self._runtime = runtime
        self._default_budgets = budgets or RuntimeBudgets()
        self._agents = AgentRepository(session)
        self._prompts = AgentPromptRepository(session)
        self._connections = AgentConnectionRepository(session)
        self._runs = AgentRunRepository(session)
        self._run_service = AgentRunService(session)
        self._messages = MessageRepository(session)
        self._system_agents = SystemAgentService(session)
        self._skills = SkillService(session)
        self._memory = MemoryService(session)
        self._tool_authority = BackendToolAuthority(session, executor=tool_executor)
        self._approval_runtime = approval_runtime
        self._approval_persistence_path = (
            approval_persistence_path or Path(".crewai") / "flow_states.db"
        )
        self._permissions = PermissionService(session)

    async def execute(self, request: AgentRuntimeRequest) -> AgentRuntimeResult:
        try:
            run = await self._run_service.get_run(user_id=request.user_id, run_id=request.run_id)
            if run.starting_agent_id != request.agent_id:
                raise ValueError("Runtime request agent does not match the persisted run")
            if request.chat_id is not None and run.chat_id != request.chat_id:
                raise ValueError("Runtime request chat does not match the persisted run")
            if run.status == AgentRunStatus.CREATED:
                run = await self._run_service.start(user_id=request.user_id, run_id=run.id)
            elif run.status != AgentRunStatus.RUNNING:
                raise ValueError(f"Run cannot execute from status: {run.status.value}")

            if run.checkpoint:
                context = AgentRuntimeContext.model_validate(run.checkpoint)
                if context.pending_approval is not None:
                    raise ValueError("Run has a pending approval and must be resumed")
            else:
                context = await self._build_context(
                    request,
                    run.prompt_version_id,
                    chat_id=run.chat_id,
                    current_message_id=run.message_id,
                )
                await self._event(
                    run_id=run.id,
                    event_type=AgentRunEventType.RUNTIME_STARTED,
                    payload={"starting_agent_id": str(request.agent_id)},
                )
                await self._persist(context)
            return await self._continue(context, request)
        except Exception as exc:
            return await self._fail(request.user_id, request.run_id, str(exc))

    async def resume(self, *, user_id: int, run_id: UUID) -> AgentRuntimeResult:
        run = await self._run_service.get_run(user_id=user_id, run_id=run_id)
        if not run.checkpoint:
            raise ValueError("Run has no persisted runtime checkpoint")
        context = AgentRuntimeContext.model_validate(run.checkpoint)
        pending = context.pending_approval
        if pending is None:
            raise ValueError("Run has no pending approval")
        try:
            tool_result = await self._get_approval_runtime().resume(
                user_id=user_id,
                flow_id=pending.flow_id,
            )
            context.pending_approval = None
            context.current_input = self._tool_feedback(
                tool_result.status, tool_result.output, tool_result.error
            )
            await self._persist(context)
            request = AgentRuntimeRequest(
                run_id=run_id,
                user_id=user_id,
                agent_id=context.starting_agent_id,
                message=context.original_message,
            )
            return await self._continue(context, request)
        except Exception as exc:
            return await self._fail(user_id, run_id, str(exc), context=context)

    async def _build_context(
        self,
        request: AgentRuntimeRequest,
        prompt_version_id: UUID,
        *,
        chat_id: UUID | None = None,
        current_message_id: UUID | None = None,
    ) -> AgentRuntimeContext:
        starting_agent = await self._require_active_agent(request.user_id, request.agent_id)
        starting_prompt = await self._prompts.get_by_id(prompt_version_id)
        if starting_prompt is None or starting_prompt.agent_id != starting_agent.id:
            raise LookupError("Pinned starting prompt version not found")
        starting_skills = await self._skills.resolve_agent_skills(
            user_id=request.user_id,
            agent_id=starting_agent.id,
        )
        starting_definition = self._persistent_definition(
            starting_agent,
            starting_prompt,
            active_skills=starting_skills,
        )

        connected: list[RuntimeAgentDefinition] = []
        for connection in await self._connections.list_children(starting_agent.id):
            agent = await self._agents.get_by_id(connection.child_agent_id)
            if agent is None or agent.user_id != request.user_id or agent.status != AgentStatus.ACTIVE:
                continue
            prompt = await self._prompts.get_latest(agent.id)
            if prompt is not None:
                connected_skills = await self._skills.resolve_agent_skills(
                    user_id=request.user_id,
                    agent_id=agent.id,
                )
                connected.append(
                    self._persistent_definition(
                        agent,
                        prompt,
                        active_skills=connected_skills,
                    )
                )

        system_definitions: list[RuntimeAgentDefinition] = []
        for system in await self._system_agents.list_available_for_user(user_id=request.user_id):
            system_skills = await self._skills.resolve_system_skills(
                allowed_keys=system.allowed_skills,
                default_keys=system.default_skills,
            )
            system_definitions.append(
                RuntimeAgentDefinition(
                identity=RuntimeAgentIdentity(
                    subject_type=PermissionSubjectType.SYSTEM_AGENT,
                    subject_id=system.key,
                    kind=RuntimeAgentKind.SYSTEM,
                    name=system.name,
                ),
                role=system.role,
                goal=system.goal,
                backstory=system.backstory,
                custom_instructions=system.custom_instructions,
                active_skills=system_skills,
                allowed_skill_keys=list(system.allowed_skills),
                default_skill_keys=list(system.default_skills),
                )
            )
        chat_context: list[RuntimeChatMessage] = []
        history_chat_id = request.chat_id or chat_id
        if history_chat_id is not None:
            chat_context = [
                RuntimeChatMessage(role=message.role, content=message.content)
                for message in await self._messages.list_recent(
                    history_chat_id,
                    limit=config.assistant_history_limit,
                    exclude_message_id=current_message_id,
                )
            ]
        return AgentRuntimeContext(
            run_id=request.run_id,
            user_id=request.user_id,
            starting_agent_id=request.agent_id,
            starting_prompt=starting_prompt,
            starting_agent=starting_definition,
            active_agent=starting_definition.identity,
            chat_context=chat_context,
            connected_persistent_agents=connected,
            available_system_agents=system_definitions,
            active_skills=starting_definition.active_skills,
            memory=await self._memory.build_runtime_context(
                user_id=request.user_id,
                agent_id=starting_agent.id,
                run_id=request.run_id,
            ),
            budgets=self._default_budgets.model_copy(deep=True),
            original_message=request.message,
            current_input=request.message,
            runtime_started_at=datetime.now(timezone.utc),
        )

    async def _continue(
        self, context: AgentRuntimeContext, request: AgentRuntimeRequest
    ) -> AgentRuntimeResult:
        try:
            while True:
                self._check_time(context)
                self._consume(context, "llm_calls", context.budgets.max_llm_calls)
                iterations = context.agent_iterations.get(context.active_agent.subject_id, 0)
                if iterations >= context.budgets.max_agent_iterations:
                    raise RuntimeBudgetExceededError("Runtime budget exceeded: max_agent_iterations")
                context.agent_iterations[context.active_agent.subject_id] = iterations + 1
                await self._refresh_active_resources(context)
                await self._persist(context)
                step = await self._runtime.run(context, request)
                context.usage.tokens += step.tokens_used
                context.usage.retries += step.retries
                self._check_total(context, "tokens", context.budgets.max_tokens)
                self._check_total(context, "retries", context.budgets.max_retries)

                if step.tool_intent is not None:
                    self._consume(context, "tool_calls", context.budgets.max_tool_calls)
                    tool_request = ToolRequest(
                        run_id=context.run_id,
                        user_id=context.user_id,
                        requesting_subject_type=context.active_agent.subject_type,
                        requesting_subject_id=context.active_agent.subject_id,
                        name=step.tool_intent.name,
                        action_class=step.tool_intent.action_class,
                        resource=step.tool_intent.resource,
                        arguments=step.tool_intent.arguments,
                    )
                    await self._persist(context)
                    runtime_scopes = None
                    if context.active_agent.kind == RuntimeAgentKind.TEMPORARY:
                        runtime_scopes = next(
                            (
                                item.permissions
                                for item in context.temporary_subagents
                                if item.identity.subject_id == context.active_agent.subject_id
                            ),
                            [],
                        )
                    if step.tool_intent.action_class.value == "read":
                        tool_result = await self._tool_authority.request(
                            tool_request, runtime_permission_scopes=runtime_scopes
                        )
                    else:
                        tool_result = await self._get_approval_runtime().begin(
                            tool_request,
                            runtime_permission_scopes=runtime_scopes,
                        )
                    if tool_result.status == ToolExecutionStatus.WAITING_APPROVAL:
                        assert tool_result.approval_id is not None
                        context.pending_approval = PendingApprovalCheckpoint(
                            approval_id=tool_result.approval_id,
                            flow_id=tool_result.flow_id or "",
                            tool_request=tool_request,
                        )
                        await self._persist(context)
                        return AgentRuntimeResult(
                            status=AgentRuntimeStatus.WAITING_APPROVAL,
                            metadata={"approval_id": str(tool_result.approval_id)},
                            context=context,
                        )
                    context.current_input = self._tool_feedback(
                        tool_result.status, tool_result.output, tool_result.error
                    )
                    await self._persist(context)
                    continue

                if step.decision.type == DelegationType.RESPOND:
                    if context.delegation_stack:
                        await self._return_to_parent(context, step.content or "")
                        continue
                    return await self._complete(context, step.content or "")

                await self._delegate(context, step.decision)
        except RuntimeBudgetExceededError as exc:
            await self._event(
                run_id=context.run_id,
                event_type=AgentRunEventType.BUDGET_EXCEEDED,
                payload={"error": str(exc)},
            )
            return await self._fail(context.user_id, context.run_id, str(exc), context=context)
        except Exception as exc:
            return await self._fail(context.user_id, context.run_id, str(exc), context=context)

    async def _delegate(
        self, context: AgentRuntimeContext, decision: DelegationDecision
    ) -> None:
        source = context.active_agent
        if decision.type in {
            DelegationType.DELEGATE_USER_AGENT,
            DelegationType.DELEGATE_SYSTEM_AGENT,
        } and source.kind != RuntimeAgentKind.PRIMARY:
            raise PermissionError("Only the primary agent can delegate to persistent workers")
        self._consume(context, "delegations", context.budgets.max_delegations)
        task_summary = (decision.task_summary or "").strip()
        if not task_summary:
            raise ValueError("Delegation task summary cannot be empty")

        if decision.type == DelegationType.DELEGATE_USER_AGENT:
            target = self._find_definition(context.connected_persistent_agents, decision.target_id)
            if target is None:
                raise PermissionError("Target user agent is not active and connected")
        elif decision.type == DelegationType.DELEGATE_SYSTEM_AGENT:
            target = self._find_definition(context.available_system_agents, decision.target_id)
            if target is None:
                raise PermissionError("Target system agent is not available")
        elif decision.type == DelegationType.CREATE_TEMPORARY_SUBAGENT:
            target = await self._create_temporary(context, decision)
        else:
            raise ValueError(f"Unsupported delegation type: {decision.type.value}")

        context.delegation_stack.append(source)
        context.active_agent = target.identity
        context.active_skills = list(target.active_skills)
        context.current_input = task_summary
        record = DelegationRecord(
            source=source,
            target=target.identity,
            delegation_type=decision.type,
            task_summary=task_summary,
            timestamp=datetime.now(timezone.utc),
        )
        context.delegation_state.append(record)
        await self._refresh_active_resources(context)
        await self._event(
            run_id=context.run_id,
            event_type=AgentRunEventType.DELEGATION_REQUESTED,
            payload=self._delegation_payload(record),
        )
        await self._persist(context)

    async def _create_temporary(
        self, context: AgentRuntimeContext, decision: DelegationDecision
    ) -> RuntimeAgentDefinition:
        parent = self._definition_for(context, context.active_agent)
        if parent is None or not parent.can_spawn_subagents:
            raise PermissionError("Active agent cannot spawn temporary subagents")
        if context.active_agent.kind == RuntimeAgentKind.TEMPORARY:
            raise PermissionError("Temporary subagents cannot spawn subagents")
        self._consume(context, "subagents", context.budgets.max_subagents)
        children = [
            item
            for item in context.temporary_subagents
            if item.parent.subject_id == context.active_agent.subject_id
        ]
        if len(children) >= context.budgets.max_subagents_per_parent:
            raise RuntimeBudgetExceededError("Temporary subagent per-parent limit exceeded")
        depth = 1
        if depth > context.budgets.max_subagent_depth:
            raise RuntimeBudgetExceededError("Temporary subagent depth limit exceeded")
        parent_scopes = set(
            await self._permissions.list_allowed_scopes(
                user_id=context.user_id,
                subject_type=context.active_agent.subject_type,
                subject_id=context.active_agent.subject_id,
            )
        )
        requested_scopes = set(decision.requested_permissions)
        if not requested_scopes.issubset(parent_scopes):
            raise PermissionError("Temporary subagent permissions exceed parent permissions")
        name = (decision.temporary_name or "Temporary Specialist").strip()
        role = (decision.temporary_role or "Temporary Specialist").strip()
        goal = (decision.temporary_goal or decision.task_summary or "Complete the task").strip()
        identity = RuntimeAgentIdentity(
            subject_type=PermissionSubjectType.TEMPORARY_SUBAGENT,
            subject_id=str(uuid4()),
            kind=RuntimeAgentKind.TEMPORARY,
            name=name,
        )
        temporary = RuntimeTemporarySubagent(
            identity=identity,
            parent=context.active_agent,
            role=role,
            goal=goal,
            backstory=decision.temporary_backstory,
            depth=depth,
            status=TemporarySubagentStatus.RUNNING,
            permissions=decision.requested_permissions,
            active_skills=list(parent.active_skills),
            created_at=datetime.now(timezone.utc),
        )
        context.temporary_subagents.append(temporary)
        await self._event(
            run_id=context.run_id,
            event_type=AgentRunEventType.TEMPORARY_SUBAGENT_CREATED,
            payload={
                "temporary_agent_id": identity.subject_id,
                "parent_id": context.active_agent.subject_id,
                "depth": depth,
            },
        )
        return RuntimeAgentDefinition(
            identity=identity,
            role=role,
            goal=goal,
            backstory=decision.temporary_backstory,
            can_spawn_subagents=False,
            active_skills=list(parent.active_skills),
        )

    async def _return_to_parent(self, context: AgentRuntimeContext, content: str) -> None:
        completed_identity = context.active_agent
        parent = context.delegation_stack.pop()
        for record in reversed(context.delegation_state):
            if record.target.subject_id == completed_identity.subject_id and record.success is None:
                record.success = True
                await self._event(
                    run_id=context.run_id,
                    event_type=AgentRunEventType.DELEGATION_COMPLETED,
                    payload=self._delegation_payload(record),
                )
                break
        for temporary in context.temporary_subagents:
            if temporary.identity.subject_id == completed_identity.subject_id:
                temporary.status = TemporarySubagentStatus.ARCHIVED
                temporary.completed_at = datetime.now(timezone.utc)
        context.active_agent = parent
        parent_definition = self._definition_for(context, parent)
        context.active_skills = (
            list(parent_definition.active_skills) if parent_definition is not None else []
        )
        context.current_input = content
        await self._refresh_active_resources(context)
        await self._persist(context)

    async def _complete(
        self, context: AgentRuntimeContext, content: str
    ) -> AgentRuntimeResult:
        if not content.strip():
            raise ValueError("Runtime returned an empty final response")
        await self._event(
            run_id=context.run_id,
            event_type=AgentRunEventType.RUNTIME_COMPLETED,
            payload={"content_length": len(content)},
        )
        await self._persist(context)
        await self._run_service.complete(
            user_id=context.user_id,
            run_id=context.run_id,
            result_metadata={"content": content},
            usage=context.usage.model_dump(mode="json"),
        )
        return AgentRuntimeResult(
            status=AgentRuntimeStatus.COMPLETED,
            content=content,
            context=context,
        )

    async def _fail(
        self,
        user_id: int,
        run_id: UUID,
        error: str,
        *,
        context: AgentRuntimeContext | None = None,
    ) -> AgentRuntimeResult:
        try:
            run = await self._run_service.get_run(user_id=user_id, run_id=run_id)
            if context is not None:
                await self._persist(context)
            if run.status == AgentRunStatus.RUNNING:
                await self._event(
                    run_id=run_id,
                    event_type=AgentRunEventType.RUNTIME_FAILED,
                    payload={"error": error},
                )
                await self._run_service.fail(user_id=user_id, run_id=run_id, error=error)
        except Exception:
            pass
        return AgentRuntimeResult(
            status=AgentRuntimeStatus.FAILED,
            error=error,
            context=context,
        )

    async def _persist(self, context: AgentRuntimeContext) -> None:
        updated = await self._runs.update_runtime_state(
            run_id=context.run_id,
            usage=context.usage.model_dump(mode="json"),
            checkpoint=context.model_dump(mode="json"),
        )
        if updated is None:
            raise LookupError(f"Agent run not found: {context.run_id}")
        await self._session.commit()

    async def _event(
        self,
        *,
        run_id: UUID,
        event_type: AgentRunEventType,
        payload: dict[str, object],
    ) -> None:
        json.dumps(payload)
        await self._runs.append_event(
            AgentRunEventCreate(run_id=run_id, event_type=event_type, payload=payload)
        )

    async def _require_active_agent(self, user_id: int, agent_id: UUID) -> Agent:
        agent = await self._agents.get_by_id(agent_id)
        if agent is None or agent.user_id != user_id:
            raise LookupError(f"Agent not found: {agent_id}")
        if agent.status != AgentStatus.ACTIVE:
            raise ValueError("Starting agent is not active")
        return agent

    @staticmethod
    def _persistent_definition(
        agent: Agent,
        prompt: AgentPromptVersion,
        *,
        active_skills: list[RuntimeSkillDefinition] | None = None,
    ) -> RuntimeAgentDefinition:
        return RuntimeAgentDefinition(
            identity=RuntimeAgentIdentity(
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id=str(agent.id),
                kind=(RuntimeAgentKind.PRIMARY if agent.kind == AgentKind.PRIMARY else RuntimeAgentKind.USER),
                name=agent.name,
            ),
            role=prompt.role,
            goal=prompt.goal,
            backstory=prompt.backstory,
            custom_instructions=prompt.custom_instructions,
            prompt_version_id=prompt.id,
            can_spawn_subagents=agent.can_spawn_subagents,
            active_skills=active_skills or [],
        )

    async def _refresh_active_resources(self, context: AgentRuntimeContext) -> None:
        definition = self._definition_for(context, context.active_agent)
        if definition is None:
            raise LookupError(
                f"Active runtime agent definition not found: {context.active_agent.subject_id}"
            )

        memory_agent_id: UUID | None = None
        if context.active_agent.kind in {RuntimeAgentKind.PRIMARY, RuntimeAgentKind.USER}:
            memory_agent_id = UUID(context.active_agent.subject_id)
            definition.active_skills = await self._skills.resolve_agent_skills(
                user_id=context.user_id,
                agent_id=memory_agent_id,
            )
        elif context.active_agent.kind == RuntimeAgentKind.SYSTEM:
            definition.active_skills = await self._skills.resolve_system_skills(
                allowed_keys=definition.allowed_skill_keys,
                default_keys=definition.default_skill_keys,
            )
        else:
            temporary = next(
                (
                    item
                    for item in context.temporary_subagents
                    if item.identity.subject_id == context.active_agent.subject_id
                ),
                None,
            )
            if temporary is None:
                raise LookupError(
                    f"Temporary subagent not found: {context.active_agent.subject_id}"
                )
            definition.active_skills = list(temporary.active_skills)

        context.active_skills = list(definition.active_skills)
        context.memory = await self._memory.build_runtime_context(
            user_id=context.user_id,
            agent_id=memory_agent_id,
            run_id=context.run_id,
        )

    @staticmethod
    def _find_definition(
        definitions: list[RuntimeAgentDefinition], target_id: str | None
    ) -> RuntimeAgentDefinition | None:
        return next(
            (
                definition
                for definition in definitions
                if definition.identity.subject_id == target_id
            ),
            None,
        )

    @staticmethod
    def _definition_for(
        context: AgentRuntimeContext, identity: RuntimeAgentIdentity
    ) -> RuntimeAgentDefinition | None:
        definitions = [
            context.starting_agent,
            *context.connected_persistent_agents,
            *context.available_system_agents,
        ]
        found = AgentRuntimeService._find_definition(definitions, identity.subject_id)
        if found is not None:
            return found
        for temporary in context.temporary_subagents:
            if temporary.identity.subject_id == identity.subject_id:
                return RuntimeAgentDefinition(
                    identity=temporary.identity,
                    role=temporary.role,
                    goal=temporary.goal,
                    backstory=temporary.backstory,
                    active_skills=list(temporary.active_skills),
                )
        return None

    @staticmethod
    def _consume(context: AgentRuntimeContext, field: str, limit: int) -> None:
        current = getattr(context.usage, field)
        if current >= limit:
            raise RuntimeBudgetExceededError(f"Runtime budget exceeded: {field}={limit}")
        setattr(context.usage, field, current + 1)

    @staticmethod
    def _check_total(context: AgentRuntimeContext, field: str, limit: int) -> None:
        if getattr(context.usage, field) > limit:
            raise RuntimeBudgetExceededError(f"Runtime budget exceeded: {field}={limit}")

    @staticmethod
    def _check_time(context: AgentRuntimeContext) -> None:
        started = context.runtime_started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        if elapsed > context.budgets.max_time:
            raise RuntimeBudgetExceededError("Runtime budget exceeded: max_time")

    @staticmethod
    def _delegation_payload(record: DelegationRecord) -> dict[str, object]:
        return {
            "source": record.source.model_dump(mode="json"),
            "target": record.target.model_dump(mode="json"),
            "delegation_type": record.delegation_type.value,
            "task_summary": record.task_summary,
            "timestamp": record.timestamp.isoformat(),
            "success": record.success,
            "error": record.error,
        }

    @staticmethod
    def _tool_feedback(status: ToolExecutionStatus, output: object, error: str | None) -> str:
        if status == ToolExecutionStatus.EXECUTED:
            return f"Tool result: {json.dumps(output, ensure_ascii=False, default=str)}"
        return f"Tool {status.value}: {error or 'no result'}"

    def _get_approval_runtime(self) -> ToolApprovalRuntime:
        if self._approval_runtime is None:
            from agents.assistant.crewai.runtime import CrewAIToolApprovalRuntime

            self._approval_runtime = CrewAIToolApprovalRuntime(
                authority=self._tool_authority,
                persistence_path=self._approval_persistence_path,
            )
        return self._approval_runtime
