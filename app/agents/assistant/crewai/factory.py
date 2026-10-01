"""Build dynamic CrewAI crews exclusively from resolved runtime models."""

import json
from typing import Any
from uuid import UUID

from crewai import Agent, Crew, Process, Task
from crewai.experimental.agent_executor import AgentExecutor
from crewai.skills.models import Skill as NativeSkill
from crewai.tools.base_tool import BaseTool
from agents.assistant.crewai.llm_budget import BudgetedLLM, LLMCallBudget

from integrations.llm.factory import create_crewai_llm
from models.agent_factory import CrewDefinition
from models.runtime import (
    AgentRuntimeContext,
    AgentRuntimeRequest,
    RuntimeAgentDefinition,
    RuntimeSkillDefinition,
    RuntimeStep,
    RuntimeBudgets,
    RuntimeAgentKind,
)


class DynamicCrewAIFactory:
    """Pure Pydantic runtime definitions -> CrewAI objects conversion."""

    def __init__(self, *, llm: Any | None = None) -> None:
        self._llm = llm

    def build(
        self, context: AgentRuntimeContext, request: AgentRuntimeRequest, *,
        native_skills: list[NativeSkill] | None = None,
        resolved_tools: list[BaseTool] | None = None,
    ) -> Crew:
        definition = self.resolve_active_definition(context)
        llm = self._llm if self._llm is not None else create_crewai_llm()
        agent = self.build_agent(
            definition, budgets=context.budgets, llm=llm,
            native_skills=native_skills, resolved_tools=resolved_tools,
        )
        task = Task(
            description=self._task_description(context, request),
            expected_output=(
                "A final RuntimeStep structured response after using any needed native tools. "
                "Choose exactly one delegation decision. Call load_skill and provided native "
                "READ tools directly before returning the final RuntimeStep. ToolIntent is "
                "reserved for backend actions unavailable as native tools, including approvals."
            ),
            agent=agent,
            output_pydantic=RuntimeStep,
        )
        return Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
            memory=False,
        )

    def build_agent(
        self, definition: RuntimeAgentDefinition, *, budgets: RuntimeBudgets,
        llm: Any | None = None,
        native_skills: list[NativeSkill] | None = None,
        resolved_tools: list[BaseTool] | None = None,
    ) -> Agent:
        """Build an agent only from backend resolved prompt, capabilities, and limits."""
        selected_llm = llm if llm is not None else (self._llm if self._llm is not None else create_crewai_llm())
        return Agent(
            role=definition.role,
            goal=definition.goal,
            backstory=self._backstory(definition),
            llm=selected_llm,
            verbose=False,
            allow_delegation=False,
            max_iter=budgets.max_agent_iterations,
            # Backend tools marshal worker-thread calls to their owning async loop.
            executor_class=AgentExecutor,
            skills=native_skills or None,
            tools=resolved_tools or [],
        )

    def build_from_definition(
        self, definition: CrewDefinition, *,
        native_skills_by_agent: dict[UUID, list[NativeSkill]] | None = None,
        resolved_tools_by_agent: dict[UUID, list[BaseTool]] | None = None,
        crew_skills: list[NativeSkill] | None = None,
        llm_call_budget: LLMCallBudget | None = None,
    ) -> Crew:
        """Build a sequential crew from a backend validated definition."""
        llm = self._llm if self._llm is not None else create_crewai_llm()
        if llm_call_budget is not None:
            llm = BudgetedLLM(llm, llm_call_budget)
        agents = [
            self.build_agent(
                item, budgets=definition.budgets, llm=llm,
                native_skills=(native_skills_by_agent or {}).get(UUID(item.identity.subject_id)),
                resolved_tools=(resolved_tools_by_agent or {}).get(UUID(item.identity.subject_id)),
            ) for item in definition.agents
        ]
        by_id = {
            item.identity.subject_id: agent
            for item, agent in zip(definition.agents, agents, strict=True)
        }
        if len(by_id) != len(agents):
            raise ValueError("Crew definition has duplicate agent identities")
        tasks = []
        for item in definition.tasks:
            agent = by_id.get(str(item.agent_id))
            if agent is None:
                raise ValueError("Crew task references an unavailable agent")
            tasks.append(Task(
                description=item.description,
                expected_output=item.expected_output,
                agent=agent,
            ))
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
            memory=False,
            skills=crew_skills or None,
        )

    @staticmethod
    def resolve_active_definition(context: AgentRuntimeContext) -> RuntimeAgentDefinition:
        definitions = [
            context.starting_agent,
            *context.connected_persistent_agents,
            *context.available_system_agents,
        ]
        for definition in definitions:
            if definition.identity.subject_id == context.active_agent.subject_id:
                return definition
        for temporary in context.temporary_subagents:
            if temporary.identity.subject_id == context.active_agent.subject_id:
                return RuntimeAgentDefinition(
                    identity=temporary.identity,
                    role=temporary.role,
                    goal=temporary.goal,
                    backstory=temporary.backstory,
                    can_spawn_subagents=False,
                    active_skills=list(temporary.active_skills),
                )
        raise LookupError(
            f"Active runtime agent definition not found: {context.active_agent.subject_id}"
        )

    @staticmethod
    def _backstory(definition: RuntimeAgentDefinition) -> str:
        parts = [definition.backstory or ""]
        if definition.custom_instructions:
            parts.append(f"Additional instructions: {definition.custom_instructions}")
        return "\n\n".join(part for part in parts if part) or definition.role

    @staticmethod
    def _task_description(
        context: AgentRuntimeContext, request: AgentRuntimeRequest
    ) -> str:
        definition = DynamicCrewAIFactory.resolve_active_definition(context)
        skills = context.active_skills or definition.active_skills
        is_primary = context.active_agent.kind == RuntimeAgentKind.PRIMARY
        user_targets = [
            {
                "id": item.identity.subject_id,
                "name": item.identity.name,
                "role": item.role,
            }
            for item in context.connected_persistent_agents if is_primary
        ]
        system_targets = [
            {
                "id": item.identity.subject_id,
                "name": item.identity.name,
                "role": item.role,
            }
            for item in context.available_system_agents if is_primary
        ]
        history = [item.model_dump(mode="json") for item in context.chat_context]
        crew_targets = [item.model_dump(mode="json") for item in context.available_crews] if is_primary else []
        if context.current_input == request.message:
            input_context = f"Current user message: {request.message}\n"
        else:
            input_context = (
                f"Original user message: {request.message}\n"
                f"Current runtime input: {context.current_input}\n"
            )
        return (
            "Process the current runtime input and return only the structured RuntimeStep.\n"
            f"Active runtime identity: {context.active_agent.model_dump_json()}\n"
            + (
                "You are the primary coordinator. Delegate specialized work to a listed worker "
                "when the user's request requires its skills.\n"
                if is_primary else
                "You are the active worker assigned to execute the current runtime input. "
                "Perform this task using your available skills and tools, then choose RESPOND "
                "to return your result to the parent. Persistent delegation decisions are unavailable. "
                "The original user message supplies background; execute the current runtime input.\n"
            )
            +
            f"{input_context}"
            f"{DynamicCrewAIFactory._skills_context(skills)}\n"
            "Native CrewAI tools, including load_skill and the provided READ tools, are callable "
            "during this execution. Invoke them directly to gather instructions and data before "
            "returning your final RuntimeStep. Never put load_skill or a provided native tool "
            "into tool_intent. For a listed Skill, use its exact native catalog name when calling "
            "load_skill. ToolIntent is for backend actions without a native tool surface.\n"
            f"{DynamicCrewAIFactory._memory_context(context)}\n"
            f"Conversation history: {json.dumps(history, ensure_ascii=False)}\n"
            f"Connected user-agent targets: {json.dumps(user_targets, ensure_ascii=False)}\n"
            f"Available system-agent targets: {json.dumps(system_targets, ensure_ascii=False)}\n"
            f"Available crews: {json.dumps(crew_targets, ensure_ascii=False)}\n"
            "Only the Primary can run a listed persistent crew. To use a listed crew, return "
            "decision type='run_crew', target_id='<listed crew UUID>', and task_summary containing "
            "the concrete user task. Crew results will return to you for the final response.\n"
            "Never invent a target identifier. Persistent worker-to-worker delegation is forbidden. "
            "Backend permission, approval, budget, and execution checks are authoritative."
        )

    @staticmethod
    def _skills_context(skills: list[RuntimeSkillDefinition]) -> str:
        lines = ["Resolved active skills (instruction metadata only):"]
        if not skills:
            lines.append("- none")
        for skill in skills:
            lines.append(f"- {skill.key}@{skill.version} — {skill.name}: {skill.description}")
            if skill.id is not None and skill.entrypoint:
                lines.append(
                    "  Executable via execute_skill only: "
                    f"skill_key={skill.key}, pinned_skill_id={skill.id}, "
                    f"version={skill.version}, action_class={skill.action_class.value}"
                )
            if skill.instructions and skill.storage_uri is None:
                lines.append(f"  Instructions: {skill.instructions}")
            for constraint in skill.constraints:
                lines.append(f"  Constraint: {constraint}")
            if skill.required_permissions:
                permissions = ", ".join(skill.required_permissions)
                lines.append(f"  Required permissions: {permissions}")
        lines.append(
            "Skill metadata never grants permissions; backend permission and approval checks remain authoritative."
        )
        if any(skill.id is not None and skill.entrypoint for skill in skills):
            lines.append(
                "To execute a skill, return ToolIntent(name='execute_skill') with "
                "arguments={'skill_key': '<listed key>', 'arguments': <JSON object>} and "
                "resource='skill:<listed immutable id>:v<listed version>'. Never invent a skill id or version."
            )
        return "\n".join(lines)

    @staticmethod
    def _memory_context(context: AgentRuntimeContext) -> str:
        lines = ["Relevant memory (filtered for this agent and run):"]
        if not context.memory:
            lines.append("- none")
        else:
            lines.extend(
                f"- [{item.scope.value}] {item.key}: {item.content}"
                for item in context.memory
            )
        return "\n".join(lines)
