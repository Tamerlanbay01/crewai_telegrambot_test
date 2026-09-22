"""Build dynamic CrewAI crews exclusively from resolved runtime models."""

import json
from typing import Any

from crewai import Agent, Crew, Process, Task

from integrations.llm.factory import create_crewai_llm
from models.runtime import (
    AgentRuntimeContext,
    AgentRuntimeRequest,
    RuntimeAgentDefinition,
    RuntimeSkillDefinition,
    RuntimeStep,
)


class DynamicCrewAIFactory:
    """Pure Pydantic runtime definitions -> CrewAI objects conversion."""

    def __init__(self, *, llm: Any | None = None) -> None:
        self._llm = llm

    def build(self, context: AgentRuntimeContext, request: AgentRuntimeRequest) -> Crew:
        definition = self.resolve_active_definition(context)
        llm = self._llm if self._llm is not None else create_crewai_llm()
        agent = Agent(
            role=definition.role,
            goal=definition.goal,
            backstory=self._backstory(definition),
            llm=llm,
            verbose=False,
            allow_delegation=False,
            max_iter=context.budgets.max_agent_iterations,
        )
        task = Task(
            description=self._task_description(context, request),
            expected_output=(
                "A RuntimeStep structured response. Choose exactly one delegation decision; "
                "include a ToolIntent only when a backend action is needed."
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
        user_targets = [
            {
                "id": item.identity.subject_id,
                "name": item.identity.name,
                "role": item.role,
            }
            for item in context.connected_persistent_agents
        ]
        system_targets = [
            {
                "id": item.identity.subject_id,
                "name": item.identity.name,
                "role": item.role,
            }
            for item in context.available_system_agents
        ]
        history = [item.model_dump(mode="json") for item in context.chat_context]
        if context.current_input == request.message:
            input_context = f"Current user message: {request.message}\n"
        else:
            input_context = (
                f"Original user message: {request.message}\n"
                f"Current runtime input: {context.current_input}\n"
            )
        return (
            "Process the current runtime input and return only the structured RuntimeStep.\n"
            f"{input_context}"
            f"{DynamicCrewAIFactory._skills_context(skills)}\n"
            f"{DynamicCrewAIFactory._memory_context(context)}\n"
            f"Conversation history: {json.dumps(history, ensure_ascii=False)}\n"
            f"Connected user-agent targets: {json.dumps(user_targets, ensure_ascii=False)}\n"
            f"Available system-agent targets: {json.dumps(system_targets, ensure_ascii=False)}\n"
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
            if skill.instructions:
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
