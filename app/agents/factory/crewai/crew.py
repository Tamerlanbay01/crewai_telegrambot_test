"""Three role CrewAI design review with no tools or persistence access."""

import json
from typing import Any

from crewai import Agent, Crew, Process, Task

from integrations.llm.factory import create_crewai_llm
from models.agent_factory import AgentBlueprint, AgentFactoryInput, CrewBlueprint
from models.memory import PERSISTENT_AGENT_MEMORY_SCOPES


class AgentFactoryCrew:
    def __init__(self, *, llm: Any | None = None) -> None:
        self._llm = llm

    def build(self, request: AgentFactoryInput, *, crew: bool = False) -> Crew:
        llm = self._llm if self._llm is not None else create_crewai_llm()
        max_iterations = min(request.limits.max_agent_iterations, 4)
        factory_context = self._context(request)
        budget_rule = (
            "Do not propose custom per-agent budgets. budget_profile must be null. "
            "Platform/runtime budgets are hard constraints only."
        )
        designer = Agent(
            role="AgentDesigner",
            goal="Design an agent or crew that fulfills the user's request",
            backstory="Propose declarative roles, goals, and capabilities only.",
            llm=llm, allow_delegation=False, verbose=False,
            max_iter=max_iterations,
        )
        resolver = Agent(
            role="SkillResolver",
            goal="Select only identifiers from the available skill catalog",
            backstory="Report unavailable capabilities as missing_capabilities; never install skills.",
            llm=llm, allow_delegation=False, verbose=False,
            max_iter=max_iterations,
        )
        reviewer = Agent(
            role="PolicyReviewer",
            goal="Return a structured proposal within the supplied constraints",
            backstory="Review permissions, memory, budgets, delegation, and topology. Backend validation is authoritative.",
            llm=llm, allow_delegation=False, verbose=False,
            max_iter=max_iterations,
        )
        design = Task(
            description=(
                "Design a declarative CrewBlueprint. Inline agents go in agents; existing catalog IDs go in existing_agent_ids. "
                "Task agent_index refers to inline agents followed by existing agents. "
                if crew else "Design a declarative AgentBlueprint. "
            ) + budget_rule + "\n" + factory_context,
            expected_output="A proposal based on the supplied catalog and limits.",
            agent=designer,
        )
        skills = Task(
            description=(
                "Resolve skill IDs against the supplied catalog. Never invent IDs. "
                "Put unmet needs in missing_capabilities and leave absent skills unselected. "
                f"{budget_rule}\n{factory_context}"
            ),
            expected_output="A proposal with only real skill identifiers and explicit missing capabilities.",
            agent=resolver,
            context=[design],
        )
        review = Task(
            description=(
                "Review the design and skill resolution. Use only supplied parent permissions. "
                "Only the primary agent may connect persistent agents; worker peer connections are forbidden. "
                "Do not generate Python or install skills. Return the structured blueprint. "
                f"{budget_rule}\n{factory_context}"
            ),
            expected_output="A validated-shape blueprint; backend will independently verify every identifier and permission.",
            agent=reviewer,
            context=[design, skills],
            output_pydantic=CrewBlueprint if crew else AgentBlueprint,
        )
        return Crew(
            agents=[designer, resolver, reviewer],
            tasks=[design, skills, review],
            process=Process.sequential,
            verbose=False,
            memory=False,
        )

    @staticmethod
    def _context(request: AgentFactoryInput) -> str:
        return (
            f"User request: {request.user_request}\n"
            "Available skills: "
            f"{json.dumps([item.model_dump(mode='json') for item in request.available_skills], ensure_ascii=False)}\n"
            "Available agents: "
            f"{json.dumps([item.model_dump(mode='json') for item in request.available_agents], ensure_ascii=False)}\n"
            "Parent permissions: "
            f"{json.dumps([item.model_dump(mode='json') for item in request.parent_permissions], ensure_ascii=False)}\n"
            f"Parent can spawn subagents: {request.parent_can_spawn_subagents}\n"
            "Allowed persistent-agent memory_scopes: "
            f"{json.dumps([scope.value for scope in PERSISTENT_AGENT_MEMORY_SCOPES])}. "
            "Use only these values, or [] for the backend defaults. "
            "CREW_SHARED and RUN_EPHEMERAL are not persistent-agent memory policies.\n"
            f"Platform limits: {request.limits.model_dump_json()}"
        )
