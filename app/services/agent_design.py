"""Backend entrypoint for designing and accepting user agent proposals."""

import re

from sqlalchemy.ext.asyncio import AsyncSession

from agents.factory.runner import AgentFactoryRunner
from models.agent import Agent, AgentKind, AgentStatus
from models.agent_factory import (
    AgentBlueprint,
    AgentFactoryInput,
    AvailableAgent,
    AvailableSkill,
    CrewBlueprint,
    RequestedPermission,
)
from models.crew import CrewProcess
from models.memory import PERSISTENT_AGENT_MEMORY_SCOPES
from models.permission import ActionClass, PermissionSubjectType
from services.agent import AgentService
from services.permission import PermissionService
from services.skill import SkillService


class AgentDesignService:
    def __init__(self, session: AsyncSession, *, runner: AgentFactoryRunner | None = None) -> None:
        self._agents = AgentService(session)
        self._skills = SkillService(session)
        self._permissions = PermissionService(session)
        self._runner = runner or AgentFactoryRunner()

    async def design_agent(self, *, user_id: int, user_request: str) -> AgentBlueprint:
        request = await self._factory_input(user_id=user_id, user_request=user_request)
        blueprint = await self._runner.design_agent(request)
        await self._agents.validate_blueprint(user_id=user_id, blueprint=blueprint)
        return blueprint

    async def design_crew(self, *, user_id: int, user_request: str) -> CrewBlueprint:
        """Return a checked proposal without writing crew or agent records."""
        request = await self._factory_input(user_id=user_id, user_request=user_request)
        blueprint = await self._runner.design_crew(request)
        await self.validate_crew_blueprint(user_id=user_id, blueprint=blueprint)
        return blueprint

    async def validate_crew_blueprint(self, *, user_id: int, blueprint: CrewBlueprint) -> None:
        if blueprint.process != CrewProcess.SEQUENTIAL:
            raise ValueError("Only sequential crews are supported")
        member_count = len(blueprint.agents) + len(blueprint.existing_agent_ids)
        if any(task.agent_index >= member_count for task in blueprint.tasks):
            raise ValueError("Crew task references an unavailable agent")
        if len(set(blueprint.existing_agent_ids)) != len(blueprint.existing_agent_ids):
            raise ValueError("Duplicate existing agent IDs")
        selected_skills = set()
        names: set[str] = set()
        for agent in blueprint.agents:
            await self._agents.validate_blueprint(user_id=user_id, blueprint=agent)
            name_key = agent.name.casefold()
            if name_key in names:
                raise ValueError("Duplicate crew agent names")
            names.add(name_key)
            selected_skills.update(agent.skill_ids)
        primary = (
            await self._agents.ensure_primary_agent(user_id=user_id, commit=False)
            if blueprint.existing_agent_ids else None
        )
        for agent_id in blueprint.existing_agent_ids:
            agent = await self._agents.get_agent(user_id=user_id, agent_id=agent_id)
            if agent.kind != AgentKind.USER or agent.status != AgentStatus.ACTIVE:
                raise ValueError("Crew member must be an active user agent")
            name_key = agent.name.casefold()
            if name_key in names:
                raise ValueError("Duplicate crew agent names")
            names.add(name_key)
            selected_skills.update(
                skill.id for skill in await self._skills.list_agent_skills(
                    user_id=user_id, agent_id=agent_id
                )
            )
            assert primary is not None
            for encoded in await self._permissions.list_allowed_scopes(
                user_id=user_id,
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id=str(agent.id),
            ):
                action, resource = encoded.split(":", 1)
                if not await self._permissions.check(
                    user_id=user_id,
                    subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                    subject_id=str(primary.id),
                    action_class=ActionClass(action),
                    resource=resource,
                ):
                    raise PermissionError("Crew member permissions exceed primary agent permissions")
        available = {
            skill.id for skill in await self._skills.list_available_skills(user_id=user_id)
        }
        if len(set(blueprint.required_skill_ids)) != len(blueprint.required_skill_ids):
            raise ValueError("Duplicate required crew skill IDs")
        if not set(blueprint.required_skill_ids).issubset(available):
            raise ValueError("Crew requires a skill unavailable to this user")
        if not set(blueprint.required_skill_ids).issubset(selected_skills):
            raise ValueError("Crew requires a skill not assigned to its agents")

    async def _factory_input(self, *, user_id: int, user_request: str) -> AgentFactoryInput:
        primary = await self._agents.ensure_primary_agent(user_id=user_id)
        if primary.status != AgentStatus.ACTIVE:
            raise ValueError("Primary agent is not active")
        skills = await self._skills.list_available_skills(user_id=user_id)
        agents = await self._agents.list_active_agents(user_id)
        scope_strings = await self._permissions.list_allowed_scopes(
            user_id=user_id,
            subject_type=PermissionSubjectType.PERSISTENT_AGENT,
            subject_id=str(primary.id),
        )
        parent_permissions = []
        for encoded in scope_strings:
            action, resource = encoded.split(":", 1)
            parent_permissions.append(RequestedPermission(
                action_class=action, resource_scope=resource,
            ))
        available_agents = []
        for agent in agents:
            if agent.kind != AgentKind.USER:
                continue
            prompt = await self._agents.get_current_prompt(user_id=user_id, agent_id=agent.id)
            available_agents.append(AvailableAgent(
                id=agent.id,
                name=agent.name,
                role=prompt.role,
                goal=prompt.goal,
                skill_ids=[skill.id for skill in await self._skills.list_agent_skills(
                    user_id=user_id, agent_id=agent.id
                )],
            ))
        return AgentFactoryInput(
            user_request=user_request.strip(),
            available_skills=[AvailableSkill(
                id=skill.id,
                key=skill.key,
                name=skill.name,
                description=skill.description,
                required_permissions=skill.required_permissions,
            ) for skill in skills],
            available_agents=available_agents,
            parent_permissions=parent_permissions,
            parent_can_spawn_subagents=primary.can_spawn_subagents,
        )

    async def create_agent(self, *, user_id: int, blueprint: AgentBlueprint) -> Agent:
        return await self._agents.create_from_blueprint(user_id=user_id, blueprint=blueprint)

    async def preview_text(self, *, user_id: int, blueprint: AgentBlueprint) -> str:
        skills = {item.id: item.name for item in await self._skills.list_available_skills(user_id=user_id)}
        selected = [skills.get(skill_id, str(skill_id)) for skill_id in blueprint.skill_ids]
        memory_scopes = blueprint.memory_scopes or list(PERSISTENT_AGENT_MEMORY_SCOPES)
        return (
            "Create this agent?\n\n"
            f"Name: {blueprint.name}\n"
            f"Role: {blueprint.role}\n"
            f"Goal: {blueprint.goal}\n"
            f"Backstory: {blueprint.backstory or 'none'}\n"
            f"Instructions: {blueprint.custom_instructions or 'none'}\n"
            f"Skills: {', '.join(selected) or 'none'}\n"
            f"Missing capabilities: {', '.join(blueprint.missing_capabilities) or 'none'}\n"
            f"Can spawn subagents: {'yes' if blueprint.can_spawn_subagents else 'no'}\n"
            f"Memory scopes: {', '.join(item.value for item in memory_scopes)}\n"
            f"Requested permissions: {', '.join(f'{item.action_class.value}:{item.resource_scope}' for item in blueprint.requested_permissions) or 'none'}"
        )

    async def crew_preview_text(self, *, user_id: int, blueprint: CrewBlueprint) -> str:
        skills = {item.id: item.name for item in await self._skills.list_available_skills(user_id=user_id)}
        agent_names = [agent.name for agent in blueprint.agents]
        selected_skill_ids = set(blueprint.required_skill_ids)
        for agent in blueprint.agents:
            selected_skill_ids.update(agent.skill_ids)
        for agent_id in blueprint.existing_agent_ids:
            agent_names.append((await self._agents.get_agent(user_id=user_id, agent_id=agent_id)).name)
            selected_skill_ids.update(
                skill.id for skill in await self._skills.list_agent_skills(
                    user_id=user_id, agent_id=agent_id
                )
            )
        missing = list(blueprint.missing_capabilities)
        missing.extend(
            capability for agent in blueprint.agents for capability in agent.missing_capabilities
        )
        lines = [
            f"Crew: {blueprint.name}",
            f"Purpose: {blueprint.purpose}",
            f"Process: {blueprint.process.value}",
            "", "Agents:",
            *(f"{index}. {name}" for index, name in enumerate(agent_names, 1)),
            *(f"  {agent.name}: role={agent.role}; goal={agent.goal}; "
              f"backstory={agent.backstory or 'none'}; instructions={agent.custom_instructions or 'none'}"
              for agent in blueprint.agents),
            *(f"  {agent.name}: memory={','.join(scope.value for scope in (agent.memory_scopes or list(PERSISTENT_AGENT_MEMORY_SCOPES)))}; "
              f"subagents={'yes' if agent.can_spawn_subagents else 'no'}; "
              f"permissions={','.join(f'{permission.action_class.value}:{permission.resource_scope}' for permission in agent.requested_permissions) or 'none'}"
              for agent in blueprint.agents),
            "", "Tasks:",
            *(f"{index}. {task.description} → {agent_names[task.agent_index]}"
              for index, task in enumerate(blueprint.tasks, 1)),
            "", f"Skills: {', '.join(skills.get(skill_id, str(skill_id)) for skill_id in sorted(selected_skill_ids, key=str)) or 'none'}",
            f"Required skills: {', '.join(skills.get(skill_id, str(skill_id)) for skill_id in blueprint.required_skill_ids) or 'none'}",
            f"Missing capabilities: {', '.join(missing) or 'none'}",
        ]
        if missing:
            lines.append("Creation is blocked until missing capabilities are available.")
        return "\n".join(lines)


def is_agent_design_request(text: str) -> bool:
    return design_request_kind(text) == "agent"


def is_crew_design_request(text: str) -> bool:
    return design_request_kind(text) == "crew"


def design_request_kind(text: str) -> str | None:
    """Transitional Telegram bridge; Primary runtime decisions will replace this classifier."""
    clean = text.casefold().strip()
    if not re.match(r"^(создай(?:те)?|сделай(?:те)?|create|build|мне нужен|мне нужна|i need)\b", clean):
        return None
    if re.search(r"\b(команд\w*|crew|team)\b", clean[:120]):
        return "crew"
    if re.search(r"\b(агент\w*|помощник\w*|agent|assistant)\b", clean[:120]):
        return "agent"
    return None
