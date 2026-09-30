"""Tenant-aware creation and runtime resolution of persistent user crews."""

import builtins
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from models.agent import Agent, AgentKind, AgentStatus
from models.agent_factory import CrewBlueprint, CrewDefinition
from models.memory import PERSISTENT_AGENT_MEMORY_SCOPES
from models.crew import (
    CrewMember, CrewMemberCreate, CrewSkill, CrewSkillCreate, CrewStatus, CrewTask, CrewTaskCreate,
    RuntimeCrewTask, UserCrew, UserCrewCreate,
)
from repositories.agent_memory_policy import AgentMemoryPolicyRepository
from repositories.agent_prompt import AgentPromptRepository
from repositories.crew import CrewRepository
from services.agent import AgentService
from services.agent_design import AgentDesignService
from services.agent_runtime import AgentRuntimeService
from services.skill import SkillService


class CrewService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._crews = CrewRepository(session)
        self._agents = AgentService(session)
        self._design = AgentDesignService(session)
        self._prompts = AgentPromptRepository(session)
        self._skills = SkillService(session)
        self._memory_policies = AgentMemoryPolicyRepository(session)

    async def create_from_blueprint(self, *, user_id: int, blueprint: CrewBlueprint) -> UserCrew:
        if blueprint.missing_capabilities or any(
            agent.missing_capabilities for agent in blueprint.agents
        ):
            raise ValueError("Crew has missing capabilities and cannot be created")
        try:
            await self._agents.ensure_primary_agent(user_id=user_id, commit=False)
            await self._design.validate_crew_blueprint(user_id=user_id, blueprint=blueprint)
            members: builtins.list[Agent] = []
            for agent_blueprint in blueprint.agents:
                members.append(await self._agents.create_from_blueprint(
                    user_id=user_id, blueprint=agent_blueprint, commit=False
                ))
            for agent_id in blueprint.existing_agent_ids:
                members.append(await self._agents.get_agent(user_id=user_id, agent_id=agent_id))

            crew = await self._crews.create(UserCrewCreate(
                user_id=user_id,
                name=blueprint.name,
                purpose=blueprint.purpose,
                process=blueprint.process,
            ))
            for position, member in enumerate(members):
                await self._crews.add_member(CrewMemberCreate(
                    crew_id=crew.id, agent_id=member.id, position=position,
                ))
            for position, task in enumerate(blueprint.tasks):
                await self._crews.add_task(CrewTaskCreate(
                    crew_id=crew.id,
                    agent_id=members[task.agent_index].id,
                    position=position,
                    description=task.description,
                    expected_output=task.expected_output,
                ))
            for skill_id in blueprint.required_skill_ids:
                await self._crews.add_skill(CrewSkillCreate(
                    crew_id=crew.id, skill_id=skill_id,
                ))
            await self._session.commit()
            return crew
        except Exception:
            await self._session.rollback()
            raise

    async def get(self, *, user_id: int, crew_id: UUID) -> UserCrew:
        crew = await self._crews.get_by_id(crew_id)
        if crew is None or crew.user_id != user_id:
            raise LookupError(f"Crew not found: {crew_id}")
        return crew

    async def list(self, *, user_id: int) -> builtins.list[UserCrew]:
        return await self._crews.list_by_user(user_id)

    async def list_members(self, *, user_id: int, crew_id: UUID) -> builtins.list[CrewMember]:
        await self.get(user_id=user_id, crew_id=crew_id)
        return await self._crews.list_members(crew_id)

    async def list_tasks(self, *, user_id: int, crew_id: UUID) -> builtins.list[CrewTask]:
        await self.get(user_id=user_id, crew_id=crew_id)
        return await self._crews.list_tasks(crew_id)

    async def list_skills(self, *, user_id: int, crew_id: UUID) -> builtins.list[CrewSkill]:
        await self.get(user_id=user_id, crew_id=crew_id)
        return await self._crews.list_skills(crew_id)

    async def archive(self, *, user_id: int, crew_id: UUID) -> UserCrew:
        crew = await self.get(user_id=user_id, crew_id=crew_id)
        if crew.status == CrewStatus.ARCHIVED:
            return crew
        updated = await self._crews.set_status(crew_id, CrewStatus.ARCHIVED)
        assert updated is not None
        await self._session.commit()
        return updated

    async def build_runtime_definition(self, *, user_id: int, crew_id: UUID) -> CrewDefinition:
        crew = await self.get(user_id=user_id, crew_id=crew_id)
        if crew.status != CrewStatus.ACTIVE:
            raise ValueError("Archived crew cannot be run")
        members = await self._crews.list_members(crew.id)
        tasks = await self._crews.list_tasks(crew.id)
        required_skills = await self._crews.list_skills(crew.id)
        if not members or not tasks:
            raise ValueError("Crew definition is incomplete")
        definitions = []
        member_ids = set()
        resolved_skill_ids = set()
        for member in members:
            agent = await self._agents.get_agent(user_id=user_id, agent_id=member.agent_id)
            if agent.kind != AgentKind.USER or agent.status != AgentStatus.ACTIVE:
                raise ValueError("Crew member is not an active user agent")
            prompt = await self._prompts.get_latest(agent.id)
            if prompt is None:
                raise LookupError("Crew member prompt is missing")
            skills = await self._skills.resolve_agent_skills(user_id=user_id, agent_id=agent.id)
            resolved_skill_ids.update(skill.id for skill in skills if skill.id is not None)
            policy = await self._memory_policies.get_scopes(agent.id)
            definitions.append(AgentRuntimeService._persistent_definition(
                agent, prompt, active_skills=skills,
                memory_scopes=policy if policy is not None else list(PERSISTENT_AGENT_MEMORY_SCOPES),
            ))
            member_ids.add(agent.id)
        if any(task.agent_id not in member_ids for task in tasks):
            raise ValueError("Crew task references a nonmember agent")
        if not {skill.skill_id for skill in required_skills if skill.required}.issubset(resolved_skill_ids):
            raise ValueError("A required crew skill is no longer active for its agents")
        return CrewDefinition(
            id=crew.id,
            user_id=user_id,
            name=crew.name,
            purpose=crew.purpose,
            config_version=crew.config_version,
            agents=definitions,
            tasks=[RuntimeCrewTask(
                agent_id=task.agent_id,
                description=task.description,
                expected_output=task.expected_output,
            ) for task in tasks],
            required_skill_ids=[skill.skill_id for skill in required_skills if skill.required],
            process=crew.process,
        )
