from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from models.agent import Agent, AgentCreate, AgentKind, AgentStatus, AgentUpdate
from models.agent_connection import AgentConnection, AgentConnectionCreate
from models.agent_prompt import AgentPromptVersion, AgentPromptVersionCreate
from models.agent_factory import AgentBlueprint
from models.memory import PERSISTENT_AGENT_MEMORY_SCOPES
from models.permission import ActionClass, PermissionCreate, PermissionSubjectType
from models.runtime import RuntimeBudgets
from models.skill import AgentSkillCreate
from repositories.agent import AgentRepository
from repositories.agent_memory_policy import AgentMemoryPolicyRepository
from repositories.agent_connection import AgentConnectionRepository
from repositories.agent_prompt import AgentPromptRepository
from repositories.permission import PermissionRepository
from repositories.skill import SkillRepository
from services.permission import PermissionService
from services.skill import SkillService

class AgentService:
    def __init__(self, session: AsyncSession):
        self._session = session
        self._agents = AgentRepository(session)
        self._memory_policies = AgentMemoryPolicyRepository(session)
        self._prompts = AgentPromptRepository(session)
        self._connections = AgentConnectionRepository(session)
        self._skills = SkillRepository(session)
        self._permissions = PermissionRepository(session)

    async def _ensure_primary_agent(self, *, user_id: int) -> Agent:
        agent = await self._agents.get_primary(user_id)

        if agent is not None:
            return agent
        
        agent = await self._agents.create(AgentCreate(
            user_id = user_id,
            kind = AgentKind.PRIMARY,
            name = "Personal Assistant",
            can_spawn_subagents=True
        ))

        await self._prompts.create(AgentPromptVersionCreate(
            agent_id = agent.id,
            version = 1,
            role = "Personal Assistant",
            goal = (
                """Understand the user's request and coordinate the appropriate agents and capabilities"""
            ),
            backstory = (
                """You are the user's primary personal assistant and the main orchestrator of their agents"""
            )
        ))
        return agent

    async def ensure_primary_agent(self, *, user_id: int, commit: bool = True) -> Agent:
        """Return the user's primary agent, creating its initial prompt if needed."""
        agent = await self._ensure_primary_agent(user_id=user_id)
        if commit:
            await self._session.commit()
        return agent

    async def create_user_agent(
        self,
        *,
        user_id: int,
        name: str,
        role: str,
        goal: str,
        backstory: str | None = None,
        custom_instructions: str | None = None,
        can_spawn_subagents: bool = False,
    ) -> Agent:
        return await self._create_user_agent(
            user_id=user_id,
            name=name,
            role=role,
            goal=goal,
            backstory=backstory,
            custom_instructions=custom_instructions,
            can_spawn_subagents=can_spawn_subagents,
        )

    async def _create_user_agent(
        self,
        *,
        user_id: int,
        name: str,
        role: str,
        goal: str,
        backstory: str | None = None,
        custom_instructions: str | None = None,
        can_spawn_subagents: bool = False,
        blueprint: AgentBlueprint | None = None,
        commit: bool = True,
    ) -> Agent:
        if blueprint is not None:
            await self.validate_blueprint(user_id=user_id, blueprint=blueprint)
        clean_name = name.strip()
        clean_role = role.strip()
        clean_goal = goal.strip()

        if not clean_name:
            raise ValueError(
                "Agent name cannot be empty"
            )

        if not clean_role:
            raise ValueError(
                "Agent role cannot be empty"
            )

        if not clean_goal:
            raise ValueError(
                "Agent goal cannot be empty"
            )

        agent = await self._agents.create(
            AgentCreate(
                user_id=user_id,
                kind=AgentKind.USER,
                name=clean_name,
                can_spawn_subagents=(
                    can_spawn_subagents
                ),
            )
        )

        await self._prompts.create(
            AgentPromptVersionCreate(
                agent_id=agent.id,
                version=1,
                role=clean_role,
                goal=clean_goal,
                backstory=(
                    backstory.strip()
                    if backstory
                    else None
                ),
                custom_instructions=(
                    custom_instructions.strip()
                    if custom_instructions
                    else None
                ),
            )
        )

        primary_agent = await self._ensure_primary_agent(user_id=user_id)
        await self._connections.create(
            AgentConnectionCreate(
                parent_agent_id=primary_agent.id,
                child_agent_id=agent.id,
            )
        )

        if blueprint is not None:
            for skill_id in blueprint.skill_ids:
                await self._skills.assign(AgentSkillCreate(agent_id=agent.id, skill_id=skill_id))
            for permission in blueprint.requested_permissions:
                await self._permissions.create(PermissionCreate(
                    user_id=user_id,
                    subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                    subject_id=str(agent.id),
                    action_class=permission.action_class,
                    resource_scope=permission.resource_scope,
                ))
        memory_scopes = (
            blueprint.memory_scopes
            if blueprint is not None and blueprint.memory_scopes
            else list(PERSISTENT_AGENT_MEMORY_SCOPES)
        )
        await self._memory_policies.set_scopes(agent.id, memory_scopes)

        if commit:
            await self._session.commit()

        return agent

    async def validate_blueprint(self, *, user_id: int, blueprint: AgentBlueprint) -> None:
        """Recheck all AI proposed capabilities against backend state."""
        primary = await self._ensure_primary_agent(user_id=user_id)
        if primary.status != AgentStatus.ACTIVE:
            raise ValueError("Parent agent is not active")
        if blueprint.can_spawn_subagents and not primary.can_spawn_subagents:
            raise PermissionError("Parent agent cannot create subagents")
        if blueprint.budget_profile is not None and blueprint.budget_profile != RuntimeBudgets():
            raise ValueError("Custom agent budgets are not supported by the current runtime")
        if not set(blueprint.memory_scopes).issubset(PERSISTENT_AGENT_MEMORY_SCOPES):
            raise ValueError("Requested memory scope is unavailable for persistent agents")
        if len(set(blueprint.memory_scopes)) != len(blueprint.memory_scopes):
            raise ValueError("Duplicate memory scopes")
        available = {
            skill.id: skill
            for skill in await SkillService(self._session).list_available_skills(user_id=user_id)
        }
        if len(set(blueprint.skill_ids)) != len(blueprint.skill_ids):
            raise ValueError("Duplicate skill IDs")
        if not set(blueprint.skill_ids).issubset(available):
            raise ValueError("Skill ID is unavailable to this user")
        permission_keys = [
            (item.action_class, item.resource_scope) for item in blueprint.requested_permissions
        ]
        if len(set(permission_keys)) != len(permission_keys):
            raise ValueError("Duplicate requested permissions")
        for skill_id in blueprint.skill_ids:
            for encoded in available[skill_id].required_permissions:
                action, separator, resource = encoded.partition(":")
                if not separator or not resource:
                    raise ValueError("Selected skill has an invalid required permission")
                try:
                    action_class = ActionClass(action)
                except ValueError as exc:
                    raise ValueError("Selected skill has an invalid required permission") from exc
                if not any(
                    requested.action_class == action_class
                    and PermissionService._scope_matches(requested.resource_scope, resource)
                    for requested in blueprint.requested_permissions
                ):
                    raise PermissionError("Selected skill requires a permission absent from the blueprint")
        for agent_id in blueprint.connected_agent_ids:
            await self.get_agent(user_id=user_id, agent_id=agent_id)
        if blueprint.connected_agent_ids:
            raise ValueError("Persistent worker-to-worker connections are forbidden")
        permissions = PermissionService(self._session)
        for requested in blueprint.requested_permissions:
            if not await permissions.check(
                user_id=user_id,
                subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id=str(primary.id),
                action_class=requested.action_class,
                resource=requested.resource_scope,
            ):
                raise PermissionError("Requested permission exceeds parent permissions")

    async def create_from_blueprint(
        self, *, user_id: int, blueprint: AgentBlueprint, commit: bool = True
    ) -> Agent:
        return await self._create_user_agent(
            user_id=user_id,
            name=blueprint.name,
            role=blueprint.role,
            goal=blueprint.goal,
            backstory=blueprint.backstory,
            custom_instructions=blueprint.custom_instructions,
            can_spawn_subagents=blueprint.can_spawn_subagents,
            blueprint=blueprint,
            commit=commit,
        )

    async def get_agent(self, *, user_id: int, agent_id: UUID) -> Agent:
        agent = await self._agents.get_by_id(
            agent_id
        )

        if agent is None or agent.user_id != user_id:
            raise LookupError(f"Agent not found: {agent_id}")

        return agent

    async def update_prompt(
        self,
        *,
        user_id: int,
        agent_id: UUID,
        role: str,
        goal: str,
        backstory: str | None = None,
        custom_instructions: str | None = None,
    ) -> AgentPromptVersion:
        agent = await self.get_agent(
            user_id=user_id,
            agent_id=agent_id,
        )

        if agent.status != AgentStatus.ACTIVE:
            raise ValueError("Cannot modify inactive agent")

        clean_role = role.strip()
        clean_goal = goal.strip()
        if not clean_role:
            raise ValueError("Agent role cannot be empty")
        if not clean_goal:
            raise ValueError("Agent goal cannot be empty")

        version = await self._prompts.get_next_version(agent_id)

        prompt = await self._prompts.create(
            AgentPromptVersionCreate(
                agent_id=agent_id,
                version=version,
                role=clean_role,
                goal=clean_goal,
                backstory=(
                    backstory.strip()
                    if backstory
                    else None
                ),
                custom_instructions=(
                    custom_instructions.strip()
                    if custom_instructions
                    else None
                ),
            )
        )

        await self._session.commit()

        return prompt

    async def get_current_prompt(
        self,
        *,
        user_id: int,
        agent_id: UUID,
    ) -> AgentPromptVersion:
        await self.get_agent(
            user_id=user_id,
            agent_id=agent_id,
        )

        prompt = await self._prompts.get_latest(
            agent_id
        )

        if prompt is None:
            raise LookupError(
                f"Prompt not found for agent: "
                f"{agent_id}"
            )

        return prompt

    async def archive_agent(self, *, user_id: int, agent_id: UUID) -> Agent:
        agent = await self.get_agent(
            user_id=user_id,
            agent_id=agent_id,
        )

        if agent.kind == AgentKind.PRIMARY:
            raise ValueError(
                "Primary agent cannot be archived"
            )

        updated = await self._agents.update(
            agent_id,
            AgentUpdate(
                status=AgentStatus.ARCHIVED,
            ),
        )

        if updated is None:
            raise LookupError(
                f"Agent not found: {agent_id}"
            )

        await self._session.commit()

        return updated
    
    async def list_active_agents(self, user_id: int) -> list[Agent]:

        agents = await self._agents.list_by_user(user_id)

        return [
            agent
            for agent in agents
            if agent.status == AgentStatus.ACTIVE
        ]

    async def connect_agents(
        self,
        *,
        user_id: int,
        parent_agent_id: UUID,
        child_agent_id: UUID,
    ) -> AgentConnection:
        parent_agent = await self.get_agent(
            user_id=user_id,
            agent_id=parent_agent_id,
        )
        child_agent = await self.get_agent(
            user_id=user_id,
            agent_id=child_agent_id,
        )

        if parent_agent_id == child_agent_id:
            raise ValueError("An agent cannot be connected to itself")

        if parent_agent.kind != AgentKind.PRIMARY:
            raise ValueError("Only primary agent can own persistent connections")
        if child_agent.kind != AgentKind.USER:
            raise ValueError("Only user agents can be connected")

        if parent_agent.status != AgentStatus.ACTIVE:
            raise ValueError("Inactive agent cannot own persistent connections")
        if child_agent.status != AgentStatus.ACTIVE:
            raise ValueError("Cannot connect to inactive agent")

        existing = await self._connections.get(
            parent_agent_id=parent_agent_id,
            child_agent_id=child_agent_id,
        )

        if existing is not None:
            return existing

        connection = await self._connections.create(
            AgentConnectionCreate(
                parent_agent_id=parent_agent_id,
                child_agent_id=child_agent_id,
            )
        )
        await self._session.commit()
        return connection

    async def list_connections(
        self,
        *,
        user_id: int,
        parent_agent_id: UUID,
    ) -> list[AgentConnection]:
        await self.get_agent(
            user_id=user_id,
            agent_id=parent_agent_id,
        )
        return await self._connections.list_children(parent_agent_id)

    async def disconnect_agents(
        self,
        *,
        user_id: int,
        parent_agent_id: UUID,
        child_agent_id: UUID,
    ) -> bool:
        await self.get_agent(
            user_id=user_id,
            agent_id=parent_agent_id,
        )
        await self.get_agent(
            user_id=user_id,
            agent_id=child_agent_id,
        )

        deleted = await self._connections.delete(
            parent_agent_id=parent_agent_id,
            child_agent_id=child_agent_id,
        )
        if deleted:
            await self._session.commit()
        return deleted
