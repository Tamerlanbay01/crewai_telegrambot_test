from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from models.agent import Agent, AgentCreate, AgentKind, AgentStatus, AgentUpdate
from models.agent_connection import AgentConnection, AgentConnectionCreate
from models.agent_prompt import AgentPromptVersion, AgentPromptVersionCreate
from repositories.agent import AgentRepository
from repositories.agent_connection import AgentConnectionRepository
from repositories.agent_prompt import AgentPromptRepository

class AgentService:
    def __init__(self, session: AsyncSession):
        self._session = session
        self._agents = AgentRepository(session)
        self._prompts = AgentPromptRepository(session)
        self._connections = AgentConnectionRepository(session)

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

    async def ensure_primary_agent(self, *, user_id: int) -> Agent:
        """Return the user's primary agent, creating its initial prompt if needed."""
        agent = await self._ensure_primary_agent(user_id=user_id)
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

        await self._session.commit()

        return agent

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
