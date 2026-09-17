from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from models.agent import Agent, AgentCreate, AgentKind, AgentStatus, AgentUpdate
from models.agent_prompt import AgentPromptVersion, AgentPromptVersionCreate
from repositories.agent import AgentRepository
from repositories. agent_prompt import AgentPromptRepositiry

class AgentService:
    def __init__(self, session: AsyncSession):
        self._session = session
        self._agents = AgentRepository(session)
        self._prompts = AgentPromptRepositiry(session)

    async def ensure_primary_agent(self, *, user_id: int) -> Agent:
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

        version = await self._prompts.get_next_version(agent_id)

        prompt = await self._prompts.create(
            AgentPromptVersionCreate(
                agent_id=agent_id,
                version=version,
                role=role.strip(),
                goal=goal.strip(),
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
    