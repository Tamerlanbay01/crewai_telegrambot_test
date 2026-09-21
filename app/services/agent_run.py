"""Application service enforcing the persisted AgentRun state machine."""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from models.agent import AgentStatus
from models.agent_run import (
    AgentRun,
    AgentRunCreate,
    AgentRunEvent,
    AgentRunEventCreate,
    AgentRunEventType,
    AgentRunStatus,
)
from repositories.agent import AgentRepository
from repositories.agent_prompt import AgentPromptRepository
from repositories.agent_run import AgentRunRepository
from repositories.chat import ChatRepository
from repositories.message import MessageRepository


ALLOWED_TRANSITIONS: dict[AgentRunStatus, set[AgentRunStatus]] = {
    AgentRunStatus.CREATED: {AgentRunStatus.RUNNING, AgentRunStatus.CANCELLED},
    AgentRunStatus.RUNNING: {
        AgentRunStatus.WAITING_APPROVAL,
        AgentRunStatus.COMPLETED,
        AgentRunStatus.FAILED,
        AgentRunStatus.CANCELLED,
    },
    AgentRunStatus.WAITING_APPROVAL: {AgentRunStatus.RUNNING, AgentRunStatus.CANCELLED},
    AgentRunStatus.COMPLETED: set(),
    AgentRunStatus.FAILED: set(),
    AgentRunStatus.CANCELLED: set(),
}


class AgentRunService:
    def __init__(self, session: AsyncSession):
        self._session = session
        self._agents = AgentRepository(session)
        self._prompts = AgentPromptRepository(session)
        self._runs = AgentRunRepository(session)
        self._chats = ChatRepository(session)
        self._messages = MessageRepository(session)

    async def create_run(
        self,
        *,
        user_id: int,
        starting_agent_id: UUID,
        model_name: str,
        chat_id: UUID | None = None,
        message_id: UUID | None = None,
    ) -> AgentRun:
        agent = await self._agents.get_by_id(starting_agent_id)
        if agent is None or agent.user_id != user_id:
            raise LookupError(f"Agent not found: {starting_agent_id}")
        if agent.status != AgentStatus.ACTIVE:
            raise ValueError("Cannot start an inactive agent")
        prompt = await self._prompts.get_latest(starting_agent_id)
        if prompt is None:
            raise LookupError(f"Prompt not found for agent: {starting_agent_id}")
        if message_id is not None:
            message = await self._messages.get_by_id(message_id)
            if message is None:
                raise LookupError(f"Message not found: {message_id}")
            if chat_id is not None and chat_id != message.chat_id:
                raise ValueError("Message does not belong to the supplied chat")
            chat_id = message.chat_id
        if chat_id is not None:
            chat = await self._chats.get_by_id(chat_id)
            if chat is None or chat.user_id != user_id:
                raise LookupError(f"Chat not found: {chat_id}")
        run = await self._runs.create(
            AgentRunCreate(
                user_id=user_id,
                starting_agent_id=starting_agent_id,
                prompt_version_id=prompt.id,
                model_name=model_name,
                chat_id=chat_id,
                message_id=message_id,
            )
        )
        await self._runs.append_event(
            AgentRunEventCreate(
                run_id=run.id,
                event_type=AgentRunEventType.CREATED,
                to_status=AgentRunStatus.CREATED,
            )
        )
        await self._session.commit()
        return run

    async def get_run(self, *, user_id: int, run_id: UUID) -> AgentRun:
        run = await self._runs.get_by_id(run_id)
        if run is None or run.user_id != user_id:
            raise LookupError(f"Agent run not found: {run_id}")
        return run

    async def list_events(self, *, user_id: int, run_id: UUID) -> list[AgentRunEvent]:
        await self.get_run(user_id=user_id, run_id=run_id)
        return await self._runs.list_events(run_id)

    async def start(self, *, user_id: int, run_id: UUID) -> AgentRun:
        return await self._transition(user_id=user_id, run_id=run_id, target=AgentRunStatus.RUNNING)

    async def wait_for_approval(self, *, user_id: int, run_id: UUID) -> AgentRun:
        return await self._transition(
            user_id=user_id, run_id=run_id, target=AgentRunStatus.WAITING_APPROVAL
        )

    async def resume(self, *, user_id: int, run_id: UUID) -> AgentRun:
        return await self._transition(user_id=user_id, run_id=run_id, target=AgentRunStatus.RUNNING)

    async def complete(
        self, *, user_id: int, run_id: UUID, result_metadata: dict[str, object] | None = None,
        usage: dict[str, object] | None = None,
    ) -> AgentRun:
        return await self._transition(
            user_id=user_id, run_id=run_id, target=AgentRunStatus.COMPLETED,
            result_metadata=result_metadata, usage=usage,
        )

    async def fail(self, *, user_id: int, run_id: UUID, error: str) -> AgentRun:
        if not error.strip():
            raise ValueError("Run error cannot be empty")
        return await self._transition(
            user_id=user_id, run_id=run_id, target=AgentRunStatus.FAILED, error=error.strip()
        )

    async def cancel(self, *, user_id: int, run_id: UUID) -> AgentRun:
        return await self._transition(user_id=user_id, run_id=run_id, target=AgentRunStatus.CANCELLED)

    async def _transition(
        self,
        *,
        user_id: int,
        run_id: UUID,
        target: AgentRunStatus,
        error: str | None = None,
        result_metadata: dict[str, object] | None = None,
        usage: dict[str, object] | None = None,
    ) -> AgentRun:
        run = await self.get_run(user_id=user_id, run_id=run_id)
        if target not in ALLOWED_TRANSITIONS[run.status]:
            raise ValueError(f"Invalid run transition: {run.status} -> {target}")
        now = datetime.now(timezone.utc)
        updated = await self._runs.update_status(
            run_id=run_id,
            status=target,
            error=error,
            result_metadata=result_metadata,
            usage=usage,
            started_at=now if target == AgentRunStatus.RUNNING and run.started_at is None else None,
            completed_at=now if target in {AgentRunStatus.COMPLETED, AgentRunStatus.FAILED, AgentRunStatus.CANCELLED} else None,
        )
        if updated is None:
            raise LookupError(f"Agent run not found: {run_id}")
        await self._runs.append_event(
            AgentRunEventCreate(
                run_id=run_id,
                event_type=AgentRunEventType.STATUS_CHANGED,
                from_status=run.status,
                to_status=target,
            )
        )
        await self._session.commit()
        return updated
