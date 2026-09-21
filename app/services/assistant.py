"""Application entrypoint for a normal primary-agent chat turn."""

from __future__ import annotations

import json
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from agents.assistant.crewai.runtime import DynamicCrewAIRuntime
from agents.protocols import AgentExecutionRuntime, ToolApprovalRuntime
from core.config import config
from models.assistant import AssistantResponse, AssistantResponseStatus
from models.runtime import AgentRuntimeRequest, AgentRuntimeStatus
from repositories.user import UserRepository
from services.agent import AgentService
from services.agent_run import AgentRunService
from services.agent_runtime import AgentRuntimeService
from services.approval import ApprovalService
from services.chat import ChatService
from services.tool_authority import ToolExecutor


_SAFE_FAILURE = "The assistant could not complete this request."


class AssistantService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        runtime: AgentExecutionRuntime | None = None,
        tool_executor: ToolExecutor | None = None,
        approval_runtime: ToolApprovalRuntime | None = None,
    ):
        self._session = session
        self._users = UserRepository(session)
        self._chats = ChatService(session)
        self._agents = AgentService(session)
        self._runs = AgentRunService(session)
        self._approvals = ApprovalService(session)
        self._runtime = runtime or DynamicCrewAIRuntime()
        self._tool_executor = tool_executor
        self._approval_runtime = approval_runtime

    async def handle_message(
        self,
        *,
        user_id: int,
        chat_id: UUID,
        text: str,
    ) -> AssistantResponse:
        if await self._users.get_by_id(user_id) is None:
            raise LookupError(f"User not found: {user_id}")
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Message cannot be empty")
        chat = await self._chats.get_chat(chat_id)
        if chat.user_id != user_id:
            raise LookupError(f"Chat not found: {chat_id}")

        user_message = await self._chats.add_message(
            chat_id=chat_id,
            role="user",
            content=text,
        )
        primary = await self._agents.ensure_primary_agent(user_id=user_id)
        run = await self._runs.create_run(
            user_id=user_id,
            starting_agent_id=primary.id,
            model_name=config.llm.model or "default",
            chat_id=chat_id,
            message_id=user_message.id,
        )
        request = AgentRuntimeRequest(
            run_id=run.id,
            user_id=user_id,
            agent_id=primary.id,
            message=text,
            chat_id=chat_id,
        )
        result = await AgentRuntimeService(
            self._session,
            runtime=self._runtime,
            tool_executor=self._tool_executor,
            approval_runtime=self._approval_runtime,
        ).execute(request)

        if result.status == AgentRuntimeStatus.COMPLETED:
            if result.content is None or not result.content.strip():
                return self._failed(run.id)
            assistant_message = await self._chats.add_message(
                chat_id=chat_id,
                role="assistant",
                content=result.content,
            )
            return AssistantResponse(
                status=AssistantResponseStatus.COMPLETED,
                run_id=run.id,
                message=assistant_message,
            )

        if result.status == AgentRuntimeStatus.WAITING_APPROVAL:
            approval = await self._approvals.get_pending_for_run(
                user_id=user_id,
                run_id=run.id,
            )
            if approval is None:
                return self._failed(run.id)
            requested_approval_id = result.metadata.get("approval_id")
            if requested_approval_id is not None and str(approval.id) != str(requested_approval_id):
                return self._failed(run.id)
            return AssistantResponse(
                status=AssistantResponseStatus.WAITING_APPROVAL,
                run_id=run.id,
                approval_id=approval.id,
                approval_summary=self._approval_summary(approval),
            )

        return self._failed(run.id)

    @staticmethod
    def _approval_summary(approval) -> str:
        arguments = json.dumps(approval.arguments, ensure_ascii=False, sort_keys=True)
        return (
            f"Action: {approval.action_name}\n"
            f"Resource: {approval.resource}\n"
            f"Arguments: {arguments}"
        )

    @staticmethod
    def _failed(run_id: UUID) -> AssistantResponse:
        return AssistantResponse(
            status=AssistantResponseStatus.FAILED,
            run_id=run_id,
            error=_SAFE_FAILURE,
        )
