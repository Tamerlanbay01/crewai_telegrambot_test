"""Backend-owned approval state machine for one concrete action."""

import json
import re
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from models.approval import Approval, ApprovalCreate, ApprovalStatus
from models.agent_run import AgentRunEventCreate, AgentRunEventType, AgentRunStatus
from models.permission import ActionClass, PermissionSubjectType
from repositories.agent_run import AgentRunRepository
from repositories.approval import ApprovalRepository
from services.agent_run import AgentRunService


class ApprovalAlreadyProcessedError(ValueError):
    """Raised when a callback attempts to decide a non-pending approval."""


_SENSITIVE_KEY_FRAGMENTS = (
    "authorization",
    "password",
    "secret",
    "token",
    "apikey",
    "privatekey",
    "credential",
    "bearer",
    "cookie",
)


class ApprovalService:
    def __init__(self, session: AsyncSession):
        self._session = session
        self._approvals = ApprovalRepository(session)
        self._runs = AgentRunRepository(session)
        self._run_service = AgentRunService(session)

    async def request(
        self,
        *,
        user_id: int,
        run_id: UUID,
        flow_id: str | None = None,
        requesting_subject_type: PermissionSubjectType,
        requesting_subject_id: str,
        action_name: str,
        action_class: ActionClass,
        arguments: dict[str, object],
        resource: str | None = None,
        expires_at: datetime | None = None,
    ) -> Approval:
        run = await self._run_service.get_run(user_id=user_id, run_id=run_id)
        if run.status != AgentRunStatus.RUNNING:
            raise ValueError("Approval can only be requested for a running run")
        if action_class == ActionClass.READ:
            raise ValueError("READ actions do not require approval")
        clean_action_name = action_name.strip()
        if not clean_action_name:
            raise ValueError("Approval action name cannot be empty")
        sanitized_arguments = self._sanitize(arguments)
        approval = await self._approvals.create(
            ApprovalCreate(
                user_id=user_id,
                run_id=run_id,
                flow_id=flow_id,
                requesting_subject_type=requesting_subject_type,
                requesting_subject_id=requesting_subject_id,
                action_name=clean_action_name,
                action_class=action_class,
                resource=(resource or clean_action_name).strip(),
                arguments=sanitized_arguments,
                expires_at=expires_at,
            )
        )
        await self._runs.append_event(
            AgentRunEventCreate(
                run_id=run_id,
                event_type=AgentRunEventType.APPROVAL_REQUESTED,
                payload={
                    "approval_id": str(approval.id),
                    "flow_id": flow_id,
                    "subject_type": requesting_subject_type.value,
                    "subject_id": requesting_subject_id,
                    "action_name": clean_action_name,
                    "action_class": action_class.value,
                    "resource": (resource or clean_action_name).strip(),
                },
            )
        )
        await self._run_service.wait_for_approval(user_id=user_id, run_id=run_id)
        return approval

    async def get(self, *, user_id: int, approval_id: UUID) -> Approval:
        approval = await self._approvals.get_by_id(approval_id)
        if approval is None or approval.user_id != user_id:
            raise LookupError(f"Approval not found: {approval_id}")
        return approval

    async def get_pending_for_run(self, *, user_id: int, run_id: UUID) -> Approval | None:
        await self._run_service.get_run(user_id=user_id, run_id=run_id)
        approval = await self._approvals.get_pending_for_run(run_id)
        if approval is None or approval.user_id != user_id:
            return None
        return approval

    async def approve(
        self,
        *,
        user_id: int,
        approval_id: UUID,
        decision_metadata: dict[str, object] | None = None,
    ) -> Approval:
        return await self._decide(
            user_id=user_id,
            approval_id=approval_id,
            status=ApprovalStatus.APPROVED,
            event_type=AgentRunEventType.APPROVAL_APPROVED,
            decision_metadata=decision_metadata,
        )

    async def reject(
        self,
        *,
        user_id: int,
        approval_id: UUID,
        decision_metadata: dict[str, object] | None = None,
    ) -> Approval:
        return await self._decide(
            user_id=user_id,
            approval_id=approval_id,
            status=ApprovalStatus.REJECTED,
            event_type=AgentRunEventType.APPROVAL_REJECTED,
            decision_metadata=decision_metadata,
        )

    async def cancel(
        self,
        *,
        user_id: int,
        approval_id: UUID,
        decision_metadata: dict[str, object] | None = None,
    ) -> Approval:
        return await self._decide(
            user_id=user_id,
            approval_id=approval_id,
            status=ApprovalStatus.CANCELLED,
            event_type=AgentRunEventType.APPROVAL_CANCELLED,
            decision_metadata=decision_metadata,
        )

    async def _decide(
        self,
        *,
        user_id: int,
        approval_id: UUID,
        status: ApprovalStatus,
        event_type: AgentRunEventType,
        decision_metadata: dict[str, object] | None,
    ) -> Approval:
        approval = await self.get(user_id=user_id, approval_id=approval_id)
        if approval.status != ApprovalStatus.PENDING:
            raise ApprovalAlreadyProcessedError(
                f"Approval already decided: {approval.status.value}"
            )
        now = datetime.now(timezone.utc)
        if approval.expires_at is not None and approval.expires_at <= now:
            status = ApprovalStatus.EXPIRED
            event_type = AgentRunEventType.APPROVAL_EXPIRED
        decided = await self._approvals.decide(
            approval_id=approval.id,
            status=status,
            decided_at=now,
            decision_metadata=self._sanitize(decision_metadata or {}),
        )
        if decided is None:
            raise ApprovalAlreadyProcessedError("Approval already processed")
        await self._runs.append_event(
            AgentRunEventCreate(
                run_id=approval.run_id,
                event_type=event_type,
                payload={
                    "approval_id": str(approval.id),
                    "status": status.value,
                },
            )
        )
        await self._run_service.resume(user_id=user_id, run_id=approval.run_id)
        return decided

    @classmethod
    def _sanitize(cls, value: dict[str, object]) -> dict[str, object]:
        def redact(item: object) -> object:
            if isinstance(item, dict):
                return {
                    str(key): "[REDACTED]" if cls._sensitive_key(key) else redact(nested)
                    for key, nested in item.items()
                }
            if isinstance(item, list):
                return [redact(nested) for nested in item]
            return item

        sanitized = redact(value)
        serialized = json.dumps(sanitized, ensure_ascii=False)
        result = json.loads(serialized)
        if not isinstance(result, dict):
            raise ValueError("Approval metadata must be a JSON object")
        return result

    @staticmethod
    def _sensitive_key(key: object) -> bool:
        normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
        return any(fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS)
