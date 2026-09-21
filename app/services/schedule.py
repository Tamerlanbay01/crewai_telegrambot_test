"""Schedule validation, lifecycle, and scheduled AgentRun coordination."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone as dt_timezone
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter
from sqlalchemy.ext.asyncio import AsyncSession

from agents.assistant.crewai.runtime import DynamicCrewAIRuntime
from agents.protocols import AgentExecutionRuntime, ToolApprovalRuntime
from core.config import config
from models.agent import Agent, AgentStatus
from models.runtime import AgentRuntimeRequest, AgentRuntimeStatus
from models.schedule import (
    Schedule,
    ScheduleCreate,
    ScheduleStatus,
    ScheduleType,
    ScheduleUpdate,
    ScheduledExecutionResult,
)
from repositories.agent import AgentRepository
from repositories.chat import ChatRepository
from repositories.schedule import ScheduleRepository
from repositories.user import UserRepository
from services.agent_run import AgentRunService
from services.agent_runtime import AgentRuntimeService
from services.tool_authority import ToolExecutor


_UTC = dt_timezone.utc
_CLAIM_LEASE = timedelta(minutes=10)


class ScheduleService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        runtime: AgentExecutionRuntime | None = None,
        tool_executor: ToolExecutor | None = None,
        approval_runtime: ToolApprovalRuntime | None = None,
    ):
        self._session = session
        self._schedules = ScheduleRepository(session)
        self._users = UserRepository(session)
        self._agents = AgentRepository(session)
        self._chats = ChatRepository(session)
        self._runs = AgentRunService(session)
        self._runtime = runtime
        self._tool_executor = tool_executor
        self._approval_runtime = approval_runtime

    async def create(
        self,
        *,
        user_id: int,
        agent_id: UUID,
        name: str,
        prompt: str,
        schedule_type: ScheduleType,
        schedule_expression: str,
        timezone: str = "UTC",
        chat_id: UUID | None = None,
    ) -> Schedule:
        await self._require_user(user_id)
        await self._require_active_agent(user_id=user_id, agent_id=agent_id)
        if chat_id is not None:
            await self._require_chat(user_id=user_id, chat_id=chat_id)
        clean_name = self._required_text(name, "Schedule name")
        clean_prompt = self._required_text(prompt, "Schedule prompt")
        clean_expression = self._required_text(schedule_expression, "Schedule expression")
        zone = self._resolve_timezone(timezone)
        next_run_at = self._next_run(schedule_type, clean_expression, zone, self._now())
        schedule = await self._schedules.create(
            ScheduleCreate(
                user_id=user_id,
                agent_id=agent_id,
                chat_id=chat_id,
                name=clean_name,
                prompt=clean_prompt,
                schedule_type=schedule_type,
                schedule_expression=clean_expression,
                timezone=zone.key,
                status=ScheduleStatus.ACTIVE,
                next_run_at=next_run_at,
            )
        )
        await self._session.commit()
        return schedule

    async def get(self, *, user_id: int, schedule_id: UUID) -> Schedule:
        schedule = await self._schedules.get_by_id(schedule_id)
        if schedule is None or schedule.user_id != user_id:
            raise LookupError(f"Schedule not found: {schedule_id}")
        return schedule

    async def list(self, *, user_id: int) -> list[Schedule]:
        await self._require_user(user_id)
        return await self._schedules.list_by_user(user_id)

    async def update(
        self, *, user_id: int, schedule_id: UUID, data: ScheduleUpdate
    ) -> Schedule:
        schedule = await self.get(user_id=user_id, schedule_id=schedule_id)
        if schedule.status == ScheduleStatus.ARCHIVED:
            raise ValueError("Archived schedules cannot be updated")
        if self._claim_is_live(schedule, self._now()):
            raise ValueError("Cannot edit a schedule while it is being executed")

        changes = data.model_dump(exclude_unset=True)
        if not changes:
            return schedule
        if "name" in changes:
            changes["name"] = self._required_text(changes["name"], "Schedule name")
        if "prompt" in changes:
            changes["prompt"] = self._required_text(changes["prompt"], "Schedule prompt")
        if "schedule_type" in changes and changes["schedule_type"] is None:
            raise ValueError("Schedule type cannot be empty")
        if "agent_id" in changes:
            if changes["agent_id"] is None:
                raise ValueError("Schedule agent_id cannot be empty")
            await self._require_active_agent(user_id=user_id, agent_id=changes["agent_id"])
        if "chat_id" in changes and changes["chat_id"] is not None:
            await self._require_chat(user_id=user_id, chat_id=changes["chat_id"])
        if "schedule_expression" in changes:
            changes["schedule_expression"] = self._required_text(
                changes["schedule_expression"], "Schedule expression"
            )
        if "timezone" in changes:
            if changes["timezone"] is None:
                raise ValueError("Schedule timezone cannot be empty")
            changes["timezone"] = self._resolve_timezone(changes["timezone"]).key

        schedule_type = changes.get("schedule_type", schedule.schedule_type)
        expression = changes.get("schedule_expression", schedule.schedule_expression)
        timezone_name = changes.get("timezone", schedule.timezone)
        if {"schedule_type", "schedule_expression", "timezone"}.intersection(changes):
            changes["next_run_at"] = self._next_run(
                schedule_type,
                expression,
                self._resolve_timezone(timezone_name),
                self._now(),
            )
            changes["status"] = ScheduleStatus.ACTIVE

        updated = await self._schedules.update(
            schedule_id,
            ScheduleUpdate(**{key: value for key, value in changes.items() if key in ScheduleUpdate.model_fields}),
        )
        if updated is None:
            raise LookupError(f"Schedule not found: {schedule_id}")
        if "next_run_at" in changes:
            await self._schedules.update_state(
                schedule_id,
                status=changes["status"],
                next_run_at=changes["next_run_at"],
                last_run_at=schedule.last_run_at,
                now=self._now(),
            )
            updated = await self.get(user_id=user_id, schedule_id=schedule_id)
        await self._session.commit()
        return updated

    async def enable(self, *, user_id: int, schedule_id: UUID) -> Schedule:
        schedule = await self.get(user_id=user_id, schedule_id=schedule_id)
        if schedule.status == ScheduleStatus.ARCHIVED:
            raise ValueError("Cannot enable an archived schedule")
        if schedule.status == ScheduleStatus.ACTIVE and schedule.next_run_at is not None:
            return schedule
        await self._require_active_agent(user_id=user_id, agent_id=schedule.agent_id)
        if schedule.chat_id is not None:
            await self._require_chat(user_id=user_id, chat_id=schedule.chat_id)
        next_run_at = self._next_run(
            schedule.schedule_type,
            schedule.schedule_expression,
            self._resolve_timezone(schedule.timezone),
            self._now(),
        )
        enabled = await self._schedules.update_state(
            schedule_id,
            status=ScheduleStatus.ACTIVE,
            next_run_at=next_run_at,
            last_run_at=schedule.last_run_at,
            now=self._now(),
        )
        if enabled is None:
            raise LookupError(f"Schedule not found: {schedule_id}")
        await self._session.commit()
        return enabled

    async def disable(self, *, user_id: int, schedule_id: UUID) -> Schedule:
        schedule = await self.get(user_id=user_id, schedule_id=schedule_id)
        if schedule.status == ScheduleStatus.ARCHIVED:
            return schedule
        if schedule.status == ScheduleStatus.DISABLED and schedule.claim_token is None:
            return schedule
        disabled = await self._schedules.update_state(
            schedule_id,
            status=ScheduleStatus.DISABLED,
            next_run_at=schedule.next_run_at,
            last_run_at=schedule.last_run_at,
            now=self._now(),
        )
        if disabled is None:
            raise LookupError(f"Schedule not found: {schedule_id}")
        await self._session.commit()
        return disabled

    async def archive(self, *, user_id: int, schedule_id: UUID) -> Schedule:
        schedule = await self.get(user_id=user_id, schedule_id=schedule_id)
        if schedule.status == ScheduleStatus.ARCHIVED:
            return schedule
        archived = await self._schedules.update_state(
            schedule_id,
            status=ScheduleStatus.ARCHIVED,
            next_run_at=None,
            last_run_at=schedule.last_run_at,
            now=self._now(),
        )
        if archived is None:
            raise LookupError(f"Schedule not found: {schedule_id}")
        await self._session.commit()
        return archived

    async def get_due(self, *, now: datetime | None = None) -> list[Schedule]:
        return await self._schedules.list_due(self._as_utc(now or self._now()))

    async def mark_executed(
        self,
        *,
        user_id: int,
        schedule_id: UUID,
        claim_token: UUID | None = None,
        executed_at: datetime | None = None,
    ) -> Schedule:
        schedule = await self.get(user_id=user_id, schedule_id=schedule_id)
        at = self._as_utc(executed_at or self._now())
        if schedule.status != ScheduleStatus.ACTIVE:
            raise ValueError("Only active schedules can be marked executed")
        if claim_token is None and schedule.claim_token is not None:
            raise ValueError("Schedule execution is claimed by another worker")
        if claim_token is not None and schedule.claim_token != claim_token:
            raise ValueError("Schedule execution claim is no longer valid")

        next_run_at, status = self._after_execution(schedule, at)
        if claim_token is not None:
            updated = await self._schedules.finish_claim(
                schedule_id=schedule_id,
                token=claim_token,
                status=status,
                next_run_at=next_run_at,
                last_run_at=at,
                now=self._now(),
            )
        else:
            updated = await self._schedules.update_state(
                schedule_id,
                status=status,
                next_run_at=next_run_at,
                last_run_at=at,
                now=self._now(),
            )
        if updated is None:
            raise ValueError("Schedule execution claim is no longer valid")
        await self._session.commit()
        return updated

    async def execute_due(
        self,
        *,
        user_id: int,
        schedule_id: UUID,
        now: datetime | None = None,
    ) -> ScheduledExecutionResult | None:
        claim_time = self._as_utc(now or self._now())
        schedule = await self._schedules.get_by_id(schedule_id)
        if schedule is None or schedule.user_id != user_id:
            raise LookupError(f"Schedule not found: {schedule_id}")
        if (
            schedule.status != ScheduleStatus.ACTIVE
            or schedule.next_run_at is None
            or self._as_utc(schedule.next_run_at) > claim_time
        ):
            return None

        token = uuid4()
        claimed = await self._schedules.claim_due(
            schedule_id=schedule_id,
            token=token,
            now=claim_time,
            expires_at=claim_time + _CLAIM_LEASE,
        )
        if claimed is None:
            return None
        await self._session.commit()

        run = None
        runtime_result = None
        try:
            await self._require_user(user_id)
            await self._require_active_agent(user_id=user_id, agent_id=claimed.agent_id)
            if claimed.chat_id is not None:
                await self._require_chat(user_id=user_id, chat_id=claimed.chat_id)
            run = await self._runs.create_run(
                user_id=user_id,
                starting_agent_id=claimed.agent_id,
                model_name=config.llm.model or "default",
                chat_id=claimed.chat_id,
            )
            runtime_result = await AgentRuntimeService(
                self._session,
                runtime=self._runtime or DynamicCrewAIRuntime(),
                tool_executor=self._tool_executor,
                approval_runtime=self._approval_runtime,
            ).execute(
                AgentRuntimeRequest(
                    run_id=run.id,
                    user_id=user_id,
                    agent_id=claimed.agent_id,
                    message=claimed.prompt,
                    chat_id=claimed.chat_id,
                )
            )
            updated_run = await self._runs.get_run(
                user_id=user_id,
                run_id=run.id,
            )
        except Exception:
            await self.mark_executed(
                user_id=user_id,
                schedule_id=schedule_id,
                claim_token=token,
                executed_at=claim_time if now is not None else self._now(),
            )
            raise

        finished_at = claim_time if now is not None else self._now()
        updated_schedule = await self.mark_executed(
            user_id=user_id,
            schedule_id=schedule_id,
            claim_token=token,
            executed_at=finished_at,
        )
        approval_id = runtime_result.metadata.get("approval_id")
        try:
            parsed_approval_id = UUID(str(approval_id)) if approval_id is not None else None
        except ValueError:
            parsed_approval_id = None
        return ScheduledExecutionResult(
            schedule=updated_schedule,
            run_id=run.id,
            run_status=updated_run.status,
            runtime_status=runtime_result.status,
            content=runtime_result.content,
            error=runtime_result.error,
            approval_id=parsed_approval_id,
        )

    async def _require_user(self, user_id: int) -> None:
        if await self._users.get_by_id(user_id) is None:
            raise LookupError(f"User not found: {user_id}")

    async def _require_active_agent(self, *, user_id: int, agent_id: UUID) -> Agent:
        agent = await self._agents.get_by_id(agent_id)
        if agent is None or agent.user_id != user_id:
            raise LookupError(f"Agent not found: {agent_id}")
        if agent.status != AgentStatus.ACTIVE:
            raise ValueError("Schedule requires an active agent")
        return agent

    async def _require_chat(self, *, user_id: int, chat_id: UUID) -> None:
        chat = await self._chats.get_by_id(chat_id)
        if chat is None or chat.user_id != user_id:
            raise LookupError(f"Chat not found: {chat_id}")

    @classmethod
    def _next_run(
        cls,
        schedule_type: ScheduleType,
        expression: str,
        zone: ZoneInfo,
        now: datetime,
    ) -> datetime:
        base = cls._as_utc(now)
        if schedule_type == ScheduleType.ONCE:
            try:
                value = datetime.fromisoformat(expression.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("ONCE expression must be an ISO datetime") from exc
            if value.tzinfo is None:
                value = value.replace(tzinfo=zone)
                if value.astimezone(_UTC).astimezone(zone).replace(tzinfo=None) != value.replace(tzinfo=None):
                    raise ValueError("ONCE expression is a nonexistent local time in the selected timezone")
            return value.astimezone(_UTC)
        if schedule_type == ScheduleType.INTERVAL:
            try:
                seconds = int(expression)
            except ValueError as exc:
                raise ValueError("INTERVAL expression must be a positive number of seconds") from exc
            if seconds <= 0:
                raise ValueError("INTERVAL expression must be a positive number of seconds")
            return base + timedelta(seconds=seconds)
        if schedule_type == ScheduleType.CRON:
            if len(expression.split()) != 5 or not croniter.is_valid(expression, strict=True):
                raise ValueError("CRON expression must be a valid five-field cron expression")
            local_base = base.astimezone(zone)
            try:
                next_local = croniter(expression, local_base).get_next(datetime)
            except Exception as exc:
                raise ValueError("Could not calculate the next CRON occurrence") from exc
            if next_local.tzinfo is None:
                next_local = next_local.replace(tzinfo=zone)
            return next_local.astimezone(_UTC)
        raise ValueError(f"Unsupported schedule type: {schedule_type}")

    @classmethod
    def _after_execution(
        cls, schedule: Schedule, executed_at: datetime
    ) -> tuple[datetime | None, ScheduleStatus]:
        if schedule.schedule_type == ScheduleType.ONCE:
            return None, ScheduleStatus.DISABLED
        zone = cls._resolve_timezone(schedule.timezone)
        next_run_at = cls._next_run(
            schedule.schedule_type,
            schedule.schedule_expression,
            zone,
            executed_at,
        )
        return next_run_at, ScheduleStatus.ACTIVE

    @staticmethod
    def _resolve_timezone(name: str) -> ZoneInfo:
        if not isinstance(name, str):
            raise ValueError("Schedule timezone cannot be empty")
        clean = name.strip()
        if not clean:
            raise ValueError("Schedule timezone cannot be empty")
        try:
            return ZoneInfo(clean)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Invalid timezone: {name}") from exc

    @staticmethod
    def _required_text(value: str | None, label: str) -> str:
        if value is None or not value.strip():
            raise ValueError(f"{label} cannot be empty")
        return value.strip()

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            value = value.replace(tzinfo=_UTC)
        return value.astimezone(_UTC)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(_UTC)

    @classmethod
    def _claim_is_live(cls, schedule: Schedule, now: datetime) -> bool:
        return (
            schedule.claim_token is not None
            and schedule.claim_expires_at is not None
            and cls._as_utc(schedule.claim_expires_at) > cls._as_utc(now)
        )
