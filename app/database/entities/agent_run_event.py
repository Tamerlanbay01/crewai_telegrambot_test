"""Immutable audit events for agent runs."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Enum, ForeignKey, JSON, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base
from models.agent_run import AgentRunEventType, AgentRunStatus


class AgentRunEventEntity(Base):
    __tablename__ = "agent_run_events"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type: Mapped[AgentRunEventType] = mapped_column(
        Enum(
            AgentRunEventType,
            name="agent_run_event_type",
            values_callable=lambda enum: [item.value for item in enum],
        ),
        nullable=False,
    )
    from_status: Mapped[AgentRunStatus | None] = mapped_column(
        Enum(
            AgentRunStatus,
            name="agent_run_status",
            values_callable=lambda enum: [item.value for item in enum],
        ),
        nullable=True,
    )
    to_status: Mapped[AgentRunStatus | None] = mapped_column(
        Enum(
            AgentRunStatus,
            name="agent_run_status",
            values_callable=lambda enum: [item.value for item in enum],
        ),
        nullable=True,
    )
    payload: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
