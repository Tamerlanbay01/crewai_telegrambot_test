"""Auditable memory proposals produced by the system Memory Agent."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base
from models.memory import MemoryCandidateDecision, MemoryScope, MemoryType


class MemoryCandidateEntity(Base):
    __tablename__ = "memory_candidates"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    extraction_run_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("memory_extraction_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    memory_type: Mapped[MemoryType] = mapped_column(
        Enum(
            MemoryType,
            name="memory_type",
            values_callable=lambda enum: [item.value for item in enum],
        ),
        nullable=False,
    )
    scope: Mapped[MemoryScope] = mapped_column(
        Enum(
            MemoryScope,
            name="memory_scope",
            values_callable=lambda enum: [item.value for item in enum],
        ),
        nullable=False,
    )
    agent_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("agents.id", ondelete="SET NULL"), nullable=True
    )
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    explicit_user_request: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    decision: Mapped[MemoryCandidateDecision] = mapped_column(
        Enum(
            MemoryCandidateDecision,
            name="memory_candidate_decision",
            values_callable=lambda enum: [item.value for item in enum],
        ),
        nullable=False,
        default=MemoryCandidateDecision.PENDING,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
