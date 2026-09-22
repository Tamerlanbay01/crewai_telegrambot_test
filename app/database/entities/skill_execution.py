"""SQLAlchemy entity for one isolated executable-skill invocation."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import BigInteger, CheckConstraint, DateTime, Enum, ForeignKey, Index, Integer, JSON, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base
from models.permission import PermissionSubjectType
from models.skill import SkillExecutionStatus


class SkillExecutionEntity(Base):
    __tablename__ = "skill_executions"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    skill_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("skills.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    skill_version: Mapped[int] = mapped_column(Integer, nullable=False)
    requesting_subject_type: Mapped[PermissionSubjectType] = mapped_column(
        Enum(
            PermissionSubjectType,
            name="skill_execution_subject_type",
            values_callable=lambda enum: [item.value for item in enum],
        ),
        nullable=False,
    )
    requesting_subject_id: Mapped[str] = mapped_column(String(255), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    status: Mapped[SkillExecutionStatus] = mapped_column(
        Enum(
            SkillExecutionStatus,
            name="skill_execution_status",
            values_callable=lambda enum: [item.value for item in enum],
        ),
        nullable=False,
        default=SkillExecutionStatus.PENDING,
    )
    arguments_sanitized: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stdout_preview: Mapped[str] = mapped_column(Text, nullable=False, default="")
    stderr_preview: Mapped[str] = mapped_column(Text, nullable=False, default="")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint("skill_version > 0", name="ck_skill_executions_version_positive"),
        Index("ix_skill_executions_run_created", "run_id", "created_at"),
    )
