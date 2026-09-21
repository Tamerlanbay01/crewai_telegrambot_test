"""Database entity for one-action approvals."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, JSON, String, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base
from models.approval import ApprovalStatus
from models.permission import ActionClass, PermissionSubjectType


class ApprovalEntity(Base):
    __tablename__ = "approvals"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    run_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    flow_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    requesting_subject_type: Mapped[PermissionSubjectType] = mapped_column(
        Enum(
            PermissionSubjectType,
            name="approval_subject_type",
            values_callable=lambda enum: [item.value for item in enum],
        ),
        nullable=False,
    )
    requesting_subject_id: Mapped[str] = mapped_column(String(255), nullable=False)
    action_name: Mapped[str] = mapped_column(String(255), nullable=False)
    action_class: Mapped[ActionClass] = mapped_column(
        Enum(
            ActionClass,
            name="approval_action_class",
            values_callable=lambda enum: [item.value for item in enum],
        ),
        nullable=False,
    )
    resource: Mapped[str] = mapped_column(String(500), nullable=False)
    arguments: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[ApprovalStatus] = mapped_column(
        Enum(
            ApprovalStatus,
            name="approval_status",
            values_callable=lambda enum: [item.value for item in enum],
        ),
        nullable=False,
        default=ApprovalStatus.PENDING,
    )
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decision_metadata: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict
    )
