"""Database entity for backend-enforced permissions."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import BigInteger, Boolean, DateTime, Enum, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base
from models.permission import ActionClass, PermissionSubjectType


class PermissionEntity(Base):
    __tablename__ = "permissions"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    subject_type: Mapped[PermissionSubjectType] = mapped_column(
        Enum(
            PermissionSubjectType,
            name="permission_subject_type",
            values_callable=lambda enum: [item.value for item in enum],
        ),
        nullable=False,
    )
    subject_id: Mapped[str] = mapped_column(String(255), nullable=False)
    action_class: Mapped[ActionClass] = mapped_column(
        Enum(
            ActionClass,
            name="permission_action_class",
            values_callable=lambda enum: [item.value for item in enum],
        ),
        nullable=False,
    )
    resource_scope: Mapped[str] = mapped_column(String(500), nullable=False)
    allowed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "subject_type",
            "subject_id",
            "action_class",
            "resource_scope",
            name="uq_permission_subject_action_scope",
        ),
    )
