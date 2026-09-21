"""SQLAlchemy entity for metadata about a platform or user skill."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base
from models.skill import SkillOwnerType, SkillStatus


class SkillEntity(Base):
    __tablename__ = "skills"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    owner_type: Mapped[SkillOwnerType] = mapped_column(
        Enum(
            SkillOwnerType,
            name="skill_owner_type",
            values_callable=lambda enum: [item.value for item in enum],
        ),
        nullable=False,
    )
    owner_user_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    key: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[SkillStatus] = mapped_column(
        Enum(
            SkillStatus,
            name="skill_status",
            values_callable=lambda enum: [item.value for item in enum],
        ),
        nullable=False,
        default=SkillStatus.ACTIVE,
    )
    storage_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    manifest: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict)
    required_permissions: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            "(owner_type = 'SYSTEM' AND owner_user_id IS NULL) "
            "OR (owner_type = 'USER' AND owner_user_id IS NOT NULL)",
            name="ck_skills_owner_matches_owner_type",
        ),
        CheckConstraint("version > 0", name="ck_skills_version_positive"),
        Index(
            "uq_system_skill_key_version",
            "key",
            "version",
            unique=True,
            postgresql_where=text("owner_type = 'SYSTEM'"),
            sqlite_where=text("owner_type = 'SYSTEM'"),
        ),
        Index(
            "uq_user_skill_key_version",
            "owner_user_id",
            "key",
            "version",
            unique=True,
            postgresql_where=text("owner_type = 'USER'"),
            sqlite_where=text("owner_type = 'USER'"),
        ),
    )
