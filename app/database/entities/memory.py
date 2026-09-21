"""SQLAlchemy entity for persisted, scoped memory records."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    JSON,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base
from models.memory import MemoryScope


class MemoryEntity(Base):
    __tablename__ = "memories"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
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
        PGUUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    run_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    memory_metadata: Mapped[dict[str, object]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "(scope = 'USER_GLOBAL' AND agent_id IS NULL AND run_id IS NULL) "
            "OR (scope = 'AGENT_PRIVATE' AND agent_id IS NOT NULL AND run_id IS NULL) "
            "OR (scope IN ('CREW_SHARED', 'RUN_EPHEMERAL') AND run_id IS NOT NULL)",
            name="ck_memories_scope_references",
        ),
        Index(
            "uq_memory_user_global_key",
            "user_id",
            "key",
            unique=True,
            postgresql_where=text("scope = 'USER_GLOBAL'"),
            sqlite_where=text("scope = 'USER_GLOBAL'"),
        ),
        Index(
            "uq_memory_agent_private_key",
            "user_id",
            "agent_id",
            "key",
            unique=True,
            postgresql_where=text("scope = 'AGENT_PRIVATE'"),
            sqlite_where=text("scope = 'AGENT_PRIVATE'"),
        ),
        Index(
            "uq_memory_crew_shared_key",
            "user_id",
            "run_id",
            "key",
            unique=True,
            postgresql_where=text("scope = 'CREW_SHARED'"),
            sqlite_where=text("scope = 'CREW_SHARED'"),
        ),
        Index(
            "uq_memory_run_ephemeral_key",
            "user_id",
            "run_id",
            "key",
            unique=True,
            postgresql_where=text("scope = 'RUN_EPHEMERAL'"),
            sqlite_where=text("scope = 'RUN_EPHEMERAL'"),
        ),
    )
