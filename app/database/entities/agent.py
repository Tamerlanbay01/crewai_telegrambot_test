from datetime import datetime
from uuid import UUID

from sqlalchemy import BigInteger, Boolean, DateTime, Enum, ForeignKey, Index, String, func, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base
from models.agent import AgentKind, AgentStatus


class AgentEntity(Base):
    __tablename__ = "agents"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
    )

    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    kind: Mapped[AgentKind] = mapped_column(
        Enum(
            AgentKind,
            name="agent_kind",
            values_callable=lambda enum: [
                item.value
                for item in enum
            ],
        ),
        nullable=False,
    )

    status: Mapped[AgentStatus] = mapped_column(
        Enum(
            AgentStatus,
            name="agent_status",
            values_callable=lambda enum: [
                item.value
                for item in enum
            ],
        ),
        nullable=False,
        default=AgentStatus.ACTIVE,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    can_spawn_subagents: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index(
            "uq_agents_primary_per_user",
            "user_id",
            unique=True,
            postgresql_where=text(
                "kind = 'primary'"
            ),
            sqlite_where=text(
                "kind = 'primary'"
            ),
        ),
    )
