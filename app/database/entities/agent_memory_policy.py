"""Per-agent memory visibility settings, separate from memory records."""

from uuid import UUID

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class AgentMemoryPolicyEntity(Base):
    __tablename__ = "agent_memory_policies"

    agent_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), primary_key=True
    )
    scope: Mapped[str] = mapped_column(String(32), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False)

    __table_args__ = (
        CheckConstraint("scope IN ('USER_GLOBAL', 'AGENT_PRIVATE')", name="ck_agent_memory_policy_scope"),
    )
