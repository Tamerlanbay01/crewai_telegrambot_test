"""Ordered membership of persistent user agents in a crew."""

from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class CrewAgentEntity(Base):
    __tablename__ = "crew_agents"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    crew_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("crews.id", ondelete="CASCADE"), nullable=False, index=True
    )
    agent_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("agents.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    position: Mapped[int] = mapped_column(nullable=False)

    __table_args__ = (
        UniqueConstraint("crew_id", "agent_id", name="uq_crew_agents_member"),
        UniqueConstraint("crew_id", "position", name="uq_crew_agents_position"),
        CheckConstraint("position >= 0", name="ck_crew_agents_position"),
    )
