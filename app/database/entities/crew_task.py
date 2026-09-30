"""Ordered crew tasks pinned to member agent IDs."""

from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKeyConstraint, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class CrewTaskEntity(Base):
    __tablename__ = "crew_tasks"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    crew_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)
    agent_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    position: Mapped[int] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    expected_output: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["crew_id", "agent_id"], ["crew_agents.crew_id", "crew_agents.agent_id"],
            ondelete="CASCADE", name="fk_crew_tasks_member",
        ),
        UniqueConstraint("crew_id", "position", name="uq_crew_tasks_position"),
        CheckConstraint("position >= 0", name="ck_crew_tasks_position"),
    )
