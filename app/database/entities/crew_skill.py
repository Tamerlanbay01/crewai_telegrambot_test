"""Crew-level required skill references; package content stays in Skill storage."""

from uuid import UUID

from sqlalchemy import Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class CrewSkillEntity(Base):
    __tablename__ = "crew_skills"

    crew_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("crews.id", ondelete="CASCADE"), primary_key=True
    )
    skill_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("skills.id", ondelete="RESTRICT"), primary_key=True
    )
    required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
