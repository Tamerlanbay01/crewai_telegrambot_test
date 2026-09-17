from datetime import datetime 
from uuid import UUID
from sqlalchemy import DateTime, ForeignKey, UniqueConstraint, CheckConstraint, func
from sqlaclhemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base

class AgentConnectionEntity(Base):
    __tablename__ = "AgentConnections"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid = True), primary_key = True)
    parent_agent_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid = True), 
        ForeignKey("agents.id", ondelete = "CASCADE"), 
        nullable = False, 
        index = True
    )
    child_agent_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "agents.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )
